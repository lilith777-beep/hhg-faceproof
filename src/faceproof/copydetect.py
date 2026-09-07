from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from .errors import FaceInputError


@dataclass(frozen=True, slots=True)
class CopyModelSpec:
    filename: str
    url: str
    sha256: str
    dimensions: int


SSCD = CopyModelSpec(
    filename="sscd_disc_mixup.torchscript.pt",
    url="https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt",
    sha256="9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56",
    dimensions=512,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_sscd_model(model_dir: Path, *, timeout_s: float = 300.0) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / SSCD.filename
    if destination.exists() and file_sha256(destination) == SSCD.sha256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with httpx.stream("GET", SSCD.url, timeout=timeout_s, follow_redirects=True) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != SSCD.sha256:
            raise FaceInputError("checksum mismatch while downloading the SSCD model")
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


class CopyDescriptorEngine(Protocol):
    model_id: str
    dimensions: int
    model_sha256: str
    device: str

    def encode(self, image: Any) -> np.ndarray: ...

    def encode_many(self, images: list[Any]) -> list[np.ndarray]: ...


class SSCDDescriptorEngine:
    """Pinned SSCD DISC-Mixup inference with the upstream square-320 preprocessing."""

    model_id = "meta-sscd-disc-mixup-resnet50"
    dimensions = SSCD.dimensions

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        path = model_dir / SSCD.filename
        if not path.is_file():
            raise FaceInputError("SSCD model is missing; run `faceproof models install`")
        actual = file_sha256(path)
        if actual != SSCD.sha256:
            raise FaceInputError(f"model checksum mismatch: {path.name}")

        try:
            import cv2
            import torch
        except ImportError as exc:
            raise FaceInputError(
                "SSCD requires the optional vision runtime; run pip install -e '.[dev,vision]'"
            ) from exc
        if device not in {"auto", "cpu", "cuda"}:
            raise FaceInputError("FACEPROOF_SSCD_DEVICE must be auto, cpu, or cuda")
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved == "auto":
            resolved = "cpu"
        if resolved == "cuda" and not torch.cuda.is_available():
            raise FaceInputError("FACEPROOF_SSCD_DEVICE=cuda but CUDA is unavailable")

        self.cv2 = cv2
        self.torch = torch
        self.runtime_version = str(torch.__version__)
        self.device = resolved
        self.batch_size = max(1, batch_size)
        self.model_sha256 = actual
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="`torch.jit.load` is deprecated.*")
                self.model = torch.jit.load(str(path), map_location=resolved).eval()
        except Exception as exc:
            raise FaceInputError(
                "the pinned SSCD model could not be loaded by this PyTorch runtime"
            ) from exc
        if resolved == "cuda":
            self.model = self.model.to(resolved)
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=resolved).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=resolved).view(
            1, 3, 1, 1
        )

    def _batch_tensor(self, images: list[Any]) -> Any:
        """Apply the pinned recipe and transfer one contiguous batch to the device."""
        rgb_batch = np.empty((len(images), 320, 320, 3), dtype=np.uint8)
        for index, image in enumerate(images):
            resized = self.cv2.resize(image, (320, 320), interpolation=self.cv2.INTER_LINEAR)
            self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB, dst=rgb_batch[index])
        # Preserve the model's established contiguous NCHW execution layout. Keeping
        # this as uint8 until after the single batch copy avoids the old per-image
        # float tensors without changing convolution accumulation order.
        tensor = self.torch.from_numpy(rgb_batch).permute(0, 3, 1, 2).contiguous()
        return tensor.to(device=self.device, dtype=self.torch.float32).div_(255.0)

    def encode(self, image: Any) -> np.ndarray:
        return self.encode_many([image])[0]

    def encode_many(self, images: list[Any]) -> list[np.ndarray]:
        if not images:
            return []
        if any(image is None or getattr(image, "size", 0) == 0 for image in images):
            raise FaceInputError("cannot compute SSCD descriptor for an empty image")
        descriptors: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = self._batch_tensor(images[start : start + self.batch_size])
                batch.sub_(self._mean).div_(self._std)
                output = self.model(batch).reshape(batch.shape[0], -1)
                output = self.torch.nn.functional.normalize(output, p=2, dim=1)
                if output.shape[1] != self.dimensions:
                    raise FaceInputError(
                        f"SSCD returned {output.shape[1]} dimensions, expected {self.dimensions}"
                    )
                rows = output.detach().cpu().numpy()
                if not np.all(np.isfinite(rows)):
                    raise FaceInputError("SSCD returned a non-finite descriptor")
                if not np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-4):
                    raise FaceInputError("SSCD returned a non-normalized descriptor")
                descriptors.extend(np.asarray(row, dtype=np.float32) for row in rows)
        return descriptors


def load_sscd_engine(
    model_dir: Path,
    *,
    mode: str,
    device: str,
    batch_size: int,
) -> SSCDDescriptorEngine | None:
    if mode == "off":
        return None
    try:
        return SSCDDescriptorEngine(model_dir, device=device, batch_size=batch_size)
    except FaceInputError:
        if mode == "required":
            raise
        return None
