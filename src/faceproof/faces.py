from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .errors import FaceInputError
from .models import BoundingBox, FaceObservation


@dataclass(frozen=True, slots=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str


YUNET = ModelSpec(
    filename="face_detection_yunet_2023mar.onnx",
    url=(
        "https://raw.githubusercontent.com/opencv/opencv_zoo/"
        "47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
SFACE = ModelSpec(
    filename="face_recognition_sface_2021dec.onnx",
    url=(
        "https://raw.githubusercontent.com/opencv/opencv_zoo/"
        "47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    ),
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)
MODEL_SPECS = (YUNET, SFACE)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(model_dir: Path, *, timeout_s: float = 120.0) -> list[Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for spec in MODEL_SPECS:
            destination = model_dir / spec.filename
            if destination.exists() and _file_sha256(destination) == spec.sha256:
                installed.append(destination)
                continue

            temporary = destination.with_suffix(destination.suffix + ".download")
            try:
                with client.stream("GET", spec.url) as response:
                    response.raise_for_status()
                    digest = hashlib.sha256()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_bytes():
                            digest.update(chunk)
                            output.write(chunk)
                if digest.hexdigest() != spec.sha256:
                    raise FaceInputError(f"checksum mismatch while downloading {spec.filename}")
                temporary.replace(destination)
                installed.append(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return installed


class FaceEngine:
    detector_id = "opencv-yunet-2023mar"
    encoder_id = "opencv-sface-2021dec"
    detector_sha256 = YUNET.sha256
    encoder_sha256 = SFACE.sha256

    def __init__(self, model_dir: Path, *, detection_threshold: float = 0.80) -> None:
        # OpenCV checks this bound from image headers before allocating decoded pixels.
        os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "40000000")
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise FaceInputError("OpenCV is not installed; run pip install -e .") from exc

        self.cv2 = cv2
        detector_path = model_dir / YUNET.filename
        recognizer_path = model_dir / SFACE.filename
        missing = [str(path) for path in (detector_path, recognizer_path) if not path.exists()]
        if missing:
            raise FaceInputError("face models are missing; run `faceproof models install`")
        for path, spec in ((detector_path, YUNET), (recognizer_path, SFACE)):
            if _file_sha256(path) != spec.sha256:
                raise FaceInputError(f"model checksum mismatch: {path.name}")

        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), detection_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def decode(self, content: bytes) -> Any:
        if not content:
            raise FaceInputError("input image is empty")
        image = self.cv2.imdecode(np.frombuffer(content, dtype=np.uint8), self.cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FaceInputError("input is not a decodable image")
        height, width = image.shape[:2]
        if width < 80 or height < 80:
            raise FaceInputError("image is too small; use at least 80 x 80 pixels")
        if width * height > 40_000_000:
            raise FaceInputError("image exceeds the 40 megapixel safety limit")
        return image

    def detect_and_encode(self, image: Any) -> list[FaceObservation]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        observations: list[FaceObservation] = []
        for face in faces:
            aligned = self.recognizer.alignCrop(image, face)
            raw_feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(raw_feature))
            if norm <= 0 or not np.all(np.isfinite(raw_feature)):
                continue
            embedding = raw_feature / norm
            if embedding.shape != (128,) or not np.all(np.isfinite(embedding)):
                continue
            x, y, box_width, box_height = (round(float(value)) for value in face[:4])
            observations.append(
                FaceObservation(
                    box=BoundingBox(
                        x=max(0, x),
                        y=max(0, y),
                        width=max(1, box_width),
                        height=max(1, box_height),
                        confidence=float(face[-1]),
                    ),
                    embedding=embedding,
                    landmarks=tuple(
                        (float(face[offset]), float(face[offset + 1])) for offset in range(4, 14, 2)
                    ),
                )
            )
        return observations

    def require_single_query_face(self, image: Any) -> FaceObservation:
        faces = self.detect_and_encode(image)
        if not faces:
            raise FaceInputError("no face detected in the input image")
        if len(faces) != 1:
            raise FaceInputError(
                f"expected exactly one query face, found {len(faces)}; crop the intended face first"
            )
        face = faces[0]
        self._require_query_quality(image, face)
        return face

    def select_query_face(self, image: Any, face_index: int | None = None) -> FaceObservation:
        faces = self.detect_and_encode(image)
        if not faces:
            raise FaceInputError("no face detected in the input image")
        if len(faces) > 1 and face_index is None:
            raise FaceInputError(
                f"found {len(faces)} faces; explicitly select an enrollment face index"
            )
        selected = 0 if face_index is None else face_index
        if selected < 0 or selected >= len(faces):
            raise FaceInputError(f"enrollment face index {selected} is out of range")
        face = faces[selected]
        self._require_query_quality(image, face)
        return face

    def _require_query_quality(self, image: Any, face: FaceObservation) -> None:
        box = face.box
        if min(box.width, box.height) < 80:
            raise FaceInputError(
                "detected face is too small; use a closer or higher-resolution scan"
            )
        crop = image[
            box.y : box.y + box.height,
            box.x : box.x + box.width,
        ]
        if crop.size == 0:
            raise FaceInputError("detected face crop is empty")
        gray = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < 25:
            raise FaceInputError("face is too dark for reliable matching")
        if brightness > 235:
            raise FaceInputError("face is overexposed for reliable matching")
        sharpness = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        if sharpness < 30:
            raise FaceInputError("face is too blurred for reliable matching")

    def crop_as_jpeg(self, image: Any, face: FaceObservation, *, padding: float = 0.35) -> bytes:
        height, width = image.shape[:2]
        box = face.box
        pad_x = round(box.width * padding)
        pad_y = round(box.height * padding)
        x1 = max(0, box.x - pad_x)
        y1 = max(0, box.y - pad_y)
        x2 = min(width, box.x + box.width + pad_x)
        y2 = min(height, box.y + box.height + pad_y)
        crop = image[y1:y2, x1:x2]
        ok, encoded = self.cv2.imencode(".jpg", crop, [self.cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise FaceInputError("could not encode the detected face crop")
        return encoded.tobytes()

    @staticmethod
    def best_match(
        query: FaceObservation, candidates: list[FaceObservation]
    ) -> tuple[float, int | None, BoundingBox | None]:
        if not candidates:
            return -1.0, None, None
        scored = [
            (float(np.dot(query.embedding, candidate.embedding)), index, candidate.box)
            for index, candidate in enumerate(candidates)
        ]
        return max(scored, key=lambda item: item[0])
