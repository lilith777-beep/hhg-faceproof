from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faceproof.copydetect import SSCD, SSCDDescriptorEngine, load_sscd_engine
from faceproof.errors import FaceInputError


def test_sscd_model_is_pinned() -> None:
    assert SSCD.filename == "sscd_disc_mixup.torchscript.pt"
    assert SSCD.dimensions == 512
    assert len(SSCD.sha256) == 64
    int(SSCD.sha256, 16)


def test_sscd_auto_mode_has_explicit_legacy_fallback(tmp_path: Path) -> None:
    assert load_sscd_engine(tmp_path, mode="auto", device="cpu", batch_size=1) is None


def test_sscd_required_mode_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FaceInputError, match="SSCD model is missing"):
        load_sscd_engine(tmp_path, mode="required", device="cpu", batch_size=1)


def test_sscd_batches_the_pinned_preprocessing_recipe() -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")

    class RecordingModel:
        def __init__(self) -> None:
            self.batches: list[object] = []

        def __call__(self, batch: object) -> object:
            self.batches.append(batch.detach().cpu().clone())
            return batch.reshape(batch.shape[0], -1)[:, : SSCD.dimensions]

    model = RecordingModel()
    engine = object.__new__(SSCDDescriptorEngine)
    engine.cv2 = cv2
    engine.torch = torch
    engine.device = "cpu"
    engine.batch_size = 2
    engine.model = model
    engine._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    engine._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    images = [
        np.full((20 + index, 30 + index, 3), (17 + index, 83, 211), dtype=np.uint8)
        for index in range(5)
    ]
    descriptors = engine.encode_many(images)

    assert [batch.shape[0] for batch in model.batches] == [2, 2, 1]
    assert len(descriptors) == len(images)
    assert all(row.dtype == np.float32 and row.shape == (SSCD.dimensions,) for row in descriptors)
    np.testing.assert_allclose(
        np.linalg.norm(np.stack(descriptors), axis=1), np.ones(len(images)), atol=1e-6
    )
    expected_rgb = cv2.cvtColor(
        cv2.resize(images[0], (320, 320), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )
    expected = torch.from_numpy(expected_rgb).permute(2, 0, 1).to(torch.float32).div_(255.0)
    expected.sub_(engine._mean[0]).div_(engine._std[0])
    torch.testing.assert_close(model.batches[0][0], expected)


def test_sscd_rejects_empty_input_before_partial_inference() -> None:
    torch = pytest.importorskip("torch")

    class UnexpectedModel:
        def __call__(self, _batch: object) -> object:
            raise AssertionError("model must not run for an invalid input list")

    engine = object.__new__(SSCDDescriptorEngine)
    engine.torch = torch
    engine.batch_size = 1
    engine.model = UnexpectedModel()
    with pytest.raises(FaceInputError, match="empty image"):
        engine.encode_many([np.ones((2, 2, 3), dtype=np.uint8), None])
