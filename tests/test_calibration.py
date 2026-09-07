from pathlib import Path

import numpy as np
import pytest

from faceproof.calibration import calibrate_copy, choose_threshold
from faceproof.errors import BlockedState


def test_calibration_separates_known_scores() -> None:
    result = choose_threshold([0.71, 0.74, 0.79], [0.05, 0.12, 0.2, 0.3, 0.41, 0.49])
    assert 0.49 < result.threshold < 0.71
    assert result.balanced_accuracy == 1.0
    assert result.false_accept_rate == 0.0
    assert result.false_reject_rate == 0.0


def test_calibration_requires_minimum_validation_set() -> None:
    with pytest.raises(
        BlockedState,
        match=r"BLOCKED_REAL_CALIBRATION_DATA.*at least 2 positive and 5 negative",
    ):
        choose_threshold([0.7], [0.1, 0.2])


def test_calibration_prefers_lower_false_accept_rate_on_tie() -> None:
    result = choose_threshold([0.6, 0.8], [0.1, 0.2, 0.3, 0.7, 0.9])
    assert result.false_accept_rate <= 0.4


class _CopyEngine:
    model_id = "test-copy-model"
    model_sha256 = "a" * 64
    dimensions = 2
    device = "cpu"

    @staticmethod
    def encode(value: bytes) -> np.ndarray:
        return _CopyEngine._descriptor(value)

    @staticmethod
    def encode_many(values: list[bytes]) -> list[np.ndarray]:
        return [_CopyEngine._descriptor(value) for value in values]

    @staticmethod
    def _descriptor(value: bytes) -> np.ndarray:
        if value.startswith(b"positive") or value == b"reference":
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


def test_copy_calibration_keeps_copy_labels_separate_from_identity(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jpg"
    positives = tmp_path / "copies"
    negatives = tmp_path / "unrelated"
    positives.mkdir()
    negatives.mkdir()
    reference.write_bytes(b"reference")
    for index in range(2):
        (positives / f"copy-{index}.jpg").write_bytes(f"positive-{index}".encode())
    for index in range(5):
        (negatives / f"negative-{index}.jpg").write_bytes(f"negative-{index}".encode())

    report = calibrate_copy(
        _CopyEngine(),
        lambda content: content,
        reference,
        positives,
        negatives,
        tmp_path / "copy-calibration.json",
    )

    assert report["schema"] == "faceproof.copy-calibration.v1"
    assert report["positive_count"] == 2
    assert report["negative_count"] == 5
    assert report["metrics"]["balanced_accuracy"] == 1.0
    assert "same-person photos are not copy-positive" in report["warning"]
