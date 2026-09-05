from pathlib import Path

import cv2
import numpy as np
import pytest

from faceproof.errors import FaceInputError
from faceproof.face_analysis import (
    PARSER_FILENAME,
    POSE_TEMPLATE,
    AnalysisState,
    FaceParser,
    FaceQualityAssessor,
    Visibility,
    estimate_pose,
)
from faceproof.models import BoundingBox, FaceObservation


def _face(points: np.ndarray) -> FaceObservation:
    return FaceObservation(
        BoundingBox(100, 100, 300, 300, 0.95),
        np.array([1.0, 0.0], dtype=np.float32),
        tuple((float(x), float(y)) for x, y in points),
    )


def _rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    x, y, z = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_five_point_pose_recovers_synthetic_rotation_convention() -> None:
    size = 1000
    camera = np.array([[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]], np.float64)
    rotation = _rotation(-8, 12, 5)
    rvec, _ = cv2.Rodrigues(rotation)
    points, _ = cv2.projectPoints(
        POSE_TEMPLATE, rvec, np.array([[0.0], [0.0], [45.0]]), camera, np.zeros((4, 1))
    )

    result = estimate_pose((size, size, 3), _face(points.reshape(-1, 2)), cv2)

    assert result.state == AnalysisState.ASSESSABLE
    assert result.pitch_deg == pytest.approx(-8, abs=0.5)
    assert result.yaw_deg == pytest.approx(12, abs=0.5)
    assert result.roll_deg == pytest.approx(5, abs=0.5)
    assert result.normalized_reprojection_error < 0.01


def test_degenerate_pose_is_unknown() -> None:
    result = estimate_pose((500, 500, 3), _face(np.ones((5, 2)) * 100), cv2)
    assert result.state == AnalysisState.UNKNOWN
    assert result.coarse_bin == "UNKNOWN"


def test_missing_parser_weights_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FaceInputError, match="parser is missing"):
        FaceParser(tmp_path, cv2)
    assert not (tmp_path / PARSER_FILENAME).exists()


def test_unvalidated_quality_components_force_review() -> None:
    points = np.array([[160, 180], [260, 180], [210, 230], [175, 285], [245, 285]])
    image = np.full((500, 500, 3), 128, dtype=np.uint8)
    quality = FaceQualityAssessor(cv2, None).assess(image, _face(points))

    assert quality.state == AnalysisState.NEEDS_REVIEW
    assert "PARSING_UNAVAILABLE" in quality.reasons
    assert "POSE_POLICY_UNVALIDATED" in quality.reasons
    assert quality.parsing is None


def test_eye_glasses_parser_label_never_becomes_automatic_clear_evidence() -> None:
    points = np.array([[160, 180], [260, 180], [210, 230], [175, 285], [245, 285]])
    probabilities = np.zeros((19, 512, 512), dtype=np.float32)
    probabilities[6] = 1.0
    regions = FaceParser._region_evidence(probabilities, _face(points), (0, 0, 500, 500))
    assert regions["eyes"].visibility == Visibility.UNKNOWN
