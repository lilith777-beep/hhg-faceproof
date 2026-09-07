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
    ParsingResult,
    RegionEvidence,
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


def _project_pose(
    pitch: float,
    yaw: float,
    roll: float,
    *,
    size: int = 1000,
) -> np.ndarray:
    camera = np.array([[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]], np.float64)
    rvec, _ = cv2.Rodrigues(_rotation(pitch, yaw, roll))
    points, _ = cv2.projectPoints(
        POSE_TEMPLATE, rvec, np.array([[0.0], [0.0], [45.0]]), camera, np.zeros((4, 1))
    )
    return points.reshape(-1, 2)


def test_five_point_pose_recovers_synthetic_rotation_convention() -> None:
    size = 1000
    points = _project_pose(-8, 12, 5, size=size)

    result = estimate_pose((size, size, 3), _face(points), cv2)

    assert result.state == AnalysisState.ASSESSABLE
    assert result.pitch_deg == pytest.approx(-8, abs=0.5)
    assert result.yaw_deg == pytest.approx(12, abs=0.5)
    assert result.roll_deg == pytest.approx(5, abs=0.5)
    assert result.normalized_reprojection_error < 0.01


@pytest.mark.parametrize(
    ("pitch", "yaw", "expected_bin"),
    [(-25, -20, "MODERATE"), (0, 0, "FRONTAL"), (10, 35, "EXTREME")],
)
def test_pose_sign_and_coarse_bin_convention(
    pitch: float, yaw: float, expected_bin: str
) -> None:
    result = estimate_pose((1000, 1000, 3), _face(_project_pose(pitch, yaw, -7)), cv2)

    assert result.state == AnalysisState.ASSESSABLE
    assert result.pitch_deg == pytest.approx(pitch, abs=0.5)
    assert result.yaw_deg == pytest.approx(yaw, abs=0.5)
    assert result.roll_deg == pytest.approx(-7, abs=0.5)
    assert result.coarse_bin == expected_bin


def test_pose_is_invariant_to_uniform_image_scaling() -> None:
    points = _project_pose(9, -18, 4)
    baseline = estimate_pose((1000, 1000, 3), _face(points), cv2)
    scaled = estimate_pose((2000, 2000, 3), _face(points * 2), cv2)

    assert scaled.state == baseline.state == AnalysisState.ASSESSABLE
    assert scaled.pitch_deg == pytest.approx(baseline.pitch_deg, abs=0.01)
    assert scaled.yaw_deg == pytest.approx(baseline.yaw_deg, abs=0.01)
    assert scaled.roll_deg == pytest.approx(baseline.roll_deg, abs=0.01)
    assert scaled.normalized_reprojection_error == pytest.approx(
        baseline.normalized_reprojection_error, abs=1e-6
    )


def test_degenerate_pose_is_unknown() -> None:
    result = estimate_pose((500, 500, 3), _face(np.ones((5, 2)) * 100), cv2)
    assert result.state == AnalysisState.UNKNOWN
    assert result.coarse_bin == "UNKNOWN"


def test_distinct_but_collinear_pose_landmarks_are_unknown() -> None:
    points = np.array([[100, 100], [200, 100], [150, 100], [125, 100], [175, 100]])
    assert estimate_pose((500, 500, 3), _face(points), cv2).state == AnalysisState.UNKNOWN


@pytest.mark.parametrize(
    ("image_shape", "points"),
    [
        ((0, 500, 3), np.ones((5, 2)) * 100),
        ((500,), np.ones((5, 2)) * 100),
        ((500, 500, 3), np.array([[np.nan, 1], [2, 3], [4, 5], [6, 7], [8, 9]])),
        ((500, 500, 3), np.array([[-1, 100], [200, 100], [150, 150], [125, 200], [175, 200]])),
    ],
)
def test_invalid_pose_shape_or_landmarks_are_unknown(
    image_shape: tuple[int, ...], points: np.ndarray
) -> None:
    assert estimate_pose(image_shape, _face(points), cv2).state == AnalysisState.UNKNOWN


def test_pose_rejects_behind_camera_solution() -> None:
    class BehindCamera:
        SOLVEPNP_SQPNP = cv2.SOLVEPNP_SQPNP

        @staticmethod
        def solvePnP(*_args: object, **_kwargs: object) -> tuple[bool, np.ndarray, np.ndarray]:
            return True, np.zeros((3, 1)), np.array([[0.0], [0.0], [-100.0]])

        Rodrigues = staticmethod(cv2.Rodrigues)

    result = estimate_pose((1000, 1000, 3), _face(_project_pose(0, 0, 0)), BehindCamera)
    assert result.state == AnalysisState.UNKNOWN


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


def test_one_supported_eye_cannot_clear_the_paired_eye_region() -> None:
    points = np.array([[160, 180], [260, 180], [210, 230], [175, 285], [245, 285]])
    probabilities = np.zeros((19, 512, 512), dtype=np.float32)
    probabilities[0] = 1.0
    yy, xx = np.ogrid[:512, :512]
    center_x, center_y = points[0] * (512 / 500)
    area = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= 30**2
    probabilities[0, area] = 0.0
    probabilities[4, area] = 1.0

    eyes = FaceParser._region_evidence(probabilities, _face(points), (0, 0, 500, 500))["eyes"]

    assert eyes.support == 0.0
    assert eyes.confidence == 0.0
    assert eyes.visibility == Visibility.UNKNOWN
    assert "not a probability" in eyes.score_interpretation


def test_region_support_rejects_malformed_scores_and_landmark_transform() -> None:
    points = np.array([[160, 180], [260, 180], [210, 230], [175, 285], [245, 285]])
    malformed = np.zeros((19, 512, 512), dtype=np.float32)
    malformed[0, 0, 0] = np.nan
    with pytest.raises(FaceInputError, match="probabilities are malformed"):
        FaceParser._region_evidence(malformed, _face(points), (0, 0, 500, 500))

    outside = points.copy()
    outside[0] = (600, 180)
    valid_scores = np.zeros((19, 512, 512), dtype=np.float32)
    valid_scores[0] = 1.0
    regions = FaceParser._region_evidence(
        valid_scores, _face(outside), (0, 0, 500, 500)
    )
    assert all(region.visibility == Visibility.UNKNOWN for region in regions.values())
    assert all(region.support is None for region in regions.values())


def test_parser_returns_unknown_without_running_model_for_nonfinite_landmarks() -> None:
    class MustNotRun:
        def setInput(self, _tensor: np.ndarray) -> None:
            raise AssertionError("model should not run")

    parser = FaceParser.__new__(FaceParser)
    parser.cv2 = cv2
    parser.net = MustNotRun()
    parser.output_names = ("output",)
    points = np.array([[np.nan, 180], [260, 180], [210, 230], [175, 285], [245, 285]])

    result = parser.parse(np.full((500, 500, 3), 128, dtype=np.uint8), _face(points))

    assert result.state == AnalysisState.UNKNOWN
    assert all(region.visibility == Visibility.UNKNOWN for region in result.regions.values())


def test_quality_uses_visible_crop_resolution_at_image_edge() -> None:
    points = np.array([[460, 460], [490, 460], [475, 475], [465, 490], [485, 490]])
    face = FaceObservation(
        BoundingBox(450, 450, 300, 300, 0.95),
        np.array([1.0, 0.0], dtype=np.float32),
        tuple((float(x), float(y)) for x, y in points),
    )
    image = np.full((500, 500, 3), 128, dtype=np.uint8)

    quality = FaceQualityAssessor(cv2, None).assess(image, face)

    assert quality.native_face_min_px == 50
    assert quality.components["native_resolution"] == pytest.approx(50 / 120)
    assert "LOW_FACE_RESOLUTION" in quality.reasons


def test_quality_rejects_nonfinite_box_evidence() -> None:
    points = np.array([[160, 180], [260, 180], [210, 230], [175, 285], [245, 285]])
    face = FaceObservation(
        BoundingBox(100, 100, 300, 300, float("nan")),
        np.array([1.0, 0.0], dtype=np.float32),
        tuple((float(x), float(y)) for x, y in points),
    )
    with pytest.raises(FaceInputError, match="box is malformed"):
        FaceQualityAssessor(cv2, None).assess(np.zeros((500, 500, 3), dtype=np.uint8), face)


def test_parser_runtime_failure_is_preserved_as_unknown_quality_evidence() -> None:
    class BrokenParser:
        @staticmethod
        def parse(_image: np.ndarray, _face: FaceObservation) -> ParsingResult:
            raise RuntimeError("backend failure")

    points = np.array([[160, 180], [260, 180], [210, 230], [175, 285], [245, 285]])
    image = np.full((500, 500, 3), 128, dtype=np.uint8)
    quality = FaceQualityAssessor(cv2, BrokenParser()).assess(image, _face(points))

    assert quality.state == AnalysisState.NEEDS_REVIEW
    assert quality.parsing is not None
    assert quality.parsing.state == AnalysisState.UNKNOWN
    assert "PARSING_UNAVAILABLE" in quality.reasons


def test_quality_features_are_deterministic() -> None:
    class ClearParser:
        @staticmethod
        def parse(_image: np.ndarray, _face: FaceObservation) -> ParsingResult:
            regions = {
                name: RegionEvidence(1.0, 1.0, Visibility.VISIBLE, "synthetic test")
                for name in ("eyes", "nose", "mouth")
            }
            return ParsingResult(AnalysisState.ASSESSABLE, regions, (0, 0, 500, 500), None)

    points = np.array([[160, 180], [260, 180], [210, 230], [175, 285], [245, 285]])
    image = np.indices((500, 500)).sum(axis=0).astype(np.uint8)
    image = np.repeat(image[:, :, None], 3, axis=2)
    assessor = FaceQualityAssessor(
        cv2,
        ClearParser(),
        pose_policy_validated=True,
        visibility_policy_validated=True,
    )

    first = assessor.assess(image, _face(points))
    second = assessor.assess(image.copy(), _face(points))

    assert first == second
    assert np.isfinite(first.quality_score)
    assert all(value is None or np.isfinite(value) for value in first.components.values())
