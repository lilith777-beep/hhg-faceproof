from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .errors import FaceInputError
from .models import FaceObservation

PARSER_FILENAME = "face_parsing_bisenet_resnet18.onnx"
PARSER_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
PARSER_SHA256 = "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
PARSER_MODEL_ID = "yakhyo-bisenet-resnet18-face-parsing"
PARSER_REVISION = "8a4729d95118d0e97c44185f9bdef3d6bfeaaf99"

# Derived from the pinned MediaPipe canonical face OBJ. YuNet order is right eye, left eye,
# nose, right mouth corner, left mouth corner. See model-lock.json for vertex mapping/source.
POSE_TEMPLATE = np.array(
    [
        [-3.16416575, 2.62690925, 3.64600075],
        [3.16416575, 2.62690925, 3.64600075],
        [0.0, -1.126865, 7.475604],
        [-2.456206, -4.342621, 4.283884],
        [2.456206, -4.342621, 4.283884],
    ],
    dtype="<f8",
)
POSE_TEMPLATE_SHA256 = "ab24547cfd9b636b0fb8433bd972cf4989e1b79ddefad57a9d8edde3ffc4f8ff"
assert hashlib.sha256(POSE_TEMPLATE.tobytes()).hexdigest() == POSE_TEMPLATE_SHA256


class Visibility(StrEnum):
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    UNKNOWN = "UNKNOWN"


class AnalysisState(StrEnum):
    ASSESSABLE = "ASSESSABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RegionEvidence:
    support: float | None
    confidence: float | None
    visibility: Visibility
    reason: str
    score_interpretation: str = "uncalibrated mean softmax support; not a probability"


@dataclass(frozen=True, slots=True)
class ParsingResult:
    state: AnalysisState
    regions: dict[str, RegionEvidence]
    crop_xyxy: tuple[int, int, int, int]
    semantic_mask: Any = field(repr=False, compare=False)
    model_id: str = PARSER_MODEL_ID
    model_sha256: str = PARSER_SHA256
    preprocessing: str = "RGB; INTER_LINEAR 512x512; ImageNet normalization"

    def evidence(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("semantic_mask", None)
        return value


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    state: AnalysisState
    yaw_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None
    coarse_bin: str
    normalized_reprojection_error: float | None
    convention: str = (
        "OpenCV camera x-right/y-down/z-forward; reported Euler x=pitch, y=yaw, z=roll; "
        "generic intrinsics and person geometry"
    )
    template_sha256: str = POSE_TEMPLATE_SHA256


@dataclass(frozen=True, slots=True)
class FaceQuality:
    state: AnalysisState
    quality_score: float
    reasons: tuple[str, ...]
    native_face_min_px: int
    interocular_px: float | None
    detector_score: float
    blur_variance: float
    mean_luminance: float
    dark_clip_fraction: float
    bright_clip_fraction: float
    components: dict[str, float | None]
    pose: PoseEstimate
    parsing: ParsingResult | None

    def evidence(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "quality_score": self.quality_score,
            "reasons": list(self.reasons),
            "native_face_min_px": self.native_face_min_px,
            "interocular_px": self.interocular_px,
            "detector_score": self.detector_score,
            "blur_variance": self.blur_variance,
            "mean_luminance": self.mean_luminance,
            "dark_clip_fraction": self.dark_clip_fraction,
            "bright_clip_fraction": self.bright_clip_fraction,
            "components": self.components,
            "pose": asdict(self.pose),
            "parsing": self.parsing.evidence() if self.parsing else None,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_parser_model(model_dir: Path, *, timeout_s: float = 300.0) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / PARSER_FILENAME
    if destination.is_file() and _sha256(destination) == PARSER_SHA256:
        return destination
    temporary = destination.with_suffix(".onnx.download")
    try:
        with httpx.stream("GET", PARSER_URL, timeout=timeout_s, follow_redirects=True) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != PARSER_SHA256:
            raise FaceInputError("checksum mismatch while downloading the face parser")
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


class FaceParser:
    """Pinned BiSeNet parser through the existing OpenCV DNN runtime."""

    def __init__(self, model_dir: Path, cv2: Any) -> None:
        path = model_dir / PARSER_FILENAME
        if not path.is_file():
            raise FaceInputError("face parser is missing; run `faceproof models install`")
        if _sha256(path) != PARSER_SHA256:
            raise FaceInputError("model checksum mismatch: face parser")
        self.cv2 = cv2
        try:
            self.net = cv2.dnn.readNetFromONNX(str(path))
        except Exception as exc:
            raise FaceInputError("OpenCV could not load the pinned face parser") from exc
        self.output_names = tuple(self.net.getUnconnectedOutLayersNames())
        if not self.output_names:
            raise FaceInputError("face parser exposes no output layers")

    @staticmethod
    def _crop(image: Any, face: FaceObservation) -> tuple[Any, tuple[int, int, int, int]]:
        height, width = _validated_image(image)
        box_x, box_y, box_width, box_height, _ = _validated_box(face)
        pad_x = round(box_width * 0.35)
        pad_y = round(box_height * 0.35)
        x1 = max(0, box_x - pad_x)
        y1 = max(0, box_y - pad_y)
        x2 = min(width, box_x + box_width + pad_x)
        y2 = min(height, box_y + box_height + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise FaceInputError("face parser crop is empty")
        return crop, (x1, y1, x2, y2)

    def parse(self, image: Any, face: FaceObservation) -> ParsingResult:
        crop, crop_xyxy = self._crop(image, face)
        landmarks = _valid_landmarks(face, crop_xyxy=crop_xyxy)
        if landmarks is None:
            return ParsingResult(
                AnalysisState.UNKNOWN,
                self._unknown_regions("five finite YuNet landmarks inside parser crop unavailable"),
                crop_xyxy,
                None,
            )
        try:
            rgb = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2RGB)
            resized = self.cv2.resize(rgb, (512, 512), interpolation=self.cv2.INTER_LINEAR)
            tensor = resized.astype(np.float32) / 255.0
            tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )
            tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)
            self.net.setInput(tensor)
            outputs = self.net.forward(self.output_names)
            first_output = outputs[0] if isinstance(outputs, tuple | list) and outputs else outputs
            logits = np.asarray(first_output, dtype=np.float32)
        except Exception as exc:
            raise FaceInputError("face parser inference failed") from exc
        if logits.shape != (1, 19, 512, 512) or not np.all(np.isfinite(logits)):
            raise FaceInputError(f"face parser returned invalid output shape {logits.shape}")
        logits = logits[0].copy()
        logits -= logits.max(axis=0, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        mask_512 = probabilities.argmax(axis=0).astype(np.uint8)
        mask = self.cv2.resize(
            mask_512, (crop.shape[1], crop.shape[0]), interpolation=self.cv2.INTER_NEAREST
        )
        regions = self._region_evidence(probabilities, face, crop_xyxy)
        state = (
            AnalysisState.ASSESSABLE
            if all(item.visibility == Visibility.VISIBLE for item in regions.values())
            else AnalysisState.NEEDS_REVIEW
        )
        return ParsingResult(state, regions, crop_xyxy, mask)

    @staticmethod
    def _unknown_regions(reason: str) -> dict[str, RegionEvidence]:
        return {
            name: RegionEvidence(None, None, Visibility.UNKNOWN, reason)
            for name in ("eyes", "nose", "mouth")
        }

    @staticmethod
    def _region_evidence(
        probabilities: np.ndarray,
        face: FaceObservation,
        crop_xyxy: tuple[int, int, int, int],
    ) -> dict[str, RegionEvidence]:
        try:
            probabilities = np.asarray(probabilities)
            valid_probabilities = (
                probabilities.shape == (19, 512, 512)
                and np.all(np.isfinite(probabilities))
                and not np.any(probabilities < 0.0)
                and not np.any(probabilities > 1.0)
                and np.allclose(probabilities.sum(axis=0), 1.0, rtol=1e-5, atol=1e-6)
            )
        except (TypeError, ValueError):
            valid_probabilities = False
        if not valid_probabilities:
            raise FaceInputError("face parser probabilities are malformed")
        try:
            crop = np.asarray(crop_xyxy, dtype=np.float64)
        except (TypeError, ValueError):
            crop = np.array([], dtype=np.float64)
        if (
            crop.shape != (4,)
            or not np.all(np.isfinite(crop))
            or not np.all(crop == np.floor(crop))
            or crop[2] <= crop[0]
            or crop[3] <= crop[1]
        ):
            return FaceParser._unknown_regions("parser crop transform unavailable")
        landmarks = _valid_landmarks(face, crop_xyxy=crop_xyxy)
        if landmarks is None:
            return FaceParser._unknown_regions(
                "five finite YuNet landmarks inside parser crop unavailable"
            )
        x1, y1, x2, y2 = crop_xyxy
        scale_x = 512 / (x2 - x1)
        scale_y = 512 / (y2 - y1)
        landmark_groups = {
            "eyes": (landmarks[0], landmarks[1]),
            "nose": (landmarks[2],),
            "mouth": (landmarks[3], landmarks[4]),
        }
        # CelebAMask-HQ class 6 is eye-glasses. It is deliberately not accepted as clear
        # eye evidence: the parser cannot distinguish transparent glasses from sunglasses.
        expected_classes = {"eyes": (4, 5), "nose": (10,), "mouth": (11, 12, 13)}
        results: dict[str, RegionEvidence] = {}
        yy, xx = np.ogrid[:512, :512]
        radius = max(5.0, min(40.0, face.box.width * scale_x * 0.08))
        predicted = probabilities.argmax(axis=0)
        for name, points in landmark_groups.items():
            point_support: list[float] = []
            point_confidence: list[float] = []
            for point_x, point_y in points:
                center_x = (point_x - x1) * scale_x
                center_y = (point_y - y1) * scale_y
                area = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
                classes = expected_classes[name]
                point_support.append(float(np.isin(predicted[area], classes).mean()))
                point_confidence.append(
                    float(probabilities[list(classes)][:, area].sum(axis=0).mean())
                )
            # A paired region is only as supported as its weaker landmark. Combining both
            # disks would allow one visible eye or mouth corner to conceal an unsupported one.
            support = min(point_support)
            confidence = min(point_confidence)
            if support >= 0.25 and confidence >= 0.50:
                visibility = Visibility.VISIBLE
                reason = "expected parser labels support every region landmark"
            else:
                # The 19-class parser has no universal hand/sticker/object-occluder class.
                visibility = Visibility.UNKNOWN
                reason = (
                    "insufficient uncalibrated parser support; arbitrary occlusion is not "
                    "inferable"
                )
            results[name] = RegionEvidence(
                round(support, 6), round(confidence, 6), visibility, reason
            )
        return results


def _validated_image(image: Any) -> tuple[int, int]:
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise FaceInputError("face analysis image is malformed") from exc
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0 or array.dtype != np.uint8:
        raise FaceInputError("face analysis requires a non-empty uint8 BGR image")
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise FaceInputError("face analysis image dimensions are invalid")
    return int(height), int(width)


def _validated_box(face: FaceObservation) -> tuple[int, int, int, int, float]:
    try:
        values = np.asarray(
            [face.box.x, face.box.y, face.box.width, face.box.height, face.box.confidence],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise FaceInputError("face analysis box is malformed") from exc
    if (
        not np.all(np.isfinite(values))
        or not np.all(values[:4] == np.floor(values[:4]))
        or values[2] <= 0
        or values[3] <= 0
    ):
        raise FaceInputError("face analysis box is malformed")
    return (
        int(values[0]),
        int(values[1]),
        int(values[2]),
        int(values[3]),
        float(values[4]),
    )


def _valid_landmarks(
    face: FaceObservation,
    *,
    image_size: tuple[int, int] | None = None,
    crop_xyxy: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    try:
        points = np.asarray(face.landmarks, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        return None
    if image_size is not None:
        height, width = image_size
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
            return None
        if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
            return None
    if crop_xyxy is not None:
        x1, y1, x2, y2 = crop_xyxy
        if x2 <= x1 or y2 <= y1:
            return None
        if np.any(points[:, 0] < x1) or np.any(points[:, 0] >= x2):
            return None
        if np.any(points[:, 1] < y1) or np.any(points[:, 1] >= y2):
            return None
    return points


def estimate_pose(image_shape: tuple[int, ...], face: FaceObservation, cv2: Any) -> PoseEstimate:
    unknown = PoseEstimate(AnalysisState.UNKNOWN, None, None, None, "UNKNOWN", None)
    try:
        if len(image_shape) < 2:
            return unknown
        dimensions = np.asarray(image_shape[:2], dtype=np.float64)
    except (TypeError, ValueError):
        return unknown
    if (
        not np.all(np.isfinite(dimensions))
        or not np.all(dimensions == np.floor(dimensions))
        or np.any(dimensions <= 0)
    ):
        return unknown
    height, width = (int(value) for value in dimensions)
    image_points = _valid_landmarks(face, image_size=(int(height), int(width)))
    if image_points is None:
        return unknown
    interocular = float(np.linalg.norm(image_points[0] - image_points[1]))
    centered = image_points - image_points.mean(axis=0)
    if (
        interocular < 5.0
        or len(np.unique(image_points, axis=0)) < 4
        or np.linalg.matrix_rank(centered) < 2
    ):
        return unknown
    focal = float(max(width, height))
    camera = np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    try:
        ok, rotation_vector, translation_vector = cv2.solvePnP(
            POSE_TEMPLATE,
            image_points,
            camera,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return unknown
        rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
        translation_vector = np.asarray(translation_vector, dtype=np.float64)
        if (
            rotation_vector.size != 3
            or translation_vector.size != 3
            or not np.all(np.isfinite(rotation_vector))
            or not np.all(np.isfinite(translation_vector))
        ):
            return unknown
        rotation_vector = rotation_vector.reshape(3, 1)
        translation_vector = translation_vector.reshape(3, 1)
        rotation, _ = cv2.Rodrigues(rotation_vector)
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            return unknown
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            return unknown
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            return unknown
        camera_points = (rotation @ POSE_TEMPLATE.T + translation_vector).T
        if not np.all(np.isfinite(camera_points)) or np.any(camera_points[:, 2] <= 0.0):
            return unknown
        angles = cv2.RQDecomp3x3(rotation)[0]
        projected, _ = cv2.projectPoints(
            POSE_TEMPLATE,
            rotation_vector,
            translation_vector,
            camera,
            np.zeros((4, 1), dtype=np.float64),
        )
    except Exception:
        return unknown
    angles = np.asarray(angles, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    if angles.shape != (3,) or projected.size != 10 or not np.all(np.isfinite(projected)):
        return unknown
    pitch, yaw, roll = (float(value) for value in angles)
    residual = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
        / interocular
    )
    values = np.array([pitch, yaw, roll, residual])
    if not np.all(np.isfinite(values)) or residual > 0.25:
        return unknown
    if abs(yaw) <= 15 and abs(pitch) <= 15:
        coarse_bin = "FRONTAL"
    elif abs(yaw) <= 30 and abs(pitch) <= 30:
        coarse_bin = "MODERATE"
    else:
        coarse_bin = "EXTREME"
    return PoseEstimate(
        AnalysisState.ASSESSABLE,
        round(yaw, 3),
        round(pitch, 3),
        round(roll, 3),
        coarse_bin,
        round(residual, 6),
    )


class FaceQualityAssessor:
    """Deterministic raw quality features; validation flags control decision eligibility."""

    def __init__(
        self,
        cv2: Any,
        parser: FaceParser | None,
        *,
        pose_policy_validated: bool = False,
        visibility_policy_validated: bool = False,
    ) -> None:
        self.cv2 = cv2
        self.parser = parser
        self.pose_policy_validated = pose_policy_validated
        self.visibility_policy_validated = visibility_policy_validated

    def assess(self, image: Any, face: FaceObservation) -> FaceQuality:
        height, width = _validated_image(image)
        box_x, box_y, box_width, box_height, detector_confidence = _validated_box(face)
        x1 = max(0, box_x)
        y1 = max(0, box_y)
        x2 = min(width, box_x + box_width)
        y2 = min(height, box_y + box_height)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise FaceInputError("quality crop is empty")
        fixed = self.cv2.resize(crop, (160, 160), interpolation=self.cv2.INTER_AREA)
        gray = self.cv2.cvtColor(fixed, self.cv2.COLOR_BGR2GRAY)
        blur = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        luminance = float(gray.mean())
        dark_clip = float((gray <= 5).mean())
        bright_clip = float((gray >= 250).mean())
        raw_metrics = np.asarray([blur, luminance, dark_clip, bright_clip], dtype=np.float64)
        if not np.all(np.isfinite(raw_metrics)):
            raise FaceInputError("quality feature extraction returned non-finite evidence")
        landmarks = _valid_landmarks(face, image_size=(height, width))
        interocular = (
            float(np.linalg.norm(landmarks[0] - landmarks[1]))
            if landmarks is not None
            else None
        )
        pose = estimate_pose(image.shape, face, self.cv2)
        parsing: ParsingResult | None = None
        parser_error = False
        if self.parser is not None:
            try:
                parsing = self.parser.parse(image, face)
            except Exception:
                parser_error = True
                parsing = ParsingResult(
                    AnalysisState.UNKNOWN,
                    FaceParser._unknown_regions("face parser inference failed"),
                    (x1, y1, x2, y2),
                    None,
                )

        visible_width = x2 - x1
        visible_height = y2 - y1
        components: dict[str, float | None] = {
            "native_resolution": min(1.0, min(visible_width, visible_height) / 120.0),
            "interocular_resolution": min(1.0, interocular / 45.0) if interocular else None,
            "detector": max(0.0, min(1.0, (detector_confidence - 0.5) / 0.5)),
            "blur": min(1.0, blur / 100.0),
            "exposure": max(
                0.0,
                min(1.0, luminance / 55.0, (255.0 - luminance) / 55.0)
                * (1.0 - min(1.0, dark_clip + bright_clip)),
            ),
            "pose": 1.0 if pose.state == AnalysisState.ASSESSABLE else None,
            "visibility": (1.0 if parsing and parsing.state == AnalysisState.ASSESSABLE else None),
        }
        measured = [value for value in components.values() if value is not None]
        quality_score = round(min(measured, default=0.0), 6)
        reasons: list[str] = []
        if components["native_resolution"] < 0.67:
            reasons.append("LOW_FACE_RESOLUTION")
        if components["blur"] < 0.30:
            reasons.append("BLUR")
        if components["exposure"] < 0.35:
            reasons.append("EXPOSURE")
        if pose.state == AnalysisState.UNKNOWN:
            reasons.append("POSE_UNKNOWN")
        if not self.pose_policy_validated:
            reasons.append("POSE_POLICY_UNVALIDATED")
        if parser_error or parsing is None:
            reasons.append("PARSING_UNAVAILABLE")
        elif parsing.state != AnalysisState.ASSESSABLE:
            reasons.append("REGION_VISIBILITY_UNKNOWN")
        if not self.visibility_policy_validated:
            reasons.append("VISIBILITY_POLICY_UNVALIDATED")
        state = (
            AnalysisState.ASSESSABLE
            if not reasons and quality_score >= 0.35
            else AnalysisState.NEEDS_REVIEW
        )
        return FaceQuality(
            state,
            quality_score,
            tuple(reasons),
            min(visible_width, visible_height),
            round(interocular, 3) if interocular is not None else None,
            round(detector_confidence, 6),
            round(blur, 6),
            round(luminance, 6),
            round(dark_clip, 6),
            round(bright_clip, 6),
            components,
            pose,
            parsing,
        )
