from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import BlockedState, FaceInputError
from .integrity import write_json


class AxisState(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    BORDERLINE = "BORDERLINE"
    UNASSESSABLE = "UNASSESSABLE"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    policy_id: str
    status: str
    face_threshold: float | None
    copy_threshold: float | None
    face_review_boundary: float | None
    copy_review_boundary: float | None
    max_gallery_media: int
    max_enrollment_references: int
    max_copy_references: int
    model_lock_sha256: str
    calibration_report_sha256: str | None
    test_report_sha256: str | None
    pose_policy_validated: bool
    visibility_policy_validated: bool
    source_sha256: str
    targets: dict[str, float] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return (
            self.status in {"CALIBRATED_PRETEST", "CALIBRATED"}
            and self.face_threshold is not None
            and self.copy_threshold is not None
            and self.calibration_report_sha256 is not None
        )

    @classmethod
    def synthetic(cls, *, face_threshold: float, copy_threshold: float) -> DecisionPolicy:
        return cls(
            "synthetic-test-policy",
            "SYNTHETIC_TEST_ONLY",
            face_threshold,
            copy_threshold,
            None,
            None,
            200,
            8,
            8,
            "0" * 64,
            None,
            None,
            False,
            False,
            "0" * 64,
        )

    @classmethod
    def review_only(cls) -> DecisionPolicy:
        return cls(
            "missing-calibration-review-only",
            "REVIEW_ONLY",
            None,
            None,
            None,
            None,
            200,
            8,
            8,
            "0" * 64,
            None,
            None,
            False,
            False,
            "0" * 64,
        )

    def axis(
        self,
        score: float | None,
        *,
        threshold: float | None,
        review_boundary: float | None,
        assessable: bool,
        ran: bool = True,
        error: bool = False,
    ) -> AxisState:
        if error:
            return AxisState.ERROR
        if not ran:
            return AxisState.NOT_RUN
        if not assessable or score is None:
            return AxisState.UNASSESSABLE
        if threshold is None or (
            not self.calibrated and self.status != "SYNTHETIC_TEST_ONLY"
        ):
            return AxisState.BORDERLINE
        if score >= threshold:
            return AxisState.SUPPORTED
        if review_boundary is not None and score >= review_boundary:
            return AxisState.BORDERLINE
        return AxisState.NOT_SUPPORTED

    def decide(self, face: AxisState, copy: AxisState) -> str:
        if face in {AxisState.ERROR, AxisState.NOT_RUN} or copy == AxisState.ERROR:
            return "execution_error"
        if copy == AxisState.NOT_RUN:
            if face == AxisState.SUPPORTED:
                return "identity_only_supported"
            if face == AxisState.BORDERLINE:
                return "review"
            return "no_accepted_match"
        if AxisState.BORDERLINE in {face, copy}:
            return "review"
        if face == AxisState.SUPPORTED and copy == AxisState.SUPPORTED:
            return "both_supported_pending_human_confirmation"
        if face == AxisState.SUPPORTED and copy == AxisState.NOT_SUPPORTED:
            return "identity_only_supported"
        if copy == AxisState.SUPPORTED and face == AxisState.UNASSESSABLE:
            return "copy_only_identity_unknown"
        if copy == AxisState.SUPPORTED and face == AxisState.NOT_SUPPORTED:
            return "copy_face_conflict_review"
        if face == AxisState.UNASSESSABLE or copy == AxisState.UNASSESSABLE:
            return "review"
        return "no_accepted_match"


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def load_policy(
    path: Path, *, project_root: Path, require_calibrated: bool = False
) -> DecisionPolicy:
    path = path.resolve()
    if not path.is_file():
        raise BlockedState("BLOCKED_CALIBRATION_POLICY", f"policy not found: {path}")
    raw_bytes = path.read_bytes()
    try:
        value = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FaceInputError("policy must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != "faceproof.policy.v1":
        raise FaceInputError("unsupported policy schema")
    bounds = value.get("bounds") or {}
    thresholds = value.get("thresholds") or {}
    validation = value.get("quality_validation") or {}
    artifacts = value.get("artifacts") or {}
    model_lock_path = project_root / "model-lock.json"
    if not model_lock_path.is_file():
        raise BlockedState("BLOCKED_MODEL_LOCK", "model-lock.json is missing")
    actual_model_lock = hashlib.sha256(model_lock_path.read_bytes()).hexdigest()
    if artifacts.get("model_lock_sha256") != actual_model_lock:
        raise BlockedState("BLOCKED_STALE_POLICY", "policy model-lock hash is missing or stale")
    calibration_digest = artifacts.get("calibration_report_sha256")
    if calibration_digest is not None and not _valid_digest(calibration_digest):
        raise FaceInputError("policy calibration report hash is invalid")
    policy = DecisionPolicy(
        policy_id=str(value.get("policy_id") or ""),
        status=str(value.get("status") or "REVIEW_ONLY"),
        face_threshold=_optional_float(thresholds.get("face_accept")),
        copy_threshold=_optional_float(thresholds.get("copy_accept")),
        face_review_boundary=_optional_float(thresholds.get("face_review")),
        copy_review_boundary=_optional_float(thresholds.get("copy_review")),
        max_gallery_media=int(bounds.get("max_gallery_media", 0)),
        max_enrollment_references=int(bounds.get("max_enrollment_references", 0)),
        max_copy_references=int(bounds.get("max_copy_references", 0)),
        model_lock_sha256=actual_model_lock,
        calibration_report_sha256=calibration_digest,
        test_report_sha256=artifacts.get("test_report_sha256"),
        pose_policy_validated=validation.get("pose") is True,
        visibility_policy_validated=validation.get("visibility") is True,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        targets={
            str(key): float(metric_value)
            for key, metric_value in (value.get("targets") or {}).items()
            if isinstance(metric_value, (int, float))
        },
    )
    if not policy.policy_id or min(
        policy.max_gallery_media,
        policy.max_enrollment_references,
        policy.max_copy_references,
    ) <= 0:
        raise FaceInputError("policy identifiers and bounds must be positive")
    if require_calibrated and not policy.calibrated:
        raise BlockedState("BLOCKED_REAL_CALIBRATION_DATA", "policy is review-only")
    return policy


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not -1.0 <= result <= 1.0:
        raise FaceInputError("policy thresholds must be between -1 and 1")
    return result


def freeze_policy(
    calibration_report: Path,
    test_report: Path | None,
    *,
    project_root: Path,
    output: Path,
    max_gallery_media: int,
    max_enrollment_references: int,
    max_copy_references: int,
) -> dict[str, Any]:
    """Bind frozen calibration decisions and a locked-test report to model contracts."""
    calibration_bytes = calibration_report.read_bytes()
    test_bytes = test_report.read_bytes() if test_report is not None else None
    try:
        calibration = json.loads(calibration_bytes)
        test = json.loads(test_bytes) if test_bytes is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FaceInputError("calibration and test reports must be UTF-8 JSON") from exc
    if calibration.get("schema") != "faceproof.calibration-decision.v1":
        raise FaceInputError("unsupported calibration decision report")
    if calibration.get("status") != "FROZEN_BEFORE_TEST":
        raise FaceInputError("calibration decisions were not frozen before the locked test")
    if test is not None and (
        test.get("schema") != "faceproof.open-set-evaluation.v1"
        or test.get("split") != "test"
    ):
        raise FaceInputError("the supplied test artifact is not a locked-test report")
    calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
    if test is not None:
        test_policy = test.get("policy") or {}
        if (
            test_policy.get("policy_id") != calibration.get("policy_id")
            or test_policy.get("calibration_report_sha256") != calibration_sha256
        ):
            raise FaceInputError(
                "locked-test report was not produced by this frozen calibration decision"
            )
    thresholds = calibration.get("thresholds") or {}
    face_accept = _optional_float(thresholds.get("face_accept"))
    copy_accept = _optional_float(thresholds.get("copy_accept"))
    if face_accept is None or copy_accept is None:
        raise FaceInputError("calibration must freeze separate face and copy thresholds")
    model_lock = project_root / "model-lock.json"
    if not model_lock.is_file():
        raise BlockedState("BLOCKED_MODEL_LOCK", "model-lock.json is missing")
    value = {
        "schema": "faceproof.policy.v1",
        "policy_id": str(calibration.get("policy_id") or "faceproof-calibrated-v1"),
        "status": "CALIBRATED" if test_bytes is not None else "CALIBRATED_PRETEST",
        "thresholds": {
            "face_accept": face_accept,
            "face_review": _optional_float(thresholds.get("face_review")),
            "copy_accept": copy_accept,
            "copy_review": _optional_float(thresholds.get("copy_review")),
        },
        "bounds": {
            "max_gallery_media": max_gallery_media,
            "max_enrollment_references": max_enrollment_references,
            "max_copy_references": max_copy_references,
        },
        "quality_validation": calibration.get("quality_validation")
        or {"pose": False, "visibility": False},
        "targets": calibration.get("targets") or {},
        "artifacts": {
            "model_lock_sha256": hashlib.sha256(model_lock.read_bytes()).hexdigest(),
            "calibration_report_sha256": calibration_sha256,
            "test_report_sha256": (
                hashlib.sha256(test_bytes).hexdigest() if test_bytes is not None else None
            ),
        },
    }
    write_json(output, value)
    return value
