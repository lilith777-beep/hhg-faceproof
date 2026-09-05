import hashlib
import json
from pathlib import Path

import pytest

from faceproof.errors import BlockedState
from faceproof.policy import AxisState, DecisionPolicy, freeze_policy, load_policy


@pytest.mark.parametrize(
    ("face", "copy", "expected"),
    [
        (
            AxisState.SUPPORTED,
            AxisState.SUPPORTED,
            "both_supported_pending_human_confirmation",
        ),
        (AxisState.SUPPORTED, AxisState.NOT_SUPPORTED, "identity_only_supported"),
        (AxisState.UNASSESSABLE, AxisState.SUPPORTED, "copy_only_identity_unknown"),
        (AxisState.NOT_SUPPORTED, AxisState.SUPPORTED, "copy_face_conflict_review"),
        (AxisState.BORDERLINE, AxisState.SUPPORTED, "review"),
        (AxisState.NOT_SUPPORTED, AxisState.NOT_SUPPORTED, "no_accepted_match"),
        (AxisState.ERROR, AxisState.SUPPORTED, "execution_error"),
        (AxisState.SUPPORTED, AxisState.NOT_RUN, "identity_only_supported"),
    ],
)
def test_independent_axis_state_table(
    face: AxisState, copy: AxisState, expected: str
) -> None:
    policy = DecisionPolicy.synthetic(face_threshold=0.5, copy_threshold=0.7)
    assert policy.decide(face, copy) == expected


def test_uncalibrated_policy_never_turns_score_into_support() -> None:
    policy = DecisionPolicy.synthetic(face_threshold=0.5, copy_threshold=0.7)
    review_policy = DecisionPolicy(
        policy.policy_id,
        "REVIEW_ONLY",
        None,
        None,
        None,
        None,
        policy.max_gallery_media,
        policy.max_enrollment_references,
        policy.max_copy_references,
        policy.model_lock_sha256,
        None,
        None,
        False,
        False,
        policy.source_sha256,
    )
    assert (
        review_policy.axis(0.99, threshold=None, review_boundary=None, assessable=True)
        == AxisState.BORDERLINE
    )


def test_stale_model_lock_blocks_policy(tmp_path: Path) -> None:
    (tmp_path / "model-lock.json").write_text("{}", encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema": "faceproof.policy.v1",
                "policy_id": "p",
                "status": "REVIEW_ONLY",
                "bounds": {
                    "max_gallery_media": 1,
                    "max_enrollment_references": 1,
                    "max_copy_references": 1,
                },
                "thresholds": {},
                "quality_validation": {},
                "artifacts": {"model_lock_sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    assert hashlib.sha256((tmp_path / "model-lock.json").read_bytes()).hexdigest() != "0" * 64
    with pytest.raises(BlockedState, match="BLOCKED_STALE_POLICY"):
        load_policy(policy_path, project_root=tmp_path)


def test_policy_freezes_before_test_then_binds_locked_report(tmp_path: Path) -> None:
    (tmp_path / "model-lock.json").write_text('{"frozen":true}', encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "faceproof.calibration-decision.v1",
                "status": "FROZEN_BEFORE_TEST",
                "policy_id": "study-1",
                "thresholds": {"face_accept": 0.6, "copy_accept": 0.8},
            }
        ),
        encoding="utf-8",
    )
    pretest = tmp_path / "pretest.json"
    freeze_policy(
        calibration,
        None,
        project_root=tmp_path,
        output=pretest,
        max_gallery_media=200,
        max_enrollment_references=4,
        max_copy_references=4,
    )
    assert load_policy(pretest, project_root=tmp_path, require_calibrated=True).calibrated

    test_report = tmp_path / "test.json"
    test_report.write_text(
        json.dumps({"schema": "faceproof.open-set-evaluation.v1", "split": "test"}),
        encoding="utf-8",
    )
    final = tmp_path / "final.json"
    value = freeze_policy(
        calibration,
        test_report,
        project_root=tmp_path,
        output=final,
        max_gallery_media=200,
        max_enrollment_references=4,
        max_copy_references=4,
    )
    assert value["status"] == "CALIBRATED"
    assert value["artifacts"]["test_report_sha256"] == hashlib.sha256(
        test_report.read_bytes()
    ).hexdigest()
