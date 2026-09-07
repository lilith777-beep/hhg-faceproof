import json

import pytest

from faceproof.errors import BlockedState, FaceInputError
from faceproof.evaluation import (
    EVALUABLE_AXIS_STATES,
    _aggregate_preview,
    _copy_search_metrics,
    calibrate_from_evaluation,
)


def _search(index: int, *, known: bool) -> dict:
    media_id = f"m-{index}"
    return {
        "query_image_id": f"q-{index}",
        "participant_ids": [f"p-{index}"],
        "copy_reference_family_ids": [f"copy-family-{index}"],
        "known_subject": known,
        "face_search_evaluable": True,
        "copy_search_evaluable": True,
        "correct_media": [media_id] if known else [],
        "copy_correct_media": [media_id] if known else [],
        "terminal_dispositions": [
            {
                "media_id": media_id,
                "detail": {
                    "face_similarity": 0.95 if known else 0.1,
                    "copy_similarity": 0.98 if known else 0.2,
                },
            }
        ],
    }


def test_calibration_uses_search_level_confidence_targets(tmp_path) -> None:
    report = tmp_path / "calibration.json"
    report.write_text(
        json.dumps(
            {
                "schema": "faceproof.open-set-evaluation.v1",
                "split": "calibration",
                "policy": {
                    "status": "REVIEW_ONLY",
                    "calibration_report_sha256": None,
                    "test_report_sha256": None,
                },
                "searches": [
                    *[_search(index, known=True) for index in range(50)],
                    *[_search(index + 50, known=False) for index in range(400)],
                ],
            }
        ),
        encoding="utf-8",
    )
    result = calibrate_from_evaluation(
        report, tmp_path / "decision.json", policy_id="study"
    )
    assert result["status"] == "FROZEN_BEFORE_TEST"
    assert result["thresholds"]["face_accept"] > 0.1
    assert result["thresholds"]["copy_accept"] > 0.2
    assert result["selected_metrics"]["face"]["unknown_searches"] == 400
    assert result["selected_metrics"]["copy"]["confidence_unit"] == (
        "copy_reference_family_ids"
    )


def test_calibration_rejects_test_split_and_underpowered_target(tmp_path) -> None:
    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps(
            {"schema": "faceproof.open-set-evaluation.v1", "split": "test", "searches": []}
        ),
        encoding="utf-8",
    )
    with pytest.raises(FaceInputError, match="calibration split"):
        calibrate_from_evaluation(wrong, tmp_path / "x", policy_id="study")

    small = tmp_path / "small.json"
    small.write_text(
        json.dumps(
            {
                "schema": "faceproof.open-set-evaluation.v1",
                "split": "calibration",
                "policy": {
                    "status": "REVIEW_ONLY",
                    "calibration_report_sha256": None,
                    "test_report_sha256": None,
                },
                "searches": [_search(0, known=True), _search(1, known=False)],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BlockedState, match="BLOCKED_REAL_CALIBRATION_DATA"):
        calibrate_from_evaluation(small, tmp_path / "x", policy_id="study")


def test_calibration_rejects_post_calibration_policy_input(tmp_path) -> None:
    report = tmp_path / "contaminated.json"
    report.write_text(
        json.dumps(
            {
                "schema": "faceproof.open-set-evaluation.v1",
                "split": "calibration",
                "policy": {
                    "status": "CALIBRATED_PRETEST",
                    "calibration_report_sha256": "a" * 64,
                    "test_report_sha256": None,
                },
                "searches": [_search(0, known=True)],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FaceInputError, match="review-only policy"):
        calibrate_from_evaluation(report, tmp_path / "x", policy_id="study")


def _preview_row(
    participant: str,
    family: str,
    *,
    truth_count: int,
    retained: bool,
    verifier_recall: float | None,
) -> dict:
    value = {
        "identity_positive_recall": float(retained) if truth_count else None,
        "copy_positive_recall": None,
        "all_positive_recovery": float(retained) if truth_count else None,
        "exhaustive_verifier_positive_retention": verifier_recall,
        "truth_positive_count": truth_count,
        "truth_positive_retained": truth_count if retained else 0,
        "truth_any_hit": retained and bool(truth_count),
        "identity_positive_count": truth_count,
        "copy_positive_count": 0,
        "identity_all_retained": retained,
        "copy_all_retained": True,
        "copy_family_all_retained": ({family: retained} if truth_count else {}),
    }
    return {
        "participant_ids": [participant],
        "source_image_family_id": f"query-{family}",
        "preview": {str(k): dict(value) for k in (5, 10, 20, 50)},
    }


def test_preview_recall_excludes_empty_truth_and_clusters_participants() -> None:
    rows = [
        _preview_row("p1", "f1", truth_count=1, retained=True, verifier_recall=1.0),
        _preview_row("p1", "f1", truth_count=1, retained=False, verifier_recall=0.0),
        _preview_row("unknown", "none", truth_count=0, retained=False, verifier_recall=None),
    ]
    metric = _aggregate_preview(rows)["5"]
    assert metric["macro_all_positive_recall"] == 0.5
    assert metric["micro_all_positive_recall"] == 0.5
    assert metric["any_hit_query_recall"] == 0.5
    assert metric["identity_cluster_count"] == 1
    assert metric["identity_candidate_recall_interval_95_clustered"][1] < 0.8
    assert metric["exhaustive_verifier_positive_retention"] == 0.5


def test_copy_search_metrics_do_not_turn_failures_into_true_negatives() -> None:
    rows = [
        {
            "copy_correct_media": [],
            "copy_search_evaluable": False,
            "copy_correct_return": False,
            "accepted_copy_media": [],
            "copy_reference_family_ids": ["failed-family"],
        },
        {
            "copy_correct_media": [],
            "copy_search_evaluable": True,
            "copy_correct_return": False,
            "accepted_copy_media": [],
            "copy_reference_family_ids": ["measured-family"],
        },
    ]
    metric = _copy_search_metrics(rows)
    assert metric["unknown_searches"] == 1
    assert metric["unevaluable_searches"] == 1
    assert metric["false_alarm_rate"] == 0.0


def test_unassessable_and_execution_states_are_not_evaluable() -> None:
    assert {"SUPPORTED", "NOT_SUPPORTED", "BORDERLINE"} == EVALUABLE_AXIS_STATES
    assert "UNASSESSABLE" not in EVALUABLE_AXIS_STATES
    assert "NOT_RUN" not in EVALUABLE_AXIS_STATES
    assert "ERROR" not in EVALUABLE_AXIS_STATES
