import json

import pytest

from faceproof.errors import BlockedState, FaceInputError
from faceproof.evaluation import calibrate_from_evaluation


def _search(index: int, *, known: bool) -> dict:
    media_id = f"m-{index}"
    return {
        "known_subject": known,
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
                "searches": [_search(0, known=True), _search(1, known=False)],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BlockedState, match="BLOCKED_REAL_CALIBRATION_DATA"):
        calibrate_from_evaluation(small, tmp_path / "x", policy_id="study")
