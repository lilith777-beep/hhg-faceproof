import json

import pytest

from faceproof.integrity import (
    SCHEMA,
    anchor_payload,
    canonical_json_bytes,
    evidence_digest,
    evidence_file_digest,
    finalize_evidence,
    write_json,
)


def test_canonical_json_is_order_independent_and_utf8() -> None:
    left = {"z": "गोवा", "a": [1, True]}
    right = {"a": [1, True], "z": "गोवा"}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert b"\\u" not in canonical_json_bytes(left)
    assert evidence_digest(left) == evidence_digest(right)


def test_tamper_changes_digest() -> None:
    original = {"match": {"page_url": "https://x.com/demo/status/1"}}
    tampered = {"match": {"page_url": "https://x.com/demo/status/2"}}
    assert evidence_digest(original) != evidence_digest(tampered)


def test_anchor_payload_is_fixed_length_domain_separated_digest() -> None:
    digest = "ab" * 32
    payload = anchor_payload(digest)
    assert payload == bytes.fromhex(digest)


def test_finalization_records_review_nonce_and_hashes_exact_bytes(tmp_path) -> None:
    draft = tmp_path / "draft.json"
    final = tmp_path / "final.json"
    write_json(
        draft,
        {
            "schema": SCHEMA,
            "status": "DRAFT_REVIEW_REQUIRED",
            "score": 0.5,
            "claims": {"reviewable_claims": ["copy supported; identity unknown"]},
        },
    )
    value = finalize_evidence(
        draft,
        final,
        reviewer="reviewer-1",
        confirmed_claims=["copy supported; identity unknown"],
    )
    assert value["status"] == "FINAL"
    assert len(value["nonce_hex"]) == 64
    assert final.read_bytes() == canonical_json_bytes(value)
    digest = evidence_file_digest(final)
    final.write_bytes(final.read_bytes().replace(b"0.5", b"0.6"))
    assert evidence_file_digest(final) != digest


def test_finalization_rejects_missing_review_and_nan(tmp_path) -> None:
    draft = tmp_path / "draft.json"
    write_json(
        draft,
        {
            "schema": SCHEMA,
            "status": "DRAFT_REVIEW_REQUIRED",
            "claims": {"reviewable_claims": ["x"]},
        },
    )
    with pytest.raises(ValueError, match="reviewer"):
        finalize_evidence(draft, tmp_path / "x", reviewer="", confirmed_claims=["x"])
    draft.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "DRAFT_REVIEW_REQUIRED",
                "claims": {"reviewable_claims": ["x"]},
                "x": float("nan"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        finalize_evidence(draft, tmp_path / "x", reviewer="r", confirmed_claims=["x"])
