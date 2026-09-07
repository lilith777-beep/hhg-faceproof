import hashlib
import json
from pathlib import Path

import pytest

from faceproof.dataset import load_manifest
from faceproof.errors import BlockedState, FaceInputError
from faceproof.models import SearchHit


def _record(
    image_id: str,
    digest: str,
    *,
    participant: str = "p1",
    split: str = "development",
    family: str | None = None,
    role: str = "candidate_face",
) -> dict:
    return {
        "image_id": image_id,
        "media_id": image_id,
        "canonical_uri": f"https://social.example/status/{image_id}",
        "participant_ids": [participant],
        "consent_ref": f"consent://{participant}",
        "biometric_consent": True,
        "roles": [role],
        "capture_session_id": f"session-{image_id}",
        "source_image_family_id": family or f"family-{image_id}",
        "split": split,
        "sha256": digest,
        "face_annotations": [
            {"face_index": 0, "participant_id": participant, "consent_ref": f"consent://{participant}"}
        ],
        "labels": {},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_manifest_requires_candidate_face_consent(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    row = _record("m1", "a" * 64)
    row["biometric_consent"] = False
    _write(path, [row])
    with pytest.raises(BlockedState, match="BLOCKED_PERMISSION_MANIFEST"):
        load_manifest(path, verify_files=False)


def test_manifest_rejects_identity_and_family_leakage(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    rows = [
        _record("dev", "a" * 64, split="development", family="shared"),
        _record("test", "b" * 64, split="test", family="shared"),
    ]
    _write(path, rows)
    with pytest.raises(FaceInputError, match=r"identity leakage|source-family leakage"):
        load_manifest(path, verify_files=False)


def test_manifest_rejects_exact_hash_leakage(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    rows = [
        _record("dev", "a" * 64, participant="p1", split="development"),
        _record("test", "a" * 64, participant="p2", split="test"),
    ]
    _write(path, rows)
    with pytest.raises(FaceInputError, match="exact-image leakage"):
        load_manifest(path, verify_files=False)


def test_manifest_rejects_ambiguous_media_and_annotation_identity(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    first = _record("one", "a" * 64, participant="p1")
    second = _record("two", "b" * 64, participant="p2")
    second["media_id"] = first["media_id"]
    _write(path, [first, second])
    with pytest.raises(FaceInputError, match="candidate media_id"):
        load_manifest(path, verify_files=False)

    second["media_id"] = "two"
    second["face_annotations"][0]["participant_id"] = "undeclared"
    _write(path, [first, second])
    with pytest.raises(FaceInputError, match="annotation participant"):
        load_manifest(path, verify_files=False)


def test_manifest_copy_lineage_must_preserve_roles_and_family(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    parent = _record("source", "a" * 64, role="copy_reference", family="original")
    child = _record("copy", "b" * 64, role="candidate_copy", family="different")
    child["copy_parent_image_id"] = "source"
    _write(path, [parent, child])
    with pytest.raises(FaceInputError, match="copy lineage changes source family"):
        load_manifest(path, verify_files=False)


def test_manifest_verifies_files_and_authorizes_matching_media(tmp_path: Path) -> None:
    image = tmp_path / "candidate.jpg"
    image.write_bytes(b"candidate")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    row = _record("opaque-media-id", digest)
    row["path"] = image.name
    path = tmp_path / "dataset.jsonl"
    _write(path, [row])

    manifest = load_manifest(path)
    hit = SearchHit(
        "https://social.example/post/1",
        row["canonical_uri"],
        "post",
        "author",
        "authorized-account",
        "2026-09-05T00:00:00Z",
        "",
        "opaque-media-id",
        "https://social.example/media.jpg",
        "https://social.example/preview.jpg",
        100,
        100,
    )
    assert manifest.face_record_for_hit(hit).image_id == "opaque-media-id"
