from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BlockedState, FaceInputError
from .models import SearchHit

SPLITS = {"development", "calibration", "test"}
ROLES = {"enrollment_face", "candidate_face", "copy_reference", "candidate_copy"}
BIOMETRIC_ROLES = {"enrollment_face", "candidate_face"}


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    image_id: str
    path: Path | None
    media_id: str | None
    canonical_uri: str | None
    participant_ids: tuple[str, ...]
    consent_ref: str | None
    biometric_consent: bool
    roles: frozenset[str]
    capture_session_id: str
    source_image_family_id: str
    split: str
    sha256: str
    face_annotations: tuple[dict[str, Any], ...]
    copy_parent_image_id: str | None
    labels: dict[str, Any]

    @property
    def face_eligible(self) -> bool:
        return bool(self.roles & BIOMETRIC_ROLES) and self.biometric_consent and bool(
            self.consent_ref
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    path: Path
    records: tuple[ManifestRecord, ...]
    sha256: str

    def record_for_path(self, path: Path, role: str) -> ManifestRecord:
        resolved = path.resolve()
        matches = [
            record
            for record in self.records
            if record.path == resolved and role in record.roles
        ]
        if not matches:
            raise BlockedState(
                "BLOCKED_PERMISSION_MANIFEST",
                f"{resolved.name} is not declared with role {role}",
            )
        record = matches[0]
        if role in BIOMETRIC_ROLES and not record.face_eligible:
            raise BlockedState(
                "BLOCKED_PERMISSION_MANIFEST",
                f"{record.image_id} lacks documented biometric consent",
            )
        return record

    def face_record_for_hit(self, hit: SearchHit) -> ManifestRecord | None:
        matches = [
            record
            for record in self.records
            if "candidate_face" in record.roles
            and (
                ((record.media_id or record.image_id) == hit.media_id)
                or (record.canonical_uri and record.canonical_uri == hit.canonical_uri)
            )
        ]
        return matches[0] if matches and matches[0].face_eligible else None


def _required_text(raw: dict[str, Any], name: str, line: int) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise FaceInputError(f"manifest line {line}: {name} must be a non-empty string")
    return value.strip()


def load_manifest(path: Path, *, verify_files: bool = True) -> DatasetManifest:
    path = path.resolve()
    if not path.is_file():
        raise BlockedState("BLOCKED_PERMISSION_MANIFEST", f"manifest not found: {path}")
    raw_bytes = path.read_bytes()
    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(raw_bytes.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FaceInputError(f"manifest line {line_number}: invalid JSON") from exc
        if not isinstance(raw, dict):
            raise FaceInputError(f"manifest line {line_number}: record must be an object")
        image_id = _required_text(raw, "image_id", line_number)
        if image_id in seen_ids:
            raise FaceInputError(f"manifest line {line_number}: duplicate image_id {image_id}")
        seen_ids.add(image_id)
        roles_value = raw.get("roles")
        if not isinstance(roles_value, list) or not roles_value:
            raise FaceInputError(f"manifest line {line_number}: roles must be a non-empty list")
        roles = frozenset(str(value) for value in roles_value)
        if not roles <= ROLES:
            raise FaceInputError(f"manifest line {line_number}: unsupported role")
        split = _required_text(raw, "split", line_number)
        if split not in SPLITS:
            raise FaceInputError(f"manifest line {line_number}: invalid split {split}")
        relative = raw.get("path")
        local_path = (path.parent / relative).resolve() if isinstance(relative, str) else None
        digest = _required_text(raw, "sha256", line_number).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise FaceInputError(f"manifest line {line_number}: invalid sha256")
        if verify_files and local_path is not None:
            if not local_path.is_file():
                raise BlockedState("BLOCKED_REAL_CALIBRATION_DATA", f"missing {relative}")
            actual = hashlib.sha256(local_path.read_bytes()).hexdigest()
            if actual != digest:
                raise FaceInputError(f"manifest line {line_number}: file hash mismatch")
        participants = raw.get("participant_ids", [])
        if not isinstance(participants, list) or any(
            not isinstance(item, str) or not item for item in participants
        ):
            raise FaceInputError(f"manifest line {line_number}: invalid participant_ids")
        annotations = raw.get("face_annotations", [])
        if not isinstance(annotations, list) or any(
            not isinstance(item, dict) for item in annotations
        ):
            raise FaceInputError(f"manifest line {line_number}: invalid face_annotations")
        biometric_consent = raw.get("biometric_consent") is True
        consent_ref = raw.get("consent_ref") if isinstance(raw.get("consent_ref"), str) else None
        if roles & BIOMETRIC_ROLES:
            if not biometric_consent or not consent_ref:
                raise BlockedState(
                    "BLOCKED_PERMISSION_MANIFEST",
                    f"{image_id} has a biometric role without documented consent",
                )
            if not participants:
                raise FaceInputError(
                    f"manifest line {line_number}: biometric records need participant_ids"
                )
            for annotation in annotations:
                if not annotation.get("participant_id") or not annotation.get("consent_ref"):
                    raise BlockedState(
                        "BLOCKED_PERMISSION_MANIFEST",
                        f"{image_id} face annotations need participant_id and consent_ref",
                    )
        labels = raw.get("labels", {})
        if not isinstance(labels, dict):
            raise FaceInputError(f"manifest line {line_number}: labels must be an object")
        records.append(
            ManifestRecord(
                image_id=image_id,
                path=local_path,
                media_id=str(raw["media_id"]) if raw.get("media_id") is not None else None,
                canonical_uri=(
                    str(raw["canonical_uri"])
                    if raw.get("canonical_uri") is not None
                    else None
                ),
                participant_ids=tuple(participants),
                consent_ref=consent_ref,
                biometric_consent=biometric_consent,
                roles=roles,
                capture_session_id=_required_text(raw, "capture_session_id", line_number),
                source_image_family_id=_required_text(
                    raw, "source_image_family_id", line_number
                ),
                split=split,
                sha256=digest,
                face_annotations=tuple(annotations),
                copy_parent_image_id=(
                    str(raw["copy_parent_image_id"])
                    if raw.get("copy_parent_image_id") is not None
                    else None
                ),
                labels=labels,
            )
        )
    if not records:
        raise FaceInputError("manifest contains no records")
    _validate_no_leakage(records)
    return DatasetManifest(path, tuple(records), hashlib.sha256(raw_bytes).hexdigest())


def _validate_no_leakage(records: list[ManifestRecord]) -> None:
    participant_splits: dict[str, set[str]] = {}
    family_splits: dict[str, set[str]] = {}
    hash_splits: dict[str, set[str]] = {}
    by_id = {record.image_id: record for record in records}
    for record in records:
        for participant in record.participant_ids:
            participant_splits.setdefault(participant, set()).add(record.split)
        family_splits.setdefault(record.source_image_family_id, set()).add(record.split)
        hash_splits.setdefault(record.sha256, set()).add(record.split)
        if record.copy_parent_image_id:
            parent = by_id.get(record.copy_parent_image_id)
            if parent is None:
                raise FaceInputError(
                    f"copy parent {record.copy_parent_image_id} for {record.image_id} is missing"
                )
            if parent.split != record.split:
                raise FaceInputError(
                    f"copy lineage crosses splits: {parent.image_id} -> {record.image_id}"
                )
    leaked_participants = sorted(
        key for key, splits in participant_splits.items() if len(splits) > 1
    )
    leaked_families = sorted(key for key, splits in family_splits.items() if len(splits) > 1)
    leaked_hashes = sorted(key for key, splits in hash_splits.items() if len(splits) > 1)
    if leaked_participants:
        raise FaceInputError(f"identity leakage across splits: {leaked_participants}")
    if leaked_families:
        raise FaceInputError(f"source-family leakage across splits: {leaked_families}")
    if leaked_hashes:
        raise FaceInputError(f"exact-image leakage across splits: {leaked_hashes}")
