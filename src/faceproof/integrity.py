from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "faceproof.evidence.v2"
ANCHOR_SCHEMA = "faceproof.ethereum-anchor.v2"
EVIDENCE_MAGIC = b"FACEPROOF-EVIDENCE\x02\x00"
ANCHOR_MAGIC = b"FACEPROOF-ANCHOR\x02\x00"


def canonical_json_bytes(value: Any) -> bytes:
    """Deterministic UTF-8 JSON used as the evidence fingerprint input."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evidence_digest(value: dict[str, Any]) -> str:
    return sha256_bytes(EVIDENCE_MAGIC + canonical_json_bytes(value))


def evidence_file_digest(path: Path) -> str:
    """Hash the exact received evidence bytes with the versioned evidence domain."""
    return sha256_bytes(EVIDENCE_MAGIC + path.read_bytes())


def finalize_evidence(
    draft_path: Path,
    output_path: Path,
    *,
    reviewer: str,
    confirmed_claims: list[str],
    notes: str = "",
) -> dict[str, Any]:
    """Finalize a reviewed private manifest exactly once, before anchoring."""
    if not reviewer.strip():
        raise ValueError("reviewer must be recorded")
    value = read_json(draft_path)
    if value.get("schema") != SCHEMA or value.get("status") != "DRAFT_REVIEW_REQUIRED":
        raise ValueError("only a FaceProof v2 draft can be finalized")
    if not confirmed_claims:
        raise ValueError("at least one explicit reviewed claim is required")
    allowed = set((value.get("claims") or {}).get("reviewable_claims") or [])
    if not set(confirmed_claims) <= allowed:
        raise ValueError("confirmed claims must be exact reviewable claims from the draft")
    final = dict(value)
    final["status"] = "FINAL"
    final["nonce_hex"] = secrets.token_hex(32)
    final["human_review"] = {
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "confirmed_claims": list(confirmed_claims),
        "notes": notes,
    }
    payload = canonical_json_bytes(final)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return final


def anchor_payload(digest_hex: str) -> bytes:
    raw = bytes.fromhex(digest_hex)
    if len(raw) != 32:
        raise ValueError("evidence digest must be exactly 32 bytes")
    return raw


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
