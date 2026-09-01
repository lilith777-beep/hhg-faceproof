from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "faceproof.evidence.v1"
ANCHOR_SCHEMA = "faceproof.ethereum-anchor.v1"
ANCHOR_MAGIC = b"FACEPROOF\x01"


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
    return sha256_bytes(canonical_json_bytes(value))


def anchor_payload(digest_hex: str) -> bytes:
    raw = bytes.fromhex(digest_hex)
    if len(raw) != 32:
        raise ValueError("evidence digest must be exactly 32 bytes")
    return ANCHOR_MAGIC + raw


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
