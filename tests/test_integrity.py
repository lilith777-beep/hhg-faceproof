from faceproof.integrity import ANCHOR_MAGIC, anchor_payload, canonical_json_bytes, evidence_digest


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


def test_anchor_payload_is_versioned_and_fixed_length() -> None:
    digest = "ab" * 32
    payload = anchor_payload(digest)
    assert payload.startswith(ANCHOR_MAGIC)
    assert payload[len(ANCHOR_MAGIC) :] == bytes.fromhex(digest)
