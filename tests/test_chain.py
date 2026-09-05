from __future__ import annotations

from pathlib import Path

import pytest

from faceproof.chain import REGISTRY_CODE_SHA256, REGISTRY_RUNTIME, EthereumAnchor
from faceproof.config import Settings
from faceproof.errors import ChainError, VerificationError
from faceproof.integrity import (
    ANCHOR_SCHEMA,
    SCHEMA,
    anchor_payload,
    evidence_file_digest,
    write_json,
)


class HexValue(bytes):
    def hex(self) -> str:
        return "0x" + super().hex()


class FakeEth:
    chain_id = 11_155_111
    block_number = 101

    def __init__(self, tx: dict, receipt: dict) -> None:
        self.tx = tx
        self.receipt = receipt

    def get_transaction(self, tx_hash: str) -> dict:
        return self.tx

    def get_transaction_receipt(self, tx_hash: str) -> dict:
        return self.receipt

    def get_block(self, block_number: int) -> dict:
        return {"hash": self.receipt["blockHash"]}

    @staticmethod
    def get_code(address: str) -> bytes:
        return REGISTRY_RUNTIME

    @staticmethod
    def get_storage_at(address: str, slot: int) -> bytes:
        return (1).to_bytes(32, "big")


class FakeWeb3:
    def __init__(self, tx: dict, receipt: dict) -> None:
        self.eth = FakeEth(tx, receipt)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        model_dir=tmp_path / "models",
        artifact_dir=tmp_path / "artifacts",
        mastodon_instance="https://social.example",
        mastodon_tag=None,
        mastodon_max_pages=1,
        mastodon_page_size=10,
        rpc_url="https://rpc.example",
        private_key=None,
        signer_mode="unlocked",
        expected_signer=None,
        expected_chain_id=11_155_111,
        explorer_tx_url="https://example/tx/{tx_hash}",
        detection_threshold=0.8,
        face_threshold=0.5,
        face_threshold_source="test",
        prefilter_face_threshold=0.3,
        ambiguity_margin=0.05,
        phash_max_distance=12,
        akaze_min_inliers=10,
        akaze_min_inlier_ratio=0.25,
        max_candidates=10,
        max_finalists=5,
    )


def _make_anchor(tmp_path: Path) -> tuple[EthereumAnchor, Path, Path]:
    evidence = {
        "schema": SCHEMA,
        "status": "FINAL",
        "match": {"page_url": "https://x.com/a/status/1"},
    }
    evidence_path = tmp_path / "evidence.json"
    write_json(evidence_path, evidence)
    digest = evidence_file_digest(evidence_path)
    tx_hash = HexValue(bytes.fromhex("11" * 32))
    block_hash = HexValue(bytes.fromhex("22" * 32))
    sender = "0x0000000000000000000000000000000000000001"
    tx = {
        "input": anchor_payload(digest),
        "hash": tx_hash,
        "from": sender,
        "to": sender,
    }
    receipt = {
        "status": 1,
        "blockNumber": 100,
        "blockHash": block_hash,
        "logs": [{"address": sender, "topics": [bytes.fromhex(digest)]}],
    }
    anchor = EthereumAnchor.__new__(EthereumAnchor)
    anchor.settings = _settings(tmp_path)
    anchor.w3 = FakeWeb3(tx, receipt)

    receipt_path = tmp_path / "anchor.json"
    write_json(
        receipt_path,
        {
            "schema": ANCHOR_SCHEMA,
            "evidence_sha256": digest,
            "payload_hex": "0x" + anchor_payload(digest).hex(),
            "transaction_hash": tx_hash.hex(),
            "chain_id": 11_155_111,
            "block_number": 100,
            "block_hash": block_hash.hex(),
            "sender": sender,
            "recipient": sender,
            "contract_code_sha256": REGISTRY_CODE_SHA256,
            "commitment_log_topic": "0x" + digest,
            "explorer_url": None,
        },
    )
    return anchor, evidence_path, receipt_path


def test_verification_reads_transaction_and_passes_all_checks(tmp_path: Path) -> None:
    anchor, evidence_path, receipt_path = _make_anchor(tmp_path)
    report = anchor.verify(evidence_path, receipt_path)
    assert report.verified is True
    assert all(report.checks.values())
    assert report.confirmations == 2


def test_tampered_evidence_fails_on_chain_verification(tmp_path: Path) -> None:
    anchor, evidence_path, receipt_path = _make_anchor(tmp_path)
    write_json(
        evidence_path,
        {"schema": SCHEMA, "status": "FINAL", "match": {"page_url": "https://x.com/a/status/2"}},
    )
    with pytest.raises(VerificationError, match=r"digest_recomputed|transaction_payload"):
        anchor.verify(evidence_path, receipt_path)


def test_chain_writes_reject_mainnet_even_with_explicit_authorization(tmp_path: Path) -> None:
    anchor, _, _ = _make_anchor(tmp_path)
    anchor.w3.eth.chain_id = 1
    anchor.rpc_url = "https://mainnet.example"
    with pytest.raises(ChainError, match="mainnet"):
        anchor._assert_write_authorized(allow_public_testnet=True)


def test_remote_rpc_write_requires_explicit_public_testnet_authorization(
    tmp_path: Path,
) -> None:
    anchor, _, _ = _make_anchor(tmp_path)
    anchor.rpc_url = "https://sepolia.example"
    with pytest.raises(ChainError, match="explicit --allow-public-testnet"):
        anchor._assert_write_authorized()
    anchor._assert_write_authorized(allow_public_testnet=True)
