from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .errors import ChainError, VerificationError
from .integrity import (
    ANCHOR_SCHEMA,
    anchor_payload,
    evidence_digest,
    read_json,
    write_json,
)
from .models import AnchorReceipt, VerificationReport


def _hex(value: Any) -> str:
    if hasattr(value, "hex"):
        result = value.hex()
        return result if result.startswith("0x") else f"0x{result}"
    text = str(value)
    return text if text.startswith("0x") else f"0x{text}"


def _input_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        text = value[2:] if value.startswith("0x") else value
        return bytes.fromhex(text)
    return bytes(value)


class EthereumAnchor:
    """Anchors a digest directly in an Ethereum transaction's immutable input field."""

    def __init__(self, settings: Settings, *, rpc_url: str | None = None) -> None:
        try:
            from web3 import HTTPProvider, Web3
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise ChainError("web3.py is not installed") from exc

        url = rpc_url or settings.rpc_url
        if not url:
            raise ChainError("FACEPROOF_RPC_URL is required")
        self.w3 = Web3(HTTPProvider(url, request_kwargs={"timeout": 20}))
        self.settings = settings
        if not self.w3.is_connected():
            raise ChainError("could not connect to the configured Ethereum RPC endpoint")

    @property
    def chain_id(self) -> int:
        return int(self.w3.eth.chain_id)

    def _assert_chain(self) -> None:
        if self.chain_id != self.settings.expected_chain_id:
            raise ChainError(
                f"RPC chain ID is {self.chain_id}, expected {self.settings.expected_chain_id}; "
                "refusing to anchor on the wrong network"
            )

    def anchor(self, evidence_path: Path, output_path: Path) -> AnchorReceipt:
        self._assert_chain()
        evidence = read_json(evidence_path)
        digest = evidence_digest(evidence)
        payload = anchor_payload(digest)
        private_key = self.settings.private_key
        if self.settings.signer_mode == "private-key":
            if not private_key:
                raise ChainError("FACEPROOF_PRIVATE_KEY is required for private-key signing")
            account = self.w3.eth.account.from_key(private_key)
            sender = account.address
        elif self.settings.signer_mode == "unlocked":
            accounts = list(self.w3.eth.accounts)
            if not accounts:
                raise ChainError("the local RPC exposes no unlocked development account")
            sender = self.settings.expected_signer or accounts[0]
            if sender.lower() not in {str(item).lower() for item in accounts}:
                raise ChainError("FACEPROOF_EXPECTED_SIGNER is not unlocked by this RPC")
        else:
            raise ChainError("FACEPROOF_SIGNER_MODE must be 'unlocked' or 'private-key'")

        nonce = self.w3.eth.get_transaction_count(sender, "pending")

        transaction: dict[str, Any] = {
            "chainId": self.chain_id,
            "nonce": nonce,
            "from": sender,
            "to": sender,
            "value": 0,
            "data": payload,
        }
        latest_block = self.w3.eth.get_block("latest")
        base_fee = int(latest_block.get("baseFeePerGas", 0) or 0)
        if base_fee:
            try:
                priority_fee = int(self.w3.eth.max_priority_fee)
            except Exception:
                priority_fee = self.w3.to_wei(1, "gwei")
            transaction["maxPriorityFeePerGas"] = priority_fee
            transaction["maxFeePerGas"] = base_fee * 2 + priority_fee
        else:
            transaction["gasPrice"] = int(self.w3.eth.gas_price)

        estimated_gas = int(self.w3.eth.estimate_gas(transaction))
        transaction["gas"] = max(21_000, int(estimated_gas * 1.15))
        if self.settings.signer_mode == "private-key":
            signed = self.w3.eth.account.sign_transaction(transaction, private_key)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        else:
            tx_hash = self.w3.eth.send_transaction(transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180, poll_latency=2)
        if int(receipt["status"]) != 1:
            raise ChainError(f"anchor transaction reverted: {_hex(tx_hash)}")

        tx_hash_hex = _hex(tx_hash)
        explorer_url = (
            self.settings.explorer_tx_url.format(tx_hash=tx_hash_hex)
            if self.settings.explorer_tx_url
            else None
        )
        anchor = AnchorReceipt(
            schema=ANCHOR_SCHEMA,
            evidence_sha256=digest,
            payload_hex="0x" + payload.hex(),
            transaction_hash=tx_hash_hex,
            chain_id=self.chain_id,
            block_number=int(receipt["blockNumber"]),
            block_hash=_hex(receipt["blockHash"]),
            sender=sender,
            recipient=sender,
            explorer_url=explorer_url,
        )
        write_json(output_path, anchor.to_dict())
        return anchor

    def verify(self, evidence_path: Path, anchor_path: Path) -> VerificationReport:
        self._assert_chain()
        evidence = read_json(evidence_path)
        stored_anchor = read_json(anchor_path)
        digest = evidence_digest(evidence)
        tx_hash = str(stored_anchor.get("transaction_hash", ""))
        if not tx_hash:
            raise VerificationError("anchor receipt has no transaction hash")

        try:
            transaction = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        except Exception as exc:
            raise VerificationError(f"could not retrieve anchor transaction {tx_hash}") from exc

        chain_payload = _input_bytes(transaction["input"])
        expected_payload = anchor_payload(digest)
        current_block = int(self.w3.eth.block_number)
        block_number = int(receipt["blockNumber"])
        canonical_block = self.w3.eth.get_block(block_number)
        confirmations = max(0, current_block - block_number + 1)

        checks = {
            "evidence_schema": evidence.get("schema") == "faceproof.evidence.v1",
            "anchor_schema": stored_anchor.get("schema") == ANCHOR_SCHEMA,
            "digest_recomputed": stored_anchor.get("evidence_sha256") == digest,
            "receipt_payload": stored_anchor.get("payload_hex") == "0x" + expected_payload.hex(),
            "transaction_payload": chain_payload == expected_payload,
            "transaction_hash": _hex(transaction["hash"]).lower() == tx_hash.lower(),
            "chain_id": int(stored_anchor.get("chain_id", -1)) == self.chain_id,
            "receipt_success": int(receipt["status"]) == 1,
            "block_number": int(stored_anchor.get("block_number", -1)) == block_number,
            "block_hash": _hex(receipt["blockHash"]).lower()
            == str(stored_anchor.get("block_hash", "")).lower(),
            "canonical_block": _hex(canonical_block["hash"]).lower()
            == _hex(receipt["blockHash"]).lower(),
            "sender": str(transaction["from"]).lower()
            == str(stored_anchor.get("sender", "")).lower(),
            "recipient": str(transaction["to"]).lower()
            == str(stored_anchor.get("recipient", "")).lower(),
            "expected_signer": not self.settings.expected_signer
            or str(transaction["from"]).lower() == self.settings.expected_signer.lower(),
            "confirmations": confirmations >= self.settings.min_confirmations,
        }
        report = VerificationReport(
            verified=all(checks.values()),
            evidence_sha256=digest,
            transaction_hash=tx_hash,
            chain_id=self.chain_id,
            block_number=block_number,
            confirmations=confirmations,
            checks=checks,
        )
        if not report.verified:
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise VerificationError(f"on-chain verification failed: {failed}")
        return report
