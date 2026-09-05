from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import Settings
from .errors import ChainError, VerificationError
from .integrity import (
    ANCHOR_SCHEMA,
    SCHEMA,
    anchor_payload,
    evidence_file_digest,
    read_json,
    write_json,
)
from .models import AnchorReceipt, VerificationReport

REGISTRY_RUNTIME = bytes.fromhex(
    "3660201060165760016000355560003560006000a1005b60006000fd"
)
REGISTRY_INIT = bytes.fromhex("601c600c600039601c6000f3") + REGISTRY_RUNTIME
REGISTRY_CODE_SHA256 = hashlib.sha256(REGISTRY_RUNTIME).hexdigest()
DEPLOYMENT_SCHEMA = "faceproof.digest-registry-deployment.v1"


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
    """Stores evidence digests in a minimal reusable on-chain registry."""

    def __init__(self, settings: Settings, *, rpc_url: str | None = None) -> None:
        try:
            from web3 import HTTPProvider, Web3
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise ChainError("web3.py is not installed") from exc

        url = rpc_url or settings.rpc_url
        if not url:
            raise ChainError("FACEPROOF_RPC_URL is required")
        self.rpc_url = url
        self.w3 = Web3(HTTPProvider(url, request_kwargs={"timeout": 20}))
        self.settings = settings
        if not self.w3.is_connected():
            raise ChainError("could not connect to the configured Ethereum RPC endpoint")

    @property
    def chain_id(self) -> int:
        return int(self.w3.eth.chain_id)

    def _assert_chain(self) -> None:
        if self.chain_id == 1:
            raise ChainError("Ethereum mainnet transactions are prohibited")
        if self.chain_id != self.settings.expected_chain_id:
            raise ChainError(
                f"RPC chain ID is {self.chain_id}, expected {self.settings.expected_chain_id}; "
                "refusing to anchor on the wrong network"
            )

    def _assert_write_authorized(self, *, allow_public_testnet: bool = False) -> None:
        self._assert_chain()
        host = (urlsplit(self.rpc_url).hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1"} and not allow_public_testnet:
            raise ChainError(
                "public-testnet writes need explicit --allow-public-testnet authorization"
            )

    @property
    def deployment_path(self) -> Path:
        return self.settings.artifact_dir / "chain" / "deployment.json"

    def deploy_registry(self, *, allow_public_testnet: bool = False) -> str:
        """Explicitly deploy the pinned minimal digest registry and persist metadata."""
        self._assert_write_authorized(allow_public_testnet=allow_public_testnet)
        sender, private_key = self._signer()
        transaction = self._base_transaction(sender, data=REGISTRY_INIT)
        tx_hash = self._send(transaction, private_key)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180, poll_latency=2)
        if int(receipt["status"]) != 1 or not receipt.get("contractAddress"):
            raise ChainError("digest registry deployment failed")
        address = str(receipt["contractAddress"])
        code = bytes(self.w3.eth.get_code(address))
        if hashlib.sha256(code).hexdigest() != REGISTRY_CODE_SHA256:
            raise ChainError("deployed digest registry bytecode does not match the pinned code")
        write_json(
            self.deployment_path,
            {
                "schema": DEPLOYMENT_SCHEMA,
                "chain_id": self.chain_id,
                "contract_address": address,
                "runtime_code_sha256": REGISTRY_CODE_SHA256,
                "deployment_transaction_hash": _hex(tx_hash),
                "deployment_block_number": int(receipt["blockNumber"]),
            },
        )
        return address

    def _registry(self, *, allow_public_testnet: bool = False) -> str:
        if not self.deployment_path.is_file():
            return self.deploy_registry(allow_public_testnet=allow_public_testnet)
        value = read_json(self.deployment_path)
        if value.get("schema") != DEPLOYMENT_SCHEMA:
            raise ChainError("unsupported digest registry deployment metadata")
        if int(value.get("chain_id", -1)) != self.chain_id:
            raise ChainError("digest registry deployment is for the wrong network")
        address = str(value.get("contract_address") or "")
        code = bytes(self.w3.eth.get_code(address))
        if not code:
            raise ChainError("digest registry contract is missing on this chain state")
        if hashlib.sha256(code).hexdigest() != REGISTRY_CODE_SHA256:
            raise ChainError("digest registry contract bytecode does not match")
        return address

    def _signer(self) -> tuple[str, str | None]:
        private_key = self.settings.private_key
        if self.settings.signer_mode == "private-key":
            if not private_key:
                raise ChainError("FACEPROOF_PRIVATE_KEY is required for private-key signing")
            return self.w3.eth.account.from_key(private_key).address, private_key
        if self.settings.signer_mode == "unlocked":
            accounts = list(self.w3.eth.accounts)
            if not accounts:
                raise ChainError("the local RPC exposes no unlocked development account")
            sender = self.settings.expected_signer or accounts[0]
            if sender.lower() not in {str(item).lower() for item in accounts}:
                raise ChainError("FACEPROOF_EXPECTED_SIGNER is not unlocked by this RPC")
            return sender, None
        raise ChainError("FACEPROOF_SIGNER_MODE must be 'unlocked' or 'private-key'")

    def _base_transaction(
        self, sender: str, *, data: bytes, to: str | None = None
    ) -> dict[str, Any]:
        transaction: dict[str, Any] = {
            "chainId": self.chain_id,
            "nonce": self.w3.eth.get_transaction_count(sender, "pending"),
            "from": sender,
            "value": 0,
            "data": data,
        }
        if to is not None:
            transaction["to"] = to
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
        return transaction

    def _send(self, transaction: dict[str, Any], private_key: str | None) -> Any:
        if private_key:
            signed = self.w3.eth.account.sign_transaction(transaction, private_key)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            return self.w3.eth.send_raw_transaction(raw)
        return self.w3.eth.send_transaction(transaction)

    def anchor(
        self,
        evidence_path: Path,
        output_path: Path,
        *,
        allow_public_testnet: bool = False,
    ) -> AnchorReceipt:
        self._assert_write_authorized(allow_public_testnet=allow_public_testnet)
        evidence = read_json(evidence_path)
        if evidence.get("schema") != SCHEMA or evidence.get("status") != "FINAL":
            raise ChainError("only finalized reviewed FaceProof v2 evidence may be anchored")
        digest = evidence_file_digest(evidence_path)
        payload = anchor_payload(digest)
        registry = self._registry(allow_public_testnet=allow_public_testnet)
        sender, private_key = self._signer()
        transaction = self._base_transaction(sender, data=payload, to=registry)
        tx_hash = self._send(transaction, private_key)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180, poll_latency=2)
        if int(receipt["status"]) != 1:
            raise ChainError(f"anchor transaction reverted: {_hex(tx_hash)}")
        commitment_topic = "0x" + digest
        logs = list(receipt.get("logs") or [])
        if not any(
            str(log.get("address", "")).lower() == registry.lower()
            and [_hex(topic).lower() for topic in log.get("topics", [])]
            == [commitment_topic]
            for log in logs
        ):
            raise ChainError("digest registry transaction omitted its commitment log")

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
            recipient=registry,
            contract_code_sha256=REGISTRY_CODE_SHA256,
            commitment_log_topic=commitment_topic,
            explorer_url=explorer_url,
        )
        write_json(output_path, anchor.to_dict())
        return anchor

    def verify(self, evidence_path: Path, anchor_path: Path) -> VerificationReport:
        self._assert_chain()
        evidence = read_json(evidence_path)
        stored_anchor = read_json(anchor_path)
        digest = evidence_file_digest(evidence_path)
        tx_hash = str(stored_anchor.get("transaction_hash", ""))
        if not tx_hash:
            raise VerificationError("anchor receipt has no transaction hash")

        try:
            transaction = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        except Exception as exc:
            raise VerificationError(
                f"HISTORY_UNAVAILABLE: could not retrieve anchor transaction {tx_hash}"
            ) from exc

        chain_payload = _input_bytes(transaction["input"])
        expected_payload = anchor_payload(digest)
        current_block = int(self.w3.eth.block_number)
        block_number = int(receipt["blockNumber"])
        canonical_block = self.w3.eth.get_block(block_number)
        confirmations = max(0, current_block - block_number + 1)
        recipient = str(transaction["to"])
        code = bytes(self.w3.eth.get_code(recipient))
        stored_value = int.from_bytes(
            bytes(self.w3.eth.get_storage_at(recipient, int(digest, 16))), "big"
        )
        commitment_topic = "0x" + digest
        event_found = any(
            str(log.get("address", "")).lower() == recipient.lower()
            and [_hex(topic).lower() for topic in log.get("topics", [])]
            == [commitment_topic]
            for log in receipt.get("logs", [])
        )

        checks = {
            "evidence_schema": evidence.get("schema") == SCHEMA
            and evidence.get("status") == "FINAL",
            "anchor_schema": stored_anchor.get("schema") == ANCHOR_SCHEMA,
            "digest_recomputed": stored_anchor.get("evidence_sha256") == digest,
            "receipt_payload": stored_anchor.get("payload_hex") == "0x" + expected_payload.hex(),
            "transaction_payload": chain_payload == expected_payload,
            "contract_bytecode": hashlib.sha256(code).hexdigest()
            == stored_anchor.get("contract_code_sha256")
            == REGISTRY_CODE_SHA256,
            "stored_commitment": stored_value == 1,
            "commitment_log": event_found
            and stored_anchor.get("commitment_log_topic") == commitment_topic,
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
            reproduction=self._reproduction_status(evidence),
        )
        if not report.verified:
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise VerificationError(f"on-chain verification failed: {failed}")
        return report

    def _reproduction_status(self, evidence: dict[str, Any]) -> dict[str, str]:
        provenance = evidence.get("provenance") or {}

        def local_status(filename: str, expected: Any) -> str:
            if not expected:
                return "NOT_APPLICABLE"
            path = self.settings.project_root / filename
            if not path.is_file():
                return "UNAVAILABLE"
            return (
                "VERIFIED"
                if hashlib.sha256(path.read_bytes()).hexdigest() == expected
                else "MISMATCH"
            )

        return {
            "model_lock": local_status(
                "model-lock.json", provenance.get("model_lock_sha256")
            ),
            "dependency_lock": local_status(
                "requirements.lock", provenance.get("dependency_lock_sha256")
            ),
            "calibration_report": (
                "UNAVAILABLE"
                if provenance.get("calibration_report_sha256")
                else "NOT_APPLICABLE"
            ),
            "test_report": (
                "UNAVAILABLE"
                if provenance.get("test_report_sha256")
                else "NOT_APPLICABLE"
            ),
        }
