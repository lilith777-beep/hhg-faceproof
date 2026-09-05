from __future__ import annotations

import copy
import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .chain import EthereumAnchor
from .config import Settings
from .copydetect import load_sscd_engine
from .errors import ChainError, VerificationError
from .faces import FaceEngine
from .integrity import write_json
from .models import SearchBatch, SearchHit
from .pipeline import DiscoveryPipeline
from .remote import RemoteImage


class _FixtureSource:
    name = "synthetic-placeholder-source"

    def discover(self) -> SearchBatch:
        hits = [
            _fixture_hit("negative", "different-person.png"),
            _fixture_hit("positive", "same-person.png"),
        ]
        return SearchBatch(
            provider=self.name,
            live_query=False,
            scope="two bundled fictional placeholder candidates",
            endpoint="fixture://bundled-synthetic-media",
            fetched_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            pages_fetched=1,
            statuses_scanned=len(hits),
            media_scanned=len(hits),
            hits=hits,
        )


def _fixture_hit(post_id: str, filename: str) -> SearchHit:
    url = f"https://placeholder.invalid/media/{filename}"
    return SearchHit(
        post_url=f"https://placeholder.invalid/@fixture/{post_id}",
        canonical_uri=f"https://placeholder.invalid/users/fixture/statuses/{post_id}",
        post_id=post_id,
        author_id="synthetic-fixture",
        author_handle="fixture",
        created_at="2026-09-01T00:00:00Z",
        content_text="Synthetic placeholder; not a live-search claim.",
        media_id=f"media-{post_id}",
        image_url=url,
        preview_url=url,
        image_width=1254,
        image_height=1254,
    )


class _FixtureFetcher:
    def __init__(self, fixture_dir: Path) -> None:
        self.fixture_dir = fixture_dir

    def fetch(self, url: str) -> RemoteImage:
        filename = url.rsplit("/", 1)[-1]
        path = (self.fixture_dir / filename).resolve()
        if path.parent != self.fixture_dir.resolve() or not path.is_file():
            raise FileNotFoundError(filename)
        return RemoteImage(url, path.read_bytes(), "image/png")


def _score_fixture_pair(engine: FaceEngine, fixture_dir: Path, candidate_name: str) -> float:
    query_image = engine.decode((fixture_dir / "query.png").read_bytes())
    query = engine.require_single_query_face(query_image)
    candidate_image = engine.decode((fixture_dir / candidate_name).read_bytes())
    candidates = engine.detect_and_encode(candidate_image)
    score, _, _ = engine.best_match(query, candidates)
    return score


def run_placeholder_acceptance(settings: Settings) -> dict[str, Any]:
    fixture_dir = Path(__file__).with_name("fixtures")
    engine = FaceEngine(settings.model_dir, detection_threshold=settings.detection_threshold)
    positive_score = _score_fixture_pair(engine, fixture_dir, "same-person.png")
    negative_score = _score_fixture_pair(engine, fixture_dir, "different-person.png")
    copy_engine = load_sscd_engine(
        settings.model_dir,
        mode=settings.sscd_mode,
        device=settings.sscd_device,
        batch_size=settings.sscd_batch_size,
    )
    sscd_positive_score: float | None = None
    sscd_negative_score: float | None = None
    if copy_engine is not None:
        decoded = [
            engine.decode((fixture_dir / name).read_bytes())
            for name in ("query.png", "same-person.png", "different-person.png")
        ]
        query_copy, positive_copy, negative_copy = copy_engine.encode_many(decoded)
        sscd_positive_score = float(np.dot(query_copy, positive_copy))
        sscd_negative_score = float(np.dot(query_copy, negative_copy))
    output_dir = settings.artifact_dir / "acceptance"
    evidence_path = output_dir / "evidence.json"
    bundle, _ = DiscoveryPipeline(
        settings=settings,
        face_engine=engine,
        candidate_source=_FixtureSource(),
        image_fetcher=_FixtureFetcher(fixture_dir),
        copy_engine=copy_engine,
    ).discover(
        fixture_dir / "query.png",
        consent_asserted=True,
        output_path=evidence_path,
    )
    checks = {
        "positive_above_threshold": positive_score >= settings.face_threshold,
        "negative_below_threshold": negative_score < settings.face_threshold,
        "score_separation": positive_score - negative_score >= 0.30,
        "positive_selected": bundle.match["post_id"] == "positive",
        "one_verified_match": bundle.match["verified_match_count"] == 1,
        "not_ambiguous": bundle.match["ambiguous"] is False,
        "sscd_enabled": copy_engine is not None,
        "sscd_separates_fixture": bool(
            sscd_positive_score is not None
            and sscd_negative_score is not None
            and sscd_positive_score - sscd_negative_score >= 0.20
        ),
        "faiss_dual_retrieval": bool(
            bundle.search["vector_backend"] == "faiss.IndexFlatIP"
            and "face-faiss" in bundle.match["retrieval_channels"]
            and "sscd-faiss" in bundle.match["retrieval_channels"]
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"placeholder biometric acceptance failed: {failed}")
    return {
        "positive_score": round(positive_score, 6),
        "negative_score": round(negative_score, 6),
        "sscd_positive_score": (
            round(sscd_positive_score, 6) if sscd_positive_score is not None else None
        ),
        "sscd_negative_score": (
            round(sscd_negative_score, 6) if sscd_negative_score is not None else None
        ),
        "threshold": settings.face_threshold,
        "selected_post_id": bundle.match["post_id"],
        "timings": bundle.search["timings"],
        "evidence_path": evidence_path,
        "checks": checks,
    }


def _local_anvil_path(settings: Settings) -> str | None:
    installed = shutil.which("anvil")
    if installed:
        return installed
    local = settings.project_root / ".tools" / "foundry" / "anvil.exe"
    return str(local) if local.is_file() else None


def _rpc_is_local(settings: Settings) -> bool:
    host = (urlsplit(settings.rpc_url or "").hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


@contextmanager
def ensure_acceptance_chain(settings: Settings) -> Iterator[EthereumAnchor]:
    try:
        existing = EthereumAnchor(settings)
    except ChainError:
        existing = None
    if existing is not None:
        yield existing
        return

    if not _rpc_is_local(settings):
        raise ChainError("configured non-local RPC is unreachable; refusing to start a local chain")
    executable = _local_anvil_path(settings)
    if not executable:
        raise ChainError(
            "Anvil is not installed; install Foundry or rerun acceptance with --skip-chain"
        )
    command = [
        executable,
        "--chain-id",
        str(settings.expected_chain_id),
        "--host",
        "127.0.0.1",
        "--port",
        str(urlsplit(settings.rpc_url or "").port or 8545),
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    try:
        deadline = time.monotonic() + 10
        anchor: EthereumAnchor | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ChainError("local Anvil process exited before becoming ready")
            try:
                anchor = EthereumAnchor(settings)
                break
            except ChainError:
                time.sleep(0.25)
        if anchor is None:
            raise ChainError("local Anvil did not become ready within 10 seconds")
        yield anchor
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_chain_acceptance(settings: Settings, evidence_path: Path) -> dict[str, Any]:
    output_dir = evidence_path.parent
    anchor_path = output_dir / "anchor.json"
    tampered_path = output_dir / "evidence-tampered.json"
    with ensure_acceptance_chain(settings) as chain:
        receipt = chain.anchor(evidence_path, anchor_path)
        report = chain.verify(evidence_path, anchor_path)
        tampered = copy.deepcopy(json.loads(evidence_path.read_text("utf-8")))
        tampered["match"]["author_handle"] = "tampered-fixture"
        write_json(tampered_path, tampered)
        tamper_rejected = False
        try:
            chain.verify(tampered_path, anchor_path)
        except VerificationError:
            tamper_rejected = True
        if not tamper_rejected:
            raise AssertionError("tampered evidence unexpectedly passed blockchain verification")
        return {
            "transaction_hash": receipt.transaction_hash,
            "block_number": report.block_number,
            "confirmations": report.confirmations,
            "verification_checks": report.checks,
            "tamper_rejected": tamper_rejected,
            "anchor_path": anchor_path,
        }
