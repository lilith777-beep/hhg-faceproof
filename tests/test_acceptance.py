from pathlib import Path

import pytest
from typer.testing import CliRunner

from faceproof import cli
from faceproof.acceptance import _FixtureSource, _rpc_is_local
from faceproof.config import Settings


def _settings(tmp_path: Path, rpc_url: str = "http://127.0.0.1:8545") -> Settings:
    return Settings(
        project_root=tmp_path,
        model_dir=tmp_path / "models",
        artifact_dir=tmp_path / "artifacts",
        mastodon_instance="https://mastodon.social",
        mastodon_tag=None,
        mastodon_max_pages=1,
        mastodon_page_size=10,
        rpc_url=rpc_url,
        private_key=None,
        signer_mode="unlocked",
        expected_signer=None,
        expected_chain_id=31_337,
        explorer_tx_url="",
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


def test_fixture_source_is_explicitly_synthetic() -> None:
    batch = _FixtureSource().discover()
    assert batch.provider == "synthetic-placeholder-source"
    assert batch.live_query is False
    assert len(batch.hits) == 2
    assert {hit.post_id for hit in batch.hits} == {"positive", "negative"}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8545", True),
        ("http://localhost:8545", True),
        ("https://rpc.example", False),
    ],
)
def test_rpc_locality(tmp_path: Path, url: str, expected: bool) -> None:
    assert _rpc_is_local(_settings(tmp_path, url)) is expected


def test_acceptance_command_reports_all_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        cli,
        "run_placeholder_acceptance",
        lambda settings: {
            "positive_score": 0.93,
            "negative_score": 0.40,
            "threshold": 0.50,
            "selected_post_id": "positive",
            "timings": {"total_ms": 100},
            "evidence_path": evidence,
            "checks": {"positive_selected": True},
        },
    )
    monkeypatch.setattr(
        cli,
        "run_chain_acceptance",
        lambda settings, path: {
            "transaction_hash": "0x123",
            "block_number": 1,
            "confirmations": 1,
            "verification_checks": {"payload": True},
            "tamper_rejected": True,
            "anchor_path": tmp_path / "anchor.json",
        },
    )
    result = CliRunner().invoke(cli.app, ["acceptance"])
    assert result.exit_code == 0
    assert "positive selected" in result.stdout
    assert "tamper rejected" in result.stdout
    assert "0x123" in result.stdout
