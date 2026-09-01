from pathlib import Path

import pytest

from faceproof.config import ANVIL_CHAIN_ID, Settings


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FACEPROOF_CHAIN_ID",
        "FACEPROOF_FACE_THRESHOLD",
        "FACEPROOF_PREFILTER_FACE_THRESHOLD",
        "FACEPROOF_MAX_CANDIDATES",
        "FACEPROOF_MAX_FINALISTS",
        "FACEPROOF_MASTODON_PAGE_SIZE",
        "FACEPROOF_MASTODON_MAX_PAGES",
        "FACEPROOF_SIGNER_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_local_anvil(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    settings = Settings.load(tmp_path)
    assert settings.expected_chain_id == ANVIL_CHAIN_ID
    assert settings.rpc_url == "http://127.0.0.1:8545"
    assert settings.signer_mode == "unlocked"
    assert settings.face_threshold == 0.5
    assert settings.face_threshold_source.startswith("conservative project default")


def test_explicit_threshold_records_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("FACEPROOF_FACE_THRESHOLD", "0.61")
    settings = Settings.load(tmp_path)
    assert settings.face_threshold == 0.61
    assert settings.face_threshold_source == "FACEPROOF_FACE_THRESHOLD"


def test_rejects_invalid_face_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FACEPROOF_FACE_THRESHOLD", "1.2")
    with pytest.raises(ValueError, match="between 0 and 1"):
        Settings.load(tmp_path)


def test_rejects_prefilter_above_final(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FACEPROOF_FACE_THRESHOLD", "0.5")
    monkeypatch.setenv("FACEPROOF_PREFILTER_FACE_THRESHOLD", "0.6")
    with pytest.raises(ValueError, match="below the final"):
        Settings.load(tmp_path)
