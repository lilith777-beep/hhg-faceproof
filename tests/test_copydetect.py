from __future__ import annotations

from pathlib import Path

import pytest

from faceproof.copydetect import SSCD, load_sscd_engine
from faceproof.errors import FaceInputError


def test_sscd_model_is_pinned() -> None:
    assert SSCD.filename == "sscd_disc_mixup.torchscript.pt"
    assert SSCD.dimensions == 512
    assert len(SSCD.sha256) == 64
    int(SSCD.sha256, 16)


def test_sscd_auto_mode_has_explicit_legacy_fallback(tmp_path: Path) -> None:
    assert load_sscd_engine(tmp_path, mode="auto", device="cpu", batch_size=1) is None


def test_sscd_required_mode_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FaceInputError, match="SSCD model is missing"):
        load_sscd_engine(tmp_path, mode="required", device="cpu", batch_size=1)
