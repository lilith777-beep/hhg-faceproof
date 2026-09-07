"""Regression: web-list enumeration artifacts must never leak into composed answers
(live finding 2026-08-17: '2 Any impact... 1 If you have... [2]' numbering soup)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generation import _LIST_ARTIFACT, ExtractiveGenerator  # noqa: E402


class _Chunk:
    def __init__(self, text, cid="c1"):
        self.text = text
        self.chunk_id = cid
        self.payload = {"language": "en"}


def test_list_artifact_patterns():
    strip = lambda s: _LIST_ARTIFACT.sub("", s)
    assert strip("2 Any impact on your credit score is minimal.") == \
        "Any impact on your credit score is minimal."
    assert strip("1. If you have other bad marks, wait.") == \
        "If you have other bad marks, wait."
    assert strip("[3] Skipping payments damages your score.") == \
        "Skipping payments damages your score."
    assert strip("A: The disc gives out 49 Celebis.") == "The disc gives out 49 Celebis."
    assert strip("• Pay bills on time.") == "Pay bills on time."


def test_legitimate_numbers_survive():
    strip = lambda s: _LIST_ARTIFACT.sub("", s)
    assert strip("2 million people are affected.") == "2 million people are affected."
    assert strip("2018 was a record year.") == "2018 was a record year."   # 4-digit spared
    assert strip("401k plans defer taxes.") == "401k plans defer taxes."


def test_composed_answer_has_no_bare_leading_digits():
    gen = ExtractiveGenerator()
    ctx = [_Chunk("2 Any impact on your credit score from opening a savings account is "
                  "minimal if you have good credit. 1 If you have other bad marks on your "
                  "credit score you might not want to open new accounts.")]
    out = gen.generate("does opening a savings account affect my credit score", ctx)
    assert out.text, "should answer"
    assert not out.text.startswith(("1 ", "2 ", "3 ")), out.text
    assert " 1 If" not in out.text and " 2 Any" not in out.text, out.text
    assert "[1]" in out.text          # real citation markers stay
