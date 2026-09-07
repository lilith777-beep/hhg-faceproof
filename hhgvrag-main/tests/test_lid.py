"""LID v2: romanized-vernacular identification + script-matched localization
(live finding 2026-08-17: romanized-Hindi askers got Devanagari LLM answers + English
fixed messages)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness import _localized  # noqa: E402
from router import detect_language_ex  # noqa: E402


def test_romanized_hindi_detected():
    lid = detect_language_ex("diabetes ke lakshan kya hai")
    assert (lid.lang, lid.script, lid.romanized) == ("hi", "latin", True)
    lid = detect_language_ex("tax refund kitne din me aata hai")
    assert lid.lang == "hi" and lid.romanized


def test_romanized_tamil_bengali_detected():
    assert detect_language_ex("corporation na enna eppadi work aagum").lang == "ta"
    assert detect_language_ex("credit score ki keno kome").lang == "bn"


def test_english_never_false_flags():
    for q in ("how is the weather today", "what is a corporation",
              "how long does a tax refund take", "main street directions",
              "ki is a japanese word"):
        lid = detect_language_ex(q)
        assert not lid.romanized, q
        assert lid.lang == "en", q


def test_native_script_wins():
    lid = detect_language_ex("मधुमेह के लक्षण क्या हैं?")
    assert (lid.lang, lid.script, lid.romanized) == ("hi", "native", False)


def test_localized_romanized_chain():
    # exact romanized table
    assert _localized("abstain", "hi-r", "FB").startswith("Mere gyaan")
    # kind missing from roman table (none here) -> native -> present
    assert _localized("greet", "hi-r", "FB").startswith("Namaste!")
    # unknown roman lang falls through to native then fallback
    assert _localized("abstain", "te-r", "FB").startswith("నా విజ్ఞాన")
    assert _localized("abstain", "xx-r", "FB") == "FB"


def test_romanized_residual_glosses_content_words():
    from router import romanized_residual
    assert romanized_residual("corporation kya hai", "hi") == "corporation"
    assert romanized_residual("tax refund kitne din me aata hai", "hi") == \
        "tax refund days arrive"
    assert romanized_residual("asthma ka ilaj kaise hota hai", "hi") == "asthma treatment"
    assert romanized_residual("diabetes ke lakshan kya hai", "hi") == "diabetes symptoms"
    # unchanged text -> no second arm. (Plain-English safety is enforced one layer up:
    # the harness only calls this for LID-flagged queries — see
    # test_english_never_false_flags.)
    assert romanized_residual("corporation", "hi") == ""
