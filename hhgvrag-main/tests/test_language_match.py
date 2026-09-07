# -*- coding: utf-8 -*-
"""Language-matched responses: fixed messages localize to the asker's language, and
extraction prefers same-language passages. (Competitive gap vs pucho.me, 2026-08-16.)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api import build_local_harness           # noqa: E402
from generation import ExtractiveGenerator    # noqa: E402
from harness import _localized                # noqa: E402
from schemas import Decision, Query, RetrievedChunk  # noqa: E402


def test_hindi_ood_abstains_in_hindi():
    h = build_local_harness()
    r = h.answer(Query(text="वेस्टेरोस का राजा अभी कौन है?"))
    assert r.decision in (Decision.ABSTAIN_OOD, Decision.ABSTAIN_UNGROUNDED)
    assert "ज्ञान आधार" in r.answer, f"abstain not localized: {r.answer}"


def test_bengali_error_and_tamil_unsafe_strings_exist():
    assert "জ্ঞানভাণ্ডারে" in _localized("abstain", "bn", "x")
    assert _localized("unsafe", "ta", "x").endswith("முடியாது.")
    assert "دوبارہ" in _localized("error", "ur", "x")
    assert _localized("abstain", "zz", "FALLBACK") == "FALLBACK"


def test_hindi_greeting_localized():
    h = build_local_harness()
    r = h.answer(Query(text="नमस्ते"))
    if r.decision == Decision.SMALL_TALK:
        assert "नमस्ते" in r.answer or "प्रश्न" in r.answer, f"greet not localized: {r.answer}"


def test_extraction_prefers_asker_language():
    gen = ExtractiveGenerator()
    ctx = [RetrievedChunk(chunk_id="en1", score=1.0,
                          text="A corporation is a company or group of people authorized "
                               "to act as a single entity in law.",
                          payload={"language": "en"}),
           RetrievedChunk(chunk_id="hi1", score=0.9,
                          text="कॉर्पोरेशन एक कंपनी या व्यक्तियों का समूह है जो कानून में एकल इकाई "
                               "के रूप में कार्य करने के लिए अधिकृत है।",
                          payload={"language": "hi"})]
    out = gen.generate("कॉर्पोरेशन क्या है?", ctx)
    assert "कॉर्पोरेशन एक कंपनी" in out.text, f"Hindi passage should win: {out.text}"


def test_english_query_unaffected():
    gen = ExtractiveGenerator()
    ctx = [RetrievedChunk(chunk_id="en1", score=1.0,
                          text="Photosynthesis converts sunlight into chemical energy in plants.",
                          payload={"language": "en"})]
    out = gen.generate("what is photosynthesis", ctx)
    assert "Photosynthesis converts" in out.text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} language-match tests passed")
