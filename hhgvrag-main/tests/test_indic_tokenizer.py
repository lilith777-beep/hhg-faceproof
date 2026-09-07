"""Devanagari/Indic tokenizer: \w+ dropped matras+virama (competitor-surfaced bug,
confirmed live 2026-08-17). Words must stay whole; sentence separators must still split."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from textnorm import tokens, WORD_RE  # noqa: E402


def test_devanagari_words_stay_whole():
    assert tokens("मधुमेह") == ["मधुमेह"]          # was ['मध','म','ह'] under \w+
    assert tokens("क्या") == ["क्या"]               # virama no longer splits
    assert tokens("कॉर्पोरेशन") == ["कॉर्पोरेशन"]


def test_bengali_and_mixed():
    assert tokens("প্রধানমন্ত্রী") == ["প্রধানমন্ত্রী"]
    assert tokens("what is मधुमेह?") == ["what", "is", "मधुमेह"]


def test_danda_and_space_still_delimit():
    assert tokens("पहला वाक्य। दूसरा") == ["पहला", "वाक्य", "दूसरा"]
    assert len(tokens("क्या है")) == 2
