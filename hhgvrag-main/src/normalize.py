"""
normalize.py — the noisy-ASR robustness layer (R3).

ASR output is dirty: filler words ("um", "uh"), disfluent lead-ins ("so, like, tell me…"),
casing/spacing noise, homophones and near-miss spellings, and a confidence score that says how
much to trust any of it. This module cleans the transcript BEFORE retrieval, offers phonetic /
fuzzy signals so exact-token lexical retrieval survives small errors, and turns the STT
confidence into concrete retrieval adjustments (widen top-k, relax the gate) instead of ignoring it.

Pure-stdlib and CPU — unit-testable with no models.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Sequence

_WORD = re.compile(r"\w+", re.UNICODE)

# spoken filler / disfluencies (English + a few common Hindi/Hinglish) — safe to drop
_FILLERS = {
    "um", "uh", "umm", "uhh", "erm", "er", "hmm", "ah", "eh", "like", "actually",
    "basically", "literally", "sorta", "kinda", "yaar", "matlab", "acha", "achha", "toh",
}
# disfluent lead-ins to strip only at the START of a query
_LEAD_INS = ("so ", "so,", "ok ", "okay ", "well ", "hey ", "um ", "uh ", "like ",
             "please ", "can you ", "could you ", "i want to know ", "tell me ",
             "i wanted to ask ", "do you know ")

# common English ASR homophone/substitution folds (query-side, conservative)
_HOMOPHONES = {
    "quadrant": "qdrant", "hydrant": "qdrant", "cadrant": "qdrant",
    "sarvamm": "sarvam", "saravam": "sarvam", "sarwam": "sarvam",
    "rag model": "rag", "raga": "rag",
    "goa's": "goa", "gore": "goa",
}


@dataclass
class NormalizedQuery:
    text: str                       # the cleaned query used for retrieval/generation
    original: str                   # what the ASR gave us (kept for display)
    tokens: list                    # content tokens of `text`
    phonetic: list                  # phonetic keys (fuzzy lexical rescue)
    trigrams: list                  # char trigrams (near-miss spelling overlap)
    expand: bool = False            # low confidence -> widen retrieval
    top_k: int = 8                  # confidence-adjusted retrieval depth
    ood_relax: float = 0.0          # subtract from the OOD threshold when confidence is low


def _metaphone_lite(token: str) -> str:
    """Cheap phonetic key (Soundex-ish): drop vowels after the first char, collapse repeats,
    fold near-homophone consonants. Good enough to bucket typos/homophones of the same word."""
    t = re.sub(r"[^a-z]", "", token.lower())
    if not t:
        return ""
    t = t.replace("ph", "f")                     # digraph fold (maketrans keys must be 1 char)
    folds = str.maketrans({"q": "k", "c": "k", "x": "k", "z": "s", "v": "f",
                           "w": "f", "y": "i", "j": "g"})
    head, rest = t[0], t[1:]
    rest = re.sub(r"[aeiou]", "", rest)          # keep only consonant skeleton
    key = (head + rest).translate(folds)
    return re.sub(r"(.)\1+", r"\1", key)[:6]     # collapse doubled letters


def _trigrams(text: str) -> list:
    s = "".join(_WORD.findall(text.lower()))
    return [s[i:i + 3] for i in range(len(s) - 2)] if len(s) >= 3 else ([s] if s else [])


class QueryNormalizer:
    """Clean + enrich a (possibly noisy) transcript for robust retrieval."""

    def __init__(self, drop_fillers: bool = True, fold_homophones: bool = True):
        self.drop_fillers = drop_fillers
        self.fold_homophones = fold_homophones

    def _clean(self, text: str) -> str:
        t = unicodedata.normalize("NFC", text or "").strip()
        if self.fold_homophones:
            for bad, good in _HOMOPHONES.items():
                t = re.sub(rf"\b{re.escape(bad)}\b", good, t, flags=re.IGNORECASE)
        if self.drop_fillers:
            toks = [w for w in t.split() if w.lower().strip(",.?!;:") not in _FILLERS]
            t = " ".join(toks)
        # strip leading disfluent lead-ins, repeatedly ("so, tell me what is X" -> "what is X")
        low = t.lower()
        changed = True
        while changed:
            changed = False
            for lead in _LEAD_INS:
                if low.startswith(lead):
                    t = t[len(lead):].lstrip(" ,")
                    low = t.lower()
                    changed = True
                    break
        return re.sub(r"\s+", " ", t).strip(" ,")

    def confidence_params(self, confidence: float, base_top_k: int = 8) -> tuple:
        """Low STT confidence -> cast a wider net and relax the OOD gate (don't silently abstain)."""
        if confidence is None or confidence >= 0.75:
            return False, base_top_k, 0.0
        if confidence >= 0.5:
            return True, int(base_top_k * 1.5), 0.03
        return True, base_top_k * 2, 0.06        # very low confidence -> widest net, most lenient gate

    def normalize(self, text: str, confidence: float = 1.0, base_top_k: int = 8) -> NormalizedQuery:
        cleaned = self._clean(text)
        toks = [w for w in _WORD.findall(cleaned.lower()) if len(w) > 1]
        expand, top_k, relax = self.confidence_params(confidence, base_top_k)
        return NormalizedQuery(
            text=cleaned or (text or "").strip(),
            original=text,
            tokens=toks,
            phonetic=[k for k in (_metaphone_lite(w) for w in toks) if k],
            trigrams=_trigrams(cleaned),
            expand=expand, top_k=top_k, ood_relax=relax,
        )
