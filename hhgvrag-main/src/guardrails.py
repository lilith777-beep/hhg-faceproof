"""
guardrails.py — the "knows when NOT to answer" layer (requirement #6).

Three independent gates, all pure/injectable so they unit-test on CPU:
1. input safety      — unsafe/inappropriate query -> REFUSE.
2. OOD gate          — top retrieval similarity below τ -> the corpus doesn't cover it -> ABSTAIN.
3. grounding check   — generated answer not supported by retrieved context -> ABSTAIN.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from schemas import GroundingResult, OODResult, SafetyResult

from textnorm import INDIC_STOP, WORD_RE as _TOK   # Indic-correct tokens + function words
_STOP = set(
    "a an the of to in on for and or is are was were be been being this that these those "
    "it its as at by with from into about over under how what when where who whom which why "
    "do does did can could should would will may might i you he she they we me my your".split()
) | INDIC_STOP   # same denominator hygiene for the 13 Indic scripts as for English
# negation / polarity markers (Latin + Devanagari/Urdu) — a claim that negates but whose context
# doesn't is a likely contradiction, which pure lexical overlap would wave through.
_NEG = {"not", "no", "never", "cannot", "cant", "n't", "without", "neither", "nor", "none",
        "false", "incorrect", "unlike", "नहीं", "नही", "ना", "मत", "बिना", "نہیں", "نہ",
        # WORD_RE splits "doesn't" -> {doesn, t}, so the contraction STEMS are the tokens
        # that actually appear ("n't" alone never matches). "can"/"won" excluded: too
        # common as ordinary words to treat as polarity markers.
        "don", "doesn", "isn", "wasn", "weren", "couldn", "shouldn", "wouldn",
        "aren", "hasn", "haven", "didn", "ain"}


def _content_tokens(text: str) -> set:
    return {t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1}


# ---- 1. input safety -------------------------------------------------------------------
@runtime_checkable
class SafetyClassifier(Protocol):
    def check(self, text: str) -> SafetyResult: ...


class KeywordSafety:
    """CPU-testable stand-in for Llama-Guard: flags a few unsafe intents by pattern."""
    PATTERNS = {
        "weapons": r"\b(build|make|assemble|construct|create|craft)\b.{0,40}"
                   r"\b(bomb|explosive|gun|weapon|grenade|ied|molotov|pipe\s+bomb)s?\b",
        # CBRN agents flag on production/acquisition intent OR recipe-shaped questions —
        # "ricin" passed the old lexicon entirely (live finding)
        "cbrn": r"\b(make|produce|synthesi[sz]e|extract|obtain|brew|culture|weaponi[sz]e)\b"
                r".{0,40}\b(ricin|sarin|vx|tabun|soman|novichok|anthrax|botulinum|"
                r"nerve\s+agent|mustard\s+gas|chlorine\s+gas)\b"
                r"|\b(ricin|sarin|novichok|anthrax|nerve\s+agent)\b.{0,30}"
                r"\b(recipe|instructions|at\s+home|how\s+to|step)",
        "self_harm": r"\b(kill|hurt|harm)\s+myself\b|\bsuicid",
        "violence": r"\b(how\s+to|steps?\s+to|ways?\s+to)\b.{0,20}"
                    r"\b(kill|murder|attack|poison)\s+(a|someone|people|my)\b",
        "illicit": r"\b(synthesi[sz]e|cook|make|produce|manufactur\w*)\b.{0,20}"
                   r"\b(meth(?:amphetamine)?|cocaine|heroin|fentanyl)\b",
        # starter Indic tier (STT feeds native script; the English-only lexicon was blind
        # to all 13 non-English languages) — Llama-Guard wiring remains the real fix
        "weapons_indic": r"(बम|विस्फोटक|बंदूक|ज़हर|जहर)\W{0,20}(बना|कैसे\s*बना)"
                         r"|(बना\w*|कैसे)\W{0,20}(बम|विस्फोटक)"
                         r"|بم\W{0,20}بنا|کیسے\W{0,20}بم",
    }

    def check(self, text: str) -> SafetyResult:
        low = text.lower()
        for cat, pat in self.PATTERNS.items():
            if re.search(pat, low):
                return SafetyResult(safe=False, category=cat, detail="matched unsafe pattern")
        return SafetyResult(safe=True)


class LlamaGuardSafety:
    """Real backend (Modal): Llama-Guard-class moderation. Lazy import."""
    def __init__(self, generate_fn):
        self._gen = generate_fn  # a callable(prompt) -> str returning 'safe'/'unsafe\n<cat>'

    def check(self, text: str) -> SafetyResult:
        out = (self._gen(text) or "").strip().lower()
        if out.startswith("unsafe"):
            cat = out.split("\n", 1)[1].strip() if "\n" in out else None
            return SafetyResult(safe=False, category=cat, detail="llama-guard flagged")
        return SafetyResult(safe=True)


# ---- 2. off-topic / OOD gate -----------------------------------------------------------
def ood_gate(retrieved: Sequence, threshold: float, query: Optional[str] = None,
             lexical_min: float = 0.34) -> OODResult:
    """
    Domain-coverage gate. PRIMARY signal = true max DENSE cosine over the dense arm — never the
    RRF fused score (a ~0.016-scale rank score compared to a 0.32 cosine τ always fails, which
    silently discards the lexical arm and false-abstains on noisy-ASR queries the sparse arm
    rescued). SECONDARY (only if dense is below τ): lexical coverage of the query by any retrieved
    chunk — so a garbled-dense-but-lexically-strong query still answers.
    """
    dense_scores = [r.dense_score for r in retrieved if r.dense_score is not None]
    dense_top = max(dense_scores) if dense_scores else 0.0
    if dense_top >= threshold:
        return OODResult(in_domain=True, top_score=round(dense_top, 4), threshold=threshold,
                         signal="dense")

    lexical_top = 0.0
    qtok = _content_tokens(query) if query else set()
    if qtok:
        for r in retrieved:
            overlap = len(qtok & _content_tokens(r.text)) / len(qtok)
            lexical_top = max(lexical_top, overlap)
    admit = lexical_top >= lexical_min
    return OODResult(in_domain=admit, top_score=round(dense_top, 4), threshold=threshold,
                     lexical_top=round(lexical_top, 4), signal="lexical" if admit else "none")


# ---- 3. grounding / anti-hallucination -------------------------------------------------
def grounding_check(answer: str, retrieved: Sequence, min_support: float,
                    cited_ids: Optional[Sequence[str]] = None,
                    nli: Optional["Callable"] = None) -> GroundingResult:
    """
    Lexical support = fraction of the answer's content tokens present in the retrieved context.
    Optionally strengthened by an injected NLI entailment callable. cited chunks = those that
    actually contribute the answer's tokens (or the ones the generator cited).
    """
    ans_tokens = _content_tokens(answer)
    if not ans_tokens:
        return GroundingResult(grounded=False, support=0.0, detail="empty answer")

    context_tokens = set()
    per_chunk = {}
    for r in retrieved:
        ct = _content_tokens(r.text)
        per_chunk[r.chunk_id] = ct
        context_tokens |= ct

    support = len(ans_tokens & context_tokens) / len(ans_tokens)

    # which chunks the answer actually draws from (>=2 shared content tokens)
    contributing = [cid for cid, ct in per_chunk.items() if len(ans_tokens & ct) >= 2]
    cited = list(cited_ids) if cited_ids else contributing

    grounded = support >= min_support
    detail = None if grounded else "insufficient support in context"

    if nli is not None:
        # authoritative entailment check (quality mode): fixes both failure modes of lexical
        # overlap — a faithful PARAPHRASE (low overlap) is entailed; a fluent CONTRADICTION that
        # reuses context vocab (high overlap) is not.
        prem_ids = set(cited_ids) if cited_ids else {r.chunk_id for r in retrieved}
        ctx = " ".join(r.text for r in retrieved if r.chunk_id in prem_ids)
        grounded = bool(nli(premise=ctx, hypothesis=answer))
        detail = None if grounded else "not entailed by cited context"
    elif grounded:
        # no-NLI cheap contradiction guard: an answer that NEGATES while its supporting context
        # does not is a likely polarity flip that lexical overlap alone waves through.
        ctx_text = " ".join(r.text for r in retrieved if r.chunk_id in set(cited or contributing))
        ans_raw = set(_TOK.findall(answer.lower()))
        ctx_raw = set(_TOK.findall(ctx_text.lower()))
        if (ans_raw & _NEG) and not (ctx_raw & _NEG):
            grounded, detail = False, "possible contradiction (answer negates, context does not)"

    return GroundingResult(grounded=grounded, support=round(support, 3),
                           cited_chunk_ids=cited or contributing, detail=detail)
