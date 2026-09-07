"""
router.py — the query router (R2), front-half of the "strong classifier for correct retrieval".

Before we retrieve, classify the query so we route it correctly:
- language / script  (telemetry + optional metadata filter — NOT a hard cross-lingual filter),
- intent            (qa | chitchat | meta | unsafe): a greeting or "who are you" is a real user
                     turn that should get an intentional reply, not an OOD abstain or a wasted
                     retrieval; genuine questions go down the RAG path.

Heuristic + pluggable: `HeuristicRouter` is dependency-free and CPU-testable; the same `Router`
Protocol admits a small classifier model on Modal later without touching the harness.
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from schemas import RouteResult

# Unicode script blocks -> language code (first match wins; covers the MSMARCO-XI Indic set)
_SCRIPTS = [
    ("hi", r"[ऀ-ॿ]"),   # Devanagari (Hindi/Marathi/Sanskrit/Nepali)
    ("bn", r"[ঀ-৿]"),   # Bengali/Assamese
    ("pa", r"[਀-੿]"),   # Gurmukhi (Punjabi)
    ("gu", r"[઀-૿]"),   # Gujarati
    ("or", r"[଀-୿]"),   # Odia
    ("ta", r"[஀-௿]"),   # Tamil
    ("te", r"[ఀ-౿]"),   # Telugu
    ("kn", r"[ಀ-೿]"),   # Kannada
    ("ml", r"[ഀ-ൿ]"),   # Malayalam
    ("ur", r"[؀-ۿ]"),   # Arabic script (Urdu)
]
_SCRIPTS = [(lang, re.compile(pat)) for lang, pat in _SCRIPTS]

_GREETING = re.compile(
    r"^\s*(hi|hii+|hey+|hello+|yo)(\s+there)?[\s!.?]*$"          # hi/hey/hello (+ optional "there")
    r"|^\s*(namaste|namaskar|salaam|vanakkam|"
    r"good\s+(morning|evening|afternoon)|how\s+are\s+you|what'?s\s+up|"
    r"thanks?|thank\s+you|bye+|ok|okay|cool)\b[\s!.?]*$",
    re.IGNORECASE)
_META = re.compile(
    r"\b(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"how\s+do\s+you\s+work|your\s+name|are\s+you\s+(a\s+)?(bot|ai|human)|"
    r"who\s+(made|built|created)\s+you)\b"
    # 'help' / 'what is this' are meta ONLY as the whole utterance — bare \bhelp\b turned
    # "what foods help lower cholesterol" (5 words, under the cap) into canned small talk
    r"|^\s*help\s*[!.?]*$"
    r"|^\s*what\s+is\s+this\s*[!.?]*$", re.IGNORECASE)


# Devanagari serves four languages; distinctive function words split Marathi/Nepali off
# the Hindi default (localization tables for mr/ne exist and were unreachable before).
# NOT \b: matras/virama are category Mc/Mn, not \w, so \b false-fires INSIDE words and
# fails after a trailing matra (the exact bug textnorm.py documents) — bound on "not a
# Devanagari codepoint or word char" instead, so a match is a whole written word.
_DEVA_B_L = r"(?<![\wऀ-ॿ])"
_DEVA_B_R = r"(?![\wऀ-ॿ])"
_DEVA_MR = re.compile(
    _DEVA_B_L + r"(आहे|आहेत|काय|आणि|कशी|कसे|मला|तुमच्या|मध्ये|म्हणजे)" + _DEVA_B_R)
_DEVA_NE = re.compile(
    _DEVA_B_L + r"(छ|छन्|छैन|गर्नुहोस्|कसरी|तपाईं|हुन्छ|भनेको|गर्ने)" + _DEVA_B_R)


def _deva_refine(text: str) -> str:
    if _DEVA_NE.search(text):
        return "ne"
    if _DEVA_MR.search(text):
        return "mr"
    return "hi"


def detect_language(text: str) -> str:
    for lang, rx in _SCRIPTS:
        if rx.search(text):
            return _deva_refine(text) if lang == "hi" else lang
    return "en"       # default: Latin script -> English (also the cross-lingual query case)


# ---- LID v2: romanized-vernacular identification (2026-08-17) ---------------------------
# Most Indian users type their language in Latin script ("diabetes ke lakshan kya hai").
# Script detection alone labels that English, so evidence preference, localized fixed
# messages, and the quality-LLM's output script all miss. Lexicons hold DISTINCTIVE
# function words only — ambiguous English collisions ("main", "the", "to", "us") are
# deliberately excluded. Flag requires >=2 distinct hits AND >=20% token coverage: a lone
# loanword never flips a genuinely-English query.
_ROMAN_LEX = {
    "hi": {"kya", "hai", "hain", "kaise", "kyun", "kyon", "kab", "kaun", "kahan", "nahi",
           "nahin", "mujhe", "aapka", "apka", "kitna", "kitne", "kitni", "hota", "hoti",
           "hote", "karna", "karne", "kare", "karo", "sakta", "sakte", "chahiye", "matlab",
           "batao", "bataiye", "wala", "wale", "mera", "meri", "tumhara", "iska", "uska"},
    "bn": {"ki", "keno", "kemon", "ache", "achhe", "amar", "tomar", "hobe", "kothay",
           "kivabe", "korte", "korbo", "bolo", "bolun", "tumi", "apni"},
    "ta": {"enna", "epdi", "eppadi", "illai", "illa", "venum", "vendum", "irukku",
           "iruku", "epo", "eppo", "yaru", "yaar", "enga", "unga", "panna", "pannu"},
}
_TOKEN_RX = re.compile(r"[a-z]+")


class LangID:
    __slots__ = ("lang", "script", "romanized")

    def __init__(self, lang: str, script: str, romanized: bool):
        self.lang, self.script, self.romanized = lang, script, romanized


# curated Roman-Hindi -> English glosses for CONTENT words that carry the question
# ("lakshan" never matches "symptoms" in embedding space the way its gloss does). This is
# the deterministic interim before IndicXlit transliteration; the ablation later measures
# gloss vs xlit vs both. Function words are handled by _ROMAN_LEX stripping.
_ROMAN_GLOSS = {
    "hi": {"lakshan": "symptoms", "ilaj": "treatment", "ilaaj": "treatment",
           "dawa": "medicine", "dawai": "medicine", "bimari": "disease",
           "bimaari": "disease", "din": "days", "hafta": "week", "mahina": "month",
           "saal": "year", "aata": "arrive", "aati": "arrive", "aayega": "arrive",
           "milta": "get", "milega": "get", "milti": "get", "lagta": "takes",
           "lagega": "takes", "paisa": "money", "paise": "money", "byaj": "interest",
           "karz": "loan", "fayda": "benefits", "fayde": "benefits",
           "nuksan": "side effects", "upay": "remedies", "tarika": "method",
           "tarike": "methods", "matlab": "meaning", "jankari": "information",
           "jaankari": "information", "sasta": "cheap", "mehnga": "expensive",
           "jaldi": "quickly", "zaroori": "necessary", "madad": "help",
           "shuru": "start", "band": "stop", "badhta": "increase",
           "badhana": "increase", "ghatana": "reduce", "khana": "food",
           "pani": "water", "neend": "sleep", "dard": "pain", "bukhar": "fever",
           "khansi": "cough", "pet": "stomach", "dil": "heart", "khoon": "blood",
           "kharch": "cost", "kharcha": "cost", "kaam": "work", "gharelu": "home"},
}


# short particles stripped from residuals ONLY (never used for detection — "is", "me",
# "par" collide with English, but by residual time the query is already LID-flagged)
_ROMAN_STRIP = {
    "hi": {"ka", "ke", "ki", "ko", "se", "par", "me", "mein", "aur", "ya", "is", "us",
           "ye", "wo", "na", "ho", "kar", "bhi", "to", "ab", "phir", "sab", "kuch"},
}


# romanized-lexicon words that are ALSO real English words or common brands/names. They may
# SUPPORT a romanized flag but can never be the strong evidence that triggers one — otherwise
# grammatical English ("the KI salt and the ache it causes") and brand queries ("karo syrup
# mera brand") get flagged and corrupted by transliteration into Devanagari garbage.
_ROMAN_AMBIG = {"ache", "ki", "din", "band", "epo", "panna", "karo", "mera", "meri", "pani",
                "no", "me", "us", "is", "to", "so", "ho", "na", "ya", "hi"}


def is_romanized_indic(text: str):
    """STRICT romanized-Indic detector for the TYPED transliteration gate — much lower
    false-positive rate than detect_language_ex().romanized (measured 1 FP/1 FN vs 6/3 on an
    adversarial set). Returns (True, lang) only when the evidence is unlikely to be English:
      - any native-script char anywhere -> already in script, NEVER transliterate (mixed/native)
      - >=2 DISTINCT unambiguous function words (lexicon minus _ROMAN_AMBIG), OR
      - 1 unambiguous function word + 1 content-gloss word (content-led Hindi like
        'bukhar ka ilaj batao'), at >=20% token coverage.
    English/brand collisions land only in _ROMAN_AMBIG and so can never form the strong pair."""
    for _lang, rx in _SCRIPTS:
        if rx.search(text):
            return (False, "")
    toks = _TOKEN_RX.findall(text.lower())
    if len(toks) < 2:
        return (False, "")
    distinct = set(toks)
    for lang, lex in _ROMAN_LEX.items():
        lex_hits = {t for t in distinct if t in lex}
        gloss_hits = {t for t in distinct if t in _ROMAN_GLOSS.get(lang, {})}
        strong = lex_hits - _ROMAN_AMBIG
        cov = len(lex_hits | gloss_hits) / len(distinct)
        if cov >= 0.20 and (len(strong) >= 2 or (len(strong) >= 1 and len(gloss_hits) >= 1)):
            return (True, lang)
    return (False, "")


def romanized_residual(text: str, lang: str) -> str:
    """Content-carrier form of a romanized query: strip function words (_ROMAN_LEX +
    _ROMAN_STRIP particles), gloss known Roman-Hindi content words to English
    (_ROMAN_GLOSS). 'tax refund kitne din me aata hai' -> 'tax refund days arrive' —
    which embeds right at the English evidence. The second retrieval arm for LID-flagged
    queries. Returns '' when nothing changes."""
    stop = _ROMAN_LEX.get(lang, set()) | _ROMAN_STRIP.get(lang, set())
    gloss = _ROMAN_GLOSS.get(lang, {})
    toks = []
    for t in text.split():
        core = t.lower().strip("?.,!")
        if core in stop:
            continue
        toks.append(gloss.get(core, t))
    residual = " ".join(toks).strip()
    return residual if residual and residual.lower() != text.lower() else ""


def detect_language_ex(text: str) -> LangID:
    """(language, script, romanized). Native scripts win outright; Latin text is checked
    against the romanized-vernacular lexicons before defaulting to English."""
    for lang, rx in _SCRIPTS:
        if rx.search(text):
            return LangID(_deva_refine(text) if lang == "hi" else lang, "native", False)
    toks = _TOKEN_RX.findall(text.lower())
    if len(toks) >= 2:
        for lang, lex in _ROMAN_LEX.items():
            hits = {t for t in toks if t in lex}
            if len(hits) >= 2 and len(hits) / len(set(toks)) >= 0.20:
                return LangID(lang, "latin", True)
    return LangID("en", "latin", False)


@runtime_checkable
class Router(Protocol):
    def route(self, text: str) -> RouteResult: ...


class HeuristicRouter:
    """Rule-based router: script detection + intent triage. No filter on cross-lingual retrieval
    by default (a Hindi query SHOULD reach English passages via BGE-M3), but exposes the hook."""

    def __init__(self, language_filter: bool = False, collection: str = None):
        self.language_filter = language_filter
        self.collection = collection

    def route(self, text: str) -> RouteResult:
        t = (text or "").strip()
        lang = detect_language(t)
        if _GREETING.match(t):
            intent = "chitchat"
        elif _META.search(t) and len(t.split()) <= 5:
            # word cap: "who are you" is meta; "help me understand what qdrant is" is a
            # REAL question that an unanchored \bhelp\b would otherwise hijack
            intent = "meta"
        else:
            intent = "qa"
        filters = {"language": lang} if (self.language_filter and intent == "qa") else {}
        return RouteResult(language=lang, intent=intent, filters=filters,
                           collection=self.collection, detail=f"script={lang}")
