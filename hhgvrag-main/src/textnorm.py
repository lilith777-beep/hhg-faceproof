"""
textnorm.py — Unicode-correct word tokenizer for every lexical path (2026-08-17).

THE BUG this fixes (independently surfaced by a competitor's write-up and confirmed live in
our own code): Python's `\w` matches base letters (category Lo) but NOT the combining marks
that carry Indic vowels — the Devanagari matras (Mc), the virama (Mn), anusvara/visarga, the
Arabic diacritics of Urdu. So `re.findall(r"\w+", "मधुमेह")` returns three consonant
fragments instead of one word, and every lexical component silently indexes/scores garbage:
  * PageIndex vectorless retrieval (the entire edge tier is lexical) — corrupted on Devanagari
  * the OOD gate's lexical-coverage fallback — Indic queries under-score
  * the extractive composer's same-language sentence scoring — Indic evidence under-ranks
  * grounding lexical support — Indic answers look ungrounded
Dense BGE-M3 retrieval is unaffected (its own SentencePiece tokenizer is correct); this is a
pure lexical-path defect, but lexical is half the pipeline and all of the edge story.

FIX: extend the word class with the Unicode Mark categories (Mn, Mc) across the Brahmic
blocks + Arabic diacritics, built once from unicodedata (self-maintaining — no hand-typed
ranges to rot). Sentence separators (danda ।॥, punctuation) are category Po, never Mn/Mc, so
they still delimit correctly.
"""
from __future__ import annotations

import re
import unicodedata

# combining marks that attach to a base letter to form one written unit
_MARKS = "".join(
    chr(c) for c in range(0x0300, 0x0D80)
    if unicodedata.category(chr(c)) in ("Mn", "Mc"))
_MARKS += "".join(
    chr(c) for c in range(0x0640, 0x06F0)          # Arabic diacritics (Urdu)
    if unicodedata.category(chr(c)) in ("Mn", "Mc"))

# a token = one or more (word char OR combining mark). Base letters come from \w; the marks
# keep मधुमेह / क्या / প্রধানমন্ত্রী whole instead of shattering on every vowel sign.
WORD_RE = re.compile(r"[\w" + re.escape(_MARKS) + r"]+", re.UNICODE)


def tokens(text: str) -> list:
    """Lowercased word tokens, Indic-correct."""
    return WORD_RE.findall(text.lower())


# High-frequency function words per script, for lexical CONTENT filtering (denominator
# hygiene). English got a stopword list from day one; without the Indic equivalents,
# case-markers and copulas counted as "evidence overlap" in the OOD gate and the composer —
# a topically-wrong same-language sentence could outscore correct evidence, and the
# full-content-token-coverage rescue lane was structurally unreachable for Indic askers.
# Shared by guardrails._STOP and generation._STOPWORDS so the gate and the composer agree.
INDIC_STOP = {
    # Hindi / Marathi / Nepali (Devanagari)
    "का", "की", "के", "है", "हैं", "में", "से", "को", "और", "या", "पर", "क्या", "यह", "वह",
    "एक", "हो", "था", "थी", "थे", "कि", "भी", "नहीं", "तो", "ही", "कौन", "अभी", "क्यों",
    "कैसे", "कब", "कहाँ", "आहे", "आहेत", "आणि", "काय", "छन्", "हुन्छ",
    # Bengali / Assamese
    "কি", "কে", "এর", "এবং", "একটি", "হয়", "আছে", "থেকে", "এই", "তার", "কেন", "কীভাবে",
    # Tamil
    "என்ன", "ஒரு", "இது", "அது", "மற்றும்", "ஆகும்", "உள்ளது", "எப்படி", "ஏன்",
    # Telugu
    "ఏమిటి", "ఒక", "ఇది", "అది", "మరియు", "ఉంది", "ఎలా", "ఎందుకు",
    # Gujarati
    "શું", "એક", "છે", "અને", "માં", "કેવી", "કેમ",
    # Kannada
    "ಏನು", "ಒಂದು", "ಇದು", "ಮತ್ತು", "ಇದೆ", "ಹೇಗೆ", "ಏಕೆ",
    # Malayalam
    "എന്ത്", "ഒരു", "ഇത്", "ആണ്", "ഉണ്ട്", "എങ്ങനെ", "എന്തുകൊണ്ട്",
    # Punjabi (Gurmukhi)
    "ਕੀ", "ਇੱਕ", "ਹੈ", "ਅਤੇ", "ਵਿੱਚ", "ਦਾ", "ਦੀ", "ਦੇ", "ਕਿਵੇਂ",
    # Odia
    "କଣ", "ଏକ", "ଅଛି", "ଏବଂ", "କିପରି",
    # Urdu (Arabic script)
    "کیا", "ایک", "ہے", "ہیں", "میں", "سے", "کو", "اور", "کا", "کی", "کے", "یہ", "وہ",
    "کیسے", "کیوں",
}
