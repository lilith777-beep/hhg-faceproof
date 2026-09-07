"""
stt.py — Speech-to-text (Sarvam). Upstream of the 200ms retrieval budget.
`MockSTT` lets the pipeline be tested without audio; `SarvamSTT` is the real call.

Verified against the Sarvam /speech-to-text docs:
- endpoint https://api.sarvam.ai/speech-to-text · header 'api-subscription-key'
- accepts WAV/MP3/OGG/OPUS/WebM/FLAC/M4A/... (so browser MediaRecorder WebM/Opus is fine —
  we just have to label the bytes correctly instead of always claiming 'audio/wav')
- language_code 'unknown' = auto-detect · response: transcript, language_code, language_probability
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Transcript:
    text: str
    language: str = "unknown"
    ms: float = 0.0
    confidence: float = 1.0        # ASR confidence (Sarvam language_probability) -> noise layer


@runtime_checkable
class STT(Protocol):
    def transcribe(self, audio: bytes, language: Optional[str] = None) -> Transcript: ...


class MockSTT:
    def __init__(self, text: str = "", language: str = "en", confidence: float = 1.0):
        self.text, self.language, self.confidence = text, language, confidence

    def transcribe(self, audio: bytes, language: Optional[str] = None) -> Transcript:
        return Transcript(text=self.text, language=language or self.language,
                          confidence=self.confidence)


def _sniff_audio(audio: bytes) -> tuple:
    """(filename, mime) from magic bytes — MediaRecorder emits WebM/Opus, not WAV."""
    b = audio[:16]
    if b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        return "audio.wav", "audio/wav"
    if b[:4] == b"\x1aE\xdf\xa3":                 # EBML -> WebM/Matroska
        return "audio.webm", "audio/webm"
    if b[:4] == b"OggS":
        return "audio.ogg", "audio/ogg"
    if b[:3] == b"ID3" or b[:2] == b"\xff\xfb":
        return "audio.mp3", "audio/mpeg"
    if b[4:8] == b"ftyp":
        return "audio.m4a", "audio/mp4"
    if b[:4] == b"fLaC":
        return "audio.flac", "audio/flac"
    return "audio.webm", "audio/webm"             # browser default; Sarvam accepts it


# Indic + Arabic(Urdu) script blocks — presence of any means the transcript is already in a
# native script (no transliteration needed).
_NATIVE_RANGES = ((0x900, 0x97f), (0x980, 0x9ff), (0xa00, 0xa7f), (0xa80, 0xaff),
                  (0xb00, 0xb7f), (0xb80, 0xbff), (0xc00, 0xc7f), (0xc80, 0xcff),
                  (0xd00, 0xd7f), (0x600, 0x6ff))


def _has_native_script(text: str) -> bool:
    return any(any(a <= ord(c) <= b for a, b in _NATIVE_RANGES) for c in text)


class SarvamSTT:
    """Sarvam /speech-to-text. Needs SARVAM_API_KEY. Auto-detects Indic + en when language is None.

    Romanized-Indic rescue: on real/code-mixed speech Sarvam sometimes returns an Indic
    utterance in Latin script ("madhumeh ke lakshan"). The corpus is native-script, so a
    romanized query can't match and retrieval fails. When the detected language is Indic but
    the transcript carries no native-script characters, we transliterate it back to the
    native script via Sarvam's /transliterate endpoint before it enters the pipeline."""
    URL = "https://api.sarvam.ai/speech-to-text"
    XLIT_URL = "https://api.sarvam.ai/transliterate"

    def __init__(self, api_key: str, model: str = "saaras:v3", mode: str = "transcribe",
                 transliterate: bool = True):
        import httpx
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.transliterate = transliterate
        self._httpx = httpx

    def _to_native(self, text: str, lang_code: str) -> str:
        """Transliterate a romanized Indic transcript to its native script (best-effort)."""
        try:
            r = self._httpx.post(
                self.XLIT_URL,
                headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"},
                json={"input": text, "source_language_code": "en-IN",
                      "target_language_code": lang_code, "spoken_form": True},
                timeout=5.0)
            r.raise_for_status()
            return r.json().get("transliterated_text") or text
        except Exception:  # noqa: BLE001 — transliteration is a rescue; keep the roman text on failure
            return text

    def transcribe(self, audio: bytes, language: Optional[str] = None) -> Transcript:
        filename, mime = _sniff_audio(audio)
        files = {"file": (filename, audio, mime)}
        data = {"model": self.model, "language_code": language or "unknown"}
        if self.model.startswith("saaras"):          # 'mode' is a saaras-only field
            data["mode"] = self.mode
        # 7s: this timeout is THE guard on the voice path (STT runs on the caller's thread,
        # not the stage pool) — it must bound how long a Sarvam brownout can pin a worker.
        # Healthy saaras:v3 calls return in ~1-3s; 7s is generous headroom, 15s was a liability.
        r = self._httpx.post(self.URL, headers={"api-subscription-key": self.api_key},
                             files=files, data=data, timeout=7.0)
        r.raise_for_status()
        j = r.json()
        text = j.get("transcript", "")
        lang = j.get("language_code", language or "unknown")
        # romanized-Indic → native script (only fires on the failure case: Latin text +
        # Indic language detected; adds one ~1s call, and only when retrieval would fail anyway)
        conf = float(j.get("language_probability") or 1.0)
        # confidence floor: if Sarvam confidently mislabels an English utterance as Indic and
        # returns Latin text, transliterating it would corrupt it into Devanagari garbage — only
        # transliterate when the language call is reasonably confident (red-team bonus finding)
        if (self.transliterate and text and lang and lang not in ("en-IN", "unknown")
                and conf >= 0.6 and not _has_native_script(text)
                and any(("a" <= c.lower() <= "z") for c in text)):
            text = self._to_native(text, lang)
        return Transcript(text=text, language=lang, confidence=conf)


# harness lid.lang (bare ISO) -> Sarvam target_language_code. Odia is 'od-IN' in Sarvam's
# scheme (not 'or-IN'); the rest follow <iso>-IN.
_SARVAM_LANG = {"hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
                "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN", "or": "od-IN",
                "ur": "ur-IN"}


class SarvamTransliterator:
    """Roman -> native-script transliteration via Sarvam /transliterate, with a thread-safe
    LRU cache. Typed romanized Indic ('madhumeh kya hai') becomes native ('मधुमेह क्या है')
    so it takes the FAST native retrieval path instead of the slow, weaker romanized dual-arm.
    p50 ~370ms uncached, instant cached. Returns the input UNCHANGED on any failure or if the
    output isn't native script — so the caller can safely fall back to the dual-arm.

    The English-false-positive danger (transliterating English -> Devanagari garbage) is NOT
    guarded here — the OUTPUT of a wrong-language call IS valid native script and would pass
    the has-native check. It MUST be gated at the call site by a reliable romanized-Indic
    language detector (harness: lid.romanized). This class only converts what it's given."""
    URL = "https://api.sarvam.ai/transliterate"

    def __init__(self, api_key: str, timeout: float = 4.0, cache_size: int = 4096):
        import threading
        import httpx
        self.api_key = api_key
        self.timeout = timeout
        self._httpx = httpx
        self._cap = cache_size
        self._cache: dict = {}
        self._order: list = []
        self._lock = threading.Lock()

    def transliterate(self, text: str, lang: str) -> str:
        tgt = _SARVAM_LANG.get((lang or "").split("-")[0])
        if not tgt or not text or not text.strip():
            return text
        key = (lang, text)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        try:
            r = self._httpx.post(
                self.URL,
                headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"},
                json={"input": text, "source_language_code": "en-IN",
                      "target_language_code": tgt, "spoken_form": True},
                timeout=self.timeout)
            r.raise_for_status()
            out = r.json().get("transliterated_text") or text
        except Exception:  # noqa: BLE001 — rescue path; caller falls back to the dual-arm
            return text
        if not _has_native_script(out):   # no-op / still-Latin -> don't accept
            out = text
        with self._lock:
            if key not in self._cache:
                self._cache[key] = out
                self._order.append(key)
                if len(self._order) > self._cap:
                    self._cache.pop(self._order.pop(0), None)
        return out
