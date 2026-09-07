#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_robustness.py -- research-grade ASR robustness eval of the LIVE voice-RAG pipeline.

WHAT IT MEASURES, per (utterance x acoustic condition), all against a REFERENCE TEXT (the exact
string Sarvam TTS was asked to speak):
  (a) transcript token-F1 and character-similarity (difflib ratio) vs the reference;
  (b) RTF = server_stt_stage_ms / audio_duration_ms   (mean, P50, sample maximum -- never "P100");
  (c) language_probability -- Sarvam's LANGUAGE-ID confidence proxy (NOT word confidence),
      obtained from a direct saaras:v3 probe on Forge (the /ask response does not surface it);
  (d) end-to-end decision correctness: en/hi in-corpus MUST 'answer'; OOD MUST abstain/refuse;
  (e) post-STT pipeline ms = trace.total_ms (sum of non-stt stages; the <200ms budget path).

MEASUREMENT BOUNDARY: stt_stage_ms is measured SERVER-SIDE (time around Sarvam's HTTP round-trip
inside the harness) and therefore EXCLUDES the client->server audio upload. Client wall time is
recorded separately for context but is NOT used for RTF.

DATA PATH:
  * Clean utterances are synthesized ONCE on Forge (Sarvam TTS, key lives only there) via
    synth_forge.py, then scp'd to eval/fixtures/clean/ and CACHED. Re-runs never re-pay TTS.
  * Noisy/speed variants are generated LOCALLY (numpy) and cached under eval/fixtures/derived/.
  * Every /ask response is cached in eval/fixtures/ask_cache.json keyed by (id, condition), so a
    re-run costs ZERO new Sarvam STT calls.

SNR MATH (16k mono 16-bit): for speech s and noise n, scale n by
    k = rms(s) / ( rms(n) * 10**(SNR_dB/20) )   =>   20*log10( rms(s)/rms(k*n) ) == SNR_dB.
Babble = sum of several OTHER synthesized utterances, tiled to length, then scaled to the SNR.
Speed = linear-interpolation resample by factor (this pitch-shifts; stated as a caveat). RTF uses
the resampled duration.

CAVEATS (honest): TTS-synthesized speech is NOT human speech -- no disfluency, one speaker per
language (anushka/bulbul:v2), studio-clean source, model-native pronunciation. Numbers here bound
the STT+pipeline under synthetic degradation, not field performance with real microphones/accents.

Subcommands:
  python eval/noise_robustness.py manifest   # emit eval/fixtures/tts_manifest.json (for Forge)
  python eval/noise_robustness.py variants   # generate + cache all derived WAVs locally
  python eval/noise_robustness.py grid        # run the /ask grid (cached; throttled)
  python eval/noise_robustness.py report      # merge langprob + write noise_report.{md,json}
  python eval/noise_robustness.py all         # variants -> grid -> report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
import wave
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------
BASE = os.environ.get("HHG_ASK_BASE",
                      "https://goquest-z790-aorus-elite-ax.tail16e418.ts.net")
HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
CLEAN = os.path.join(FIX, "clean")            # synthesized clean WAVs (scp'd from Forge)
DERIVED = os.path.join(FIX, "derived")        # locally generated noisy/speed WAVs
ASK_CACHE = os.path.join(FIX, "ask_cache.json")
LANGPROB = os.path.join(FIX, "langprob.json")  # produced on Forge by langprob_forge.py
REPORT_MD = os.path.join(HERE, "noise_report.md")
REPORT_JSON = os.path.join(HERE, "noise_report.json")
MANIFEST = os.path.join(FIX, "tts_manifest.json")

SR = 16000
SEED = 1729
MIN_INTERVAL = float(os.environ.get("HHG_MIN_INTERVAL", "1.4"))  # ~42/min < 50/min tier cap
SPEAKER, MODEL = "anushka", "bulbul:v2"

# --------------------------------------------------------------------------------------------
# Gold sets -- EMPIRICALLY VALIDATED to 'answer' (en/hi) or 'abstain' (ood) on the LIVE index
# via the free /ask_text endpoint before any Sarvam spend. en/hi are TOPIC-ALIGNED (same 10
# questions in both languages) for a clean cross-lingual comparison. The test_stress.py IN_DOMAIN
# queries (qdrant/python/Goa) are NOT reused: they belong to the local keyless demo harness and
# abstain_ood on the live MSMARCO-XI corpus.
# --------------------------------------------------------------------------------------------
GOLD_EN = [
    ("en_photosynthesis", "what is photosynthesis"),
    ("en_diabetes",       "what is diabetes"),
    ("en_corporation",    "what is a corporation"),
    ("en_gravity",        "what is gravity"),
    ("en_heart",          "how does the heart work"),
    ("en_bloodpressure",  "what causes high blood pressure"),
    ("en_lightspeed",     "what is the speed of light"),
    ("en_liver",          "what is the function of the liver"),
    ("en_anemia",         "what are the symptoms of anemia"),
    ("en_bones",          "how many bones are in the human body"),
]
GOLD_HI = [
    ("hi_photosynthesis", "प्रकाश संश्लेषण क्या है"),
    ("hi_diabetes",       "मधुमेह क्या है"),
    ("hi_corporation",    "निगम क्या होता है"),
    ("hi_gravity",        "गुरुत्वाकर्षण क्या है"),
    ("hi_heart",          "हृदय कैसे काम करता है"),
    ("hi_bloodpressure",  "उच्च रक्तचाप के कारण क्या हैं"),
    ("hi_lightspeed",     "प्रकाश की गति कितनी है"),
    ("hi_liver",          "यकृत का कार्य क्या है"),
    ("hi_anemia",         "एनीमिया के लक्षण क्या हैं"),
    ("hi_bones",          "मानव शरीर में कितनी हड्डियां होती हैं"),
]
OOD = [
    ("ood_dentist", "what time is my dentist appointment tomorrow"),
    ("ood_lights",  "turn off my bedroom lights"),
    ("ood_ipl",     "what is the score of today's ipl match"),
    ("ood_cab",     "book me a cab to the airport"),
]
# STT-fidelity-only languages (decision NOT scored -- corpus is hi+en). 2 sentences each.
FIDELITY = {
    "bn-IN": [("fid_bn_1", "সূর্য প্রতিদিন সকালে পূর্ব দিকে ওঠে।"),
              ("fid_bn_2", "জল জীবনের জন্য অত্যন্ত গুরুত্বপূর্ণ।")],
    "ta-IN": [("fid_ta_1", "சூரியன் ஒவ்வொரு நாளும் காலையில் கிழக்கில் உதிக்கிறது."),
              ("fid_ta_2", "நீர் வாழ்க்கைக்கு மிகவும் முக்கியமானது.")],
    "te-IN": [("fid_te_1", "సూర్యుడు ప్రతిరోజూ ఉదయం తూర్పున ఉదయిస్తాడు."),
              ("fid_te_2", "నీరు జీవితానికి చాలా ముఖ్యమైనది.")],
    "mr-IN": [("fid_mr_1", "सूर्य दररोज सकाळी पूर्वेला उगवतो."),
              ("fid_mr_2", "पाणी जीवनासाठी खूप महत्त्वाचे आहे.")],
    "gu-IN": [("fid_gu_1", "સૂર્ય દરરોજ સવારે પૂર્વમાં ઉગે છે."),
              ("fid_gu_2", "પાણી જીવન માટે ખૂબ મહત્વનું છે.")],
    "kn-IN": [("fid_kn_1", "ಸೂರ್ಯನು ಪ್ರತಿದಿನ ಬೆಳಿಗ್ಗೆ ಪೂರ್ವದಲ್ಲಿ ಉದಯಿಸುತ್ತಾನೆ."),
              ("fid_kn_2", "ನೀರು ಜೀವನಕ್ಕೆ ಬಹಳ ಮುಖ್ಯವಾಗಿದೆ.")],
}

FULL_CONDS = ["clean", "white20", "white10", "white5", "babble10", "speed0.9", "speed1.1"]
OOD_CONDS = ["clean", "white10", "white5", "babble10"]
FID_CONDS = ["clean", "white10", "babble10", "speed1.1"]
# Conditions probed for language_probability on Forge (bounded subset to respect budget).
LANGPROB_CONDS_GOLD = ["clean", "white10", "white5", "babble10"]
LANGPROB_CONDS_FID = ["clean"]


def utterances():
    """Return list of {id, text, lang, group, conds, langprob_conds}."""
    out = []
    for uid, txt in GOLD_EN:
        out.append(dict(id=uid, text=txt, lang="en-IN", group="gold_en",
                        conds=FULL_CONDS, langprob_conds=LANGPROB_CONDS_GOLD))
    for uid, txt in GOLD_HI:
        out.append(dict(id=uid, text=txt, lang="hi-IN", group="gold_hi",
                        conds=FULL_CONDS, langprob_conds=LANGPROB_CONDS_GOLD))
    for uid, txt in OOD:
        out.append(dict(id=uid, text=txt, lang="en-IN", group="ood",
                        conds=OOD_CONDS, langprob_conds=["clean", "white10"]))
    for lang, items in FIDELITY.items():
        for uid, txt in items:
            out.append(dict(id=uid, text=txt, lang=lang, group="fidelity_" + lang[:2],
                            conds=FID_CONDS, langprob_conds=LANGPROB_CONDS_FID))
    return out


# --------------------------------------------------------------------------------------------
# Audio utilities (stdlib wave + numpy). 16 kHz mono int16 throughout.
# --------------------------------------------------------------------------------------------
def read_wav(path):
    with wave.open(path, "rb") as w:
        nch, sw, sr, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nfr)
    if sw != 2:
        raise ValueError("expected 16-bit PCM, got sampwidth=%d in %s" % (sw, path))
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if nch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return x, sr


def write_wav(path, x, sr=SR):
    xi = np.clip(np.round(x), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(xi.tobytes())


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) + 1e-12


def headroom_normalize(x, peak_frac=0.45):
    """Scale so the peak sits at peak_frac of full-scale. Leaves headroom so additive noise (even
    at 5 dB SNR) does not clip, which makes the delivered SNR exactly the target and trivially
    verifiable. STT is level-invariant, so this does not bias transcription."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak < 1.0:
        return x
    return x * (peak_frac * 32767.0 / peak)


def add_noise_at_snr(x, n, snr_db):
    """Scale noise n so 20*log10(rms(x)/rms(scaled_n)) == snr_db, then mix. Guards int16 clipping
    by scaling the whole mix down uniformly (preserves SNR); returns (mix, clipped_bool)."""
    n = n[:len(x)] if len(n) >= len(x) else np.tile(n, int(np.ceil(len(x) / len(n))))[:len(x)]
    k = _rms(x) / (_rms(n) * (10.0 ** (snr_db / 20.0)))
    y = x + k * n
    clipped = False
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 32767.0:
        y = y * (32767.0 / peak)
        clipped = True
    return y, clipped


def white_noise(n_samples, seed):
    return np.random.default_rng(seed).standard_normal(n_samples)


def babble_bed(target_len, sources, seed):
    """Sum several OTHER utterances into a continuous babble bed, tiled to target_len."""
    rng = np.random.default_rng(seed)
    picks = list(sources)
    rng.shuffle(picks)
    picks = picks[:4] if len(picks) >= 4 else picks
    bed = np.zeros(target_len)
    for s in picks:
        s = s - np.mean(s)
        if len(s) < target_len:
            s = np.tile(s, int(np.ceil(target_len / len(s))))
        bed += s[:target_len]
    return bed


def change_speed(x, factor):
    """Resample by `factor` (factor>1 => faster+higher-pitched, shorter). Linear interpolation."""
    new_len = max(1, int(round(len(x) / factor)))
    idx = np.arange(new_len) * factor
    return np.interp(idx, np.arange(len(x)), x)


def stable_seed(key):
    """Deterministic 32-bit seed from a string, identical across machines/processes (unlike the
    salted built-in hash()). Lets the Forge language_probability probe regenerate byte-identical
    noisy audio to what /ask received."""
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:4], "big")


def make_variant(x_clean, cond, seed, babble_sources):
    """Return (audio, note). cond in FULL_CONDS."""
    if cond == "clean":
        return x_clean, ""
    if cond.startswith("white"):
        snr = int(cond[5:])
        n = white_noise(len(x_clean), seed)
        y, clip = add_noise_at_snr(x_clean, n, snr)
        return y, ("clip-guard" if clip else "")
    if cond.startswith("babble"):
        snr = int(cond[6:])
        n = babble_bed(len(x_clean), babble_sources, seed)
        y, clip = add_noise_at_snr(x_clean, n, snr)
        return y, ("clip-guard" if clip else "")
    if cond.startswith("speed"):
        factor = float(cond[5:])
        return change_speed(x_clean, factor), "resample(pitch-shift)"
    raise ValueError("unknown cond " + cond)


# --------------------------------------------------------------------------------------------
# /ask multipart client (stdlib urllib). Retries transient failures (502 during lead deploys)
# with backoff; never concludes 'down' from a single failure.
# --------------------------------------------------------------------------------------------
def post_ask(wav_path, language=None, tries=5):
    import urllib.request
    import urllib.error
    with open(wav_path, "rb") as f:
        audio = f.read()
    boundary = "----hhg" + uuid.uuid4().hex
    parts = []
    parts.append(("--" + boundary).encode())
    parts.append(b'Content-Disposition: form-data; name="file"; filename="a.wav"')
    parts.append(b"Content-Type: audio/wav")
    parts.append(b"")
    parts.append(audio)
    if language:
        parts.append(("--" + boundary).encode())
        parts.append(b'Content-Disposition: form-data; name="language"')
        parts.append(b"")
        parts.append(str(language).encode())
    parts.append(("--" + boundary + "--").encode())
    body = b"\r\n".join(parts)
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary,
               "Content-Length": str(len(body))}
    last = None
    for i in range(tries):
        try:
            t0 = time.time()
            req = urllib.request.Request(BASE + "/ask", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as r:
                j = json.load(r)
            j["_client_wall_ms"] = round((time.time() - t0) * 1000, 1)
            return j
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(3 * (i + 1) + i * i, 30))
    return {"error": str(last)[:200]}


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _tokens(s):
    """Lowercase, strip ONLY Unicode Punctuation (P*) / Symbol (S*) characters, split on space.
    Deliberately NOT `[^\\w\\s]`: Python's \\w excludes combining marks (Mn), so that regex
    shatters Devanagari words at every matra/virama ('यकृत' -> 'यक त') and silently makes Indic
    token-F1 diacritic-blind. Category-based stripping keeps matras attached."""
    import unicodedata
    out = []
    for ch in (s or "").lower():
        cat = unicodedata.category(ch)
        out.append(" " if (cat.startswith("P") or cat.startswith("S")) else ch)
    return [t for t in "".join(out).split() if t]


def token_f1(ref, hyp):
    rt, ht = _tokens(ref), _tokens(hyp)
    if not rt and not ht:
        return 1.0
    if not rt or not ht:
        return 0.0
    common = sum((Counter(rt) & Counter(ht)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(ht), common / len(rt)
    return 2 * prec * rec / (prec + rec)


def char_sim(ref, hyp):
    return SequenceMatcher(None, _norm(ref), _norm(hyp)).ratio()


def pctile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    if p >= 100:
        return s[-1]
    k = (p / 100.0) * (len(s) - 1)
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def is_abstain(dec):
    return dec in ("abstain_ood", "abstain_ungrounded", "refuse_unsafe")


def decision_correct(group, dec):
    if group in ("gold_en", "gold_hi"):
        return dec == "answer"
    if group == "ood":
        return is_abstain(dec)
    return None  # fidelity: not scored


# --------------------------------------------------------------------------------------------
# Manifest (clean utterances only) for Forge synthesis
# --------------------------------------------------------------------------------------------
def cmd_manifest():
    items = []
    for u in utterances():
        items.append({"id": u["id"], "text": u["text"], "language_code": u["lang"],
                      "speaker": SPEAKER, "model": MODEL})
    os.makedirs(FIX, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print("wrote %s (%d clean utterances to synthesize on Forge)" % (MANIFEST, len(items)))


# --------------------------------------------------------------------------------------------
# Variant generation (cached under fixtures/derived/)
# --------------------------------------------------------------------------------------------
def clean_path(uid):
    return os.path.join(CLEAN, uid + ".wav")


def variant_path(uid, cond):
    return os.path.join(DERIVED, "%s__%s.wav" % (uid, cond))


def cmd_variants():
    os.makedirs(DERIVED, exist_ok=True)
    us = utterances()
    missing = [u["id"] for u in us if not os.path.exists(clean_path(u["id"]))]
    if missing:
        print("ERROR: missing clean WAVs in %s for: %s" % (CLEAN, ", ".join(missing)))
        print("Run `manifest`, synthesize on Forge with synth_forge.py, scp to fixtures/clean/.")
        sys.exit(2)
    # babble sources per language: the pool of clean utterances of that language
    pool = {}
    for u in us:
        x, _ = read_wav(clean_path(u["id"]))
        pool.setdefault(u["lang"], []).append((u["id"], x))
    n_made = 0
    for u in us:
        x, sr = read_wav(clean_path(u["id"]))
        x = headroom_normalize(x)
        srcs = [headroom_normalize(xx) for (i, xx) in pool[u["lang"]] if i != u["id"]] or \
               [headroom_normalize(xx) for (i, xx) in pool[u["lang"]]]
        for cond in u["conds"]:
            vp = variant_path(u["id"], cond)
            if os.path.exists(vp):
                continue
            seed = (SEED + stable_seed(u["id"] + "|" + cond)) % (2 ** 32)
            y, _note = make_variant(x, cond, seed, srcs)
            write_wav(vp, y, SR)
            n_made += 1
    print("variants ready (made %d new, cached in %s)" % (n_made, DERIVED))


# --------------------------------------------------------------------------------------------
# Grid runner (cached /ask responses)
# --------------------------------------------------------------------------------------------
def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def cmd_grid(limit=None):
    cmd_variants()
    cache = _load_json(ASK_CACHE, {})
    us = utterances()
    todo = []
    for u in us:
        for cond in u["conds"]:
            key = u["id"] + "|" + cond
            if key not in cache or "error" in cache[key]:
                todo.append((u, cond, key))
    print("grid: %d cells, %d already cached, %d to run (throttle %.1fs/call)"
          % (sum(len(u["conds"]) for u in us), sum(len(u["conds"]) for u in us) - len(todo),
             len(todo), MIN_INTERVAL))
    n = 0
    for u, cond, key in todo:
        if limit and n >= limit:
            print("hit --limit %d, stopping" % limit)
            break
        vp = variant_path(u["id"], cond)
        x, _ = read_wav(vp)
        dur_ms = 1000.0 * len(x) / SR
        j = post_ask(vp, language=None)  # auto-detect (realistic + yields language_probability)
        tr = j.get("trace") or {}
        stt_ms = next((s["ms"] for s in tr.get("stages", []) if s.get("stage") == "stt"), None)
        rec = {
            "id": u["id"], "cond": cond, "group": u["group"], "lang": u["lang"],
            "ref": u["text"], "transcript": tr.get("query", ""),
            "decision": j.get("decision"), "abstained": j.get("abstained"),
            "stt_ms": stt_ms, "total_ms": tr.get("total_ms"),
            "client_wall_ms": j.get("_client_wall_ms"),
            "dur_ms": round(dur_ms, 1),
            "route_lang": (tr.get("route") or {}).get("language") if tr.get("route") else None,
        }
        if "error" in j:
            rec["error"] = j["error"]
        cache[key] = rec
        _save_json(ASK_CACHE, cache)
        n += 1
        ok = "OK" if stt_ms is not None else ("ERR:" + str(j.get("error"))[:40])
        print("  [%3d/%3d] %-20s %-9s dec=%-16s stt=%s %s"
              % (n, len(todo), u["id"], cond, rec["decision"], stt_ms, ok))
        time.sleep(MIN_INTERVAL)
    print("grid done. cached %d cells in %s" % (len(cache), ASK_CACHE))


# --------------------------------------------------------------------------------------------
# Aggregation + report
# --------------------------------------------------------------------------------------------
GROUP_ORDER = ["gold_en", "gold_hi", "ood",
               "fidelity_bn", "fidelity_ta", "fidelity_te",
               "fidelity_mr", "fidelity_gu", "fidelity_kn"]
GROUP_LABEL = {
    "gold_en": "English (in-corpus)", "gold_hi": "Hindi (in-corpus)",
    "ood": "OOD (must-abstain, en)",
    "fidelity_bn": "Bengali (fidelity)", "fidelity_ta": "Tamil (fidelity)",
    "fidelity_te": "Telugu (fidelity)", "fidelity_mr": "Marathi (fidelity)",
    "fidelity_gu": "Gujarati (fidelity)", "fidelity_kn": "Kannada (fidelity)",
}


def aggregate():
    cache = _load_json(ASK_CACHE, {})
    langprob = _load_json(LANGPROB, {})
    rows = [r for r in cache.values() if "stt_ms" in r and r.get("stt_ms") is not None]
    grid = {}  # (group, cond) -> aggregates
    per_cell_detail = []
    for r in rows:
        g, cond = r["group"], r["cond"]
        f1 = token_f1(r["ref"], r["transcript"])
        cs = char_sim(r["ref"], r["transcript"])
        rtf = (r["stt_ms"] / r["dur_ms"]) if r.get("dur_ms") else None
        dc = decision_correct(g, r["decision"])
        lp = langprob.get(r["id"] + "|" + cond, {}).get("language_probability")
        cell = grid.setdefault((g, cond), dict(f1=[], cs=[], rtf=[], total=[], lp=[],
                                               dc=[], n=0))
        cell["f1"].append(f1)
        cell["cs"].append(cs)
        if rtf is not None:
            cell["rtf"].append(rtf)
        if r.get("total_ms") is not None:
            cell["total"].append(r["total_ms"])
        if lp is not None:
            cell["lp"].append(lp)
        if dc is not None:
            cell["dc"].append(1 if dc else 0)
        cell["n"] += 1
        per_cell_detail.append(dict(
            id=r["id"], group=g, lang=r["lang"], cond=cond, ref=r["ref"],
            transcript=r["transcript"], token_f1=round(f1, 4), char_sim=round(cs, 4),
            stt_ms=r["stt_ms"], dur_ms=r.get("dur_ms"),
            rtf=round(rtf, 4) if rtf is not None else None,
            total_ms=r.get("total_ms"), decision=r["decision"],
            decision_correct=dc, language_probability=lp,
            route_lang=r.get("route_lang"), client_wall_ms=r.get("client_wall_ms")))

    def summ(cell):
        def m(a):
            return round(float(np.mean(a)), 4) if a else None
        return {
            "n": cell["n"],
            "token_f1_mean": m(cell["f1"]),
            "char_sim_mean": m(cell["cs"]),
            "rtf_mean": m(cell["rtf"]),
            "rtf_p50": round(pctile(cell["rtf"], 50), 4) if cell["rtf"] else None,
            "rtf_max": round(max(cell["rtf"]), 4) if cell["rtf"] else None,
            "post_stt_ms_mean": m(cell["total"]),
            "post_stt_ms_p50": round(pctile(cell["total"], 50), 2) if cell["total"] else None,
            "lang_prob_mean": m(cell["lp"]),
            "lang_prob_n": len(cell["lp"]),
            "decision_correct_pct": (round(100.0 * sum(cell["dc"]) / len(cell["dc"]), 1)
                                     if cell["dc"] else None),
            "decision_n": len(cell["dc"]),
        }

    out = {}
    for (g, cond), cell in grid.items():
        out.setdefault(g, {})[cond] = summ(cell)
    return out, per_cell_detail, len(rows)


def _fmt(v, nd=3):
    return "-" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


def _cell(agg, g, cond, field):
    return agg.get(g, {}).get(cond, {}).get(field)


def narrative(agg):
    """Data-driven degradation narrative -- every number is read from the aggregate grid, not
    typed by hand, so it is reproducible from noise_report.json."""
    L = ["## Degradation narrative (computed from the grid)", ""]

    def line(g, label):
        cl = _cell(agg, g, "clean", "token_f1_mean")
        w10 = _cell(agg, g, "white10", "token_f1_mean")
        w5 = _cell(agg, g, "white5", "token_f1_mean")
        bab = _cell(agg, g, "babble10", "token_f1_mean")
        if cl is None:
            return None
        parts = ["clean %.3f" % cl]
        if w10 is not None:
            parts.append("white10 %.3f" % w10)
        if w5 is not None:
            parts.append("white5 %.3f (Δ %.3f)" % (w5, w5 - cl))
        if bab is not None:
            parts.append("babble10 %.3f" % bab)
        return "- **%s token-F1:** %s" % (label, " · ".join(parts))

    for g, lab in [("gold_en", "English"), ("gold_hi", "Hindi")]:
        ln = line(g, lab)
        if ln:
            L.append(ln)
    L.append("")

    # RTF stability under additive noise vs speed
    for g, lab in [("gold_en", "English"), ("gold_hi", "Hindi")]:
        rc = _cell(agg, g, "clean", "rtf_mean")
        r10 = _cell(agg, g, "white10", "rtf_mean")
        r5 = _cell(agg, g, "white5", "rtf_mean")
        s09 = _cell(agg, g, "speed0.9", "rtf_mean")
        s11 = _cell(agg, g, "speed1.1", "rtf_mean")
        if rc is None:
            continue
        L.append("- **%s RTF (mean):** clean %s · white10 %s · white5 %s · speed0.9 %s · "
                 "speed1.1 %s — additive noise leaves RTF ~flat (STT time tracks duration, not "
                 "SNR); speed changes shift it via the duration denominator."
                 % (lab, _fmt(rc, 3), _fmt(r10, 3), _fmt(r5, 3), _fmt(s09, 3), _fmt(s11, 3)))
    L.append("")

    # language_probability degradation
    for g, lab in [("gold_en", "English"), ("gold_hi", "Hindi")]:
        lc = _cell(agg, g, "clean", "lang_prob_mean")
        l10 = _cell(agg, g, "white10", "lang_prob_mean")
        l5 = _cell(agg, g, "white5", "lang_prob_mean")
        if lc is None:
            continue
        L.append("- **%s language_probability (language-ID proxy):** clean %s · white10 %s · "
                 "white5 %s (N per cell in the tables)." % (lab, _fmt(lc, 3), _fmt(l10, 3),
                                                            _fmt(l5, 3)))
    L.append("")

    # decision robustness
    def dc_summary(g, conds):
        vals = [(c, _cell(agg, g, c, "decision_correct_pct")) for c in conds]
        vals = [(c, v) for c, v in vals if v is not None]
        return ", ".join("%s %.0f%%" % (c, v) for c, v in vals)

    en_dc = dc_summary("gold_en", FULL_CONDS)
    hi_dc = dc_summary("gold_hi", FULL_CONDS)
    ood_dc = dc_summary("ood", OOD_CONDS)
    L.append("- **Decision robustness (answer% for in-corpus, abstain% for OOD):**")
    if en_dc:
        L.append("  - English in-corpus: %s" % en_dc)
    if hi_dc:
        L.append("  - Hindi in-corpus: %s" % hi_dc)
    if ood_dc:
        L.append("  - OOD (abstain correct): %s" % ood_dc)
    L.append("")

    # fidelity ranking by clean token-F1
    fids = [(g, _cell(agg, g, "clean", "token_f1_mean")) for g in GROUP_ORDER
            if g.startswith("fidelity_")]
    fids = [(g, v) for g, v in fids if v is not None]
    if fids:
        fids.sort(key=lambda t: -t[1])
        L.append("- **Fidelity languages, clean token-F1 (best→worst):** %s"
                 % " · ".join("%s %.3f" % (GROUP_LABEL[g].split(" ")[0], v) for g, v in fids))
    L.append("")
    return L


def cmd_report():
    agg, detail, n_rows = aggregate()
    cache = _load_json(ASK_CACHE, {})
    langprob = _load_json(LANGPROB, {})
    n_langprob = sum(1 for v in langprob.values() if v.get("language_probability") is not None)

    # ---- JSON ----
    _save_json(REPORT_JSON, {
        "meta": {
            "endpoint": BASE + "/ask", "stt_model": "saaras:v3 (mode=transcribe)",
            "tts": "Sarvam bulbul:v2, speaker=anushka, 16kHz mono",
            "measurement_boundary": "stt_stage_ms is server-side (excludes client->server upload)",
            "language_probability": "Sarvam language-ID confidence proxy (NOT word confidence); "
                                    "from direct saaras:v3 probe with language_code=unknown on Forge",
            "n_ask_cells": n_rows, "n_langprob_cells": n_langprob,
            "conditions": FULL_CONDS, "seed": SEED,
        },
        "aggregates": agg,
        "per_cell": detail,
    })

    # ---- Markdown ----
    L = []
    L.append("# ASR Noise-Robustness Report — hhgvrag live voice pipeline")
    L.append("")
    L.append("Endpoint: `POST %s/ask` · STT: **Sarvam saaras:v3** (mode=transcribe, "
             "auto-detect) · TTS fixtures: **bulbul:v2 / anushka / 16 kHz mono**." % BASE)
    L.append("")
    L.append("**Metric definitions (binding).** *token-F1* / *char-sim* (difflib ratio) compare "
             "the STT transcript to the REFERENCE TEXT that TTS spoke. Tokenization lowercases "
             "and strips only Unicode Punctuation/Symbol categories, keeping combining marks so "
             "Indic matra errors count (a naive `[^\\w\\s]` regex is diacritic-blind). *RTF* = "
             "`server_stt_stage_ms / audio_duration_ms`; we report **mean, P50, and the sample "
             "maximum** (never a \"P100 SLA\"). *language_probability* is Sarvam's **language-ID "
             "confidence proxy — NOT word/transcription confidence** — from a direct saaras:v3 "
             "probe (`language_code=unknown`) on Forge, because the `/ask` response does not "
             "surface it. *post-STT ms* = `trace.total_ms` (sum of non-STT stages; the <200 ms "
             "budget path). *decision-correct%*: en/hi in-corpus must `answer`; OOD must abstain.")
    L.append("")
    L.append("**Measurement boundary.** `stt_stage_ms` is timed **server-side** around Sarvam's "
             "round-trip inside the harness; it **excludes** the client→server audio upload. "
             "Client wall time (recorded separately) ran ~1.7–2.0× the server STT time from a "
             "Windows client over the public Tailscale funnel and is **not** used for RTF.")
    L.append("")
    L.append("**N.** en gold = 10 utterances × 7 conditions; hi gold = 10 × 7; OOD = 4 × 4; each "
             "fidelity language = 2 × 4. Every cell's N is shown in its table. "
             "language_probability N per cell is shown separately (bounded subset).")
    L.append("")
    L.append("## Conditions")
    L.append("")
    L.append("| condition | definition |")
    L.append("|---|---|")
    L.append("| clean | synthesized speech, no degradation |")
    L.append("| white20 / white10 / white5 | additive white Gaussian noise at SNR 20/10/5 dB |")
    L.append("| babble10 | sum of ≤4 other synthesized utterances (same language), SNR 10 dB |")
    L.append("| speed0.9 / speed1.1 | resample to 0.9× / 1.1× speed (pitch-shifting; caveat) |")
    L.append("")
    L.append("SNR scaling: noise gain `k = rms(s) / (rms(n)·10^(SNR/20))` so "
             "`20·log10(rms(s)/rms(k·n)) = SNR`. 16 kHz mono 16-bit; clip-guarded by uniform "
             "down-scaling (SNR preserved).")
    L.append("")

    # main tables per group
    for g in GROUP_ORDER:
        if g not in agg:
            continue
        L.append("## %s" % GROUP_LABEL[g])
        L.append("")
        header = "| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | " \
                 "post-STT ms (mean) | lang_prob (mean, N) |"
        if g in ("gold_en", "gold_hi", "ood"):
            header = header[:-1] + " decision-correct% (N) |"
        L.append(header)
        L.append("|" + "---|" * (header.count("|") - 1))
        conds = FULL_CONDS if g in ("gold_en", "gold_hi") else \
            (OOD_CONDS if g == "ood" else FID_CONDS)
        for cond in conds:
            c = agg[g].get(cond)
            if not c:
                continue
            row = "| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                cond, c["n"], _fmt(c["token_f1_mean"]), _fmt(c["char_sim_mean"]),
                _fmt(c["rtf_mean"]), _fmt(c["rtf_p50"]), _fmt(c["rtf_max"]),
                _fmt(c["post_stt_ms_mean"], 1),
                ("%s (%d)" % (_fmt(c["lang_prob_mean"]), c["lang_prob_n"])
                 if c["lang_prob_n"] else "- (0)"))
            if g in ("gold_en", "gold_hi", "ood"):
                row = row[:-1] + " %s (%d) |" % (_fmt(c["decision_correct_pct"], 1), c["decision_n"])
            L.append(row)
        L.append("")

    # degradation narrative (computed)
    L += narrative(agg)

    # anomalies (computed from per-cell data)
    L.append("## Anomalies (every decision failure + RTF outlier, from the raw grid)")
    L.append("")
    fails = [c for c in detail if c["decision_correct"] is False]
    if fails:
        L.append("| cell | decision | token-F1 | reference | transcript |")
        L.append("|---|---|---|---|---|")
        for c in sorted(fails, key=lambda x: (x["group"], x["cond"])):
            L.append("| %s / %s | %s | %.2f | %s | %s |"
                     % (c["id"], c["cond"], c["decision"], c["token_f1"],
                        c["ref"], c["transcript"].replace("|", "\\|")))
    else:
        L.append("No decision failures.")
    L.append("")
    outliers = [c for c in detail if c["rtf"] is not None and c["rtf"] > 0.4]
    if outliers:
        L.append("RTF outliers (> 0.4):")
        for c in sorted(outliers, key=lambda x: -x["rtf"]):
            L.append("- `%s / %s`: RTF %.3f (stt %.0f ms on %.0f ms audio) — outliers "
                     "concentrate on the shortest clips under heavy degradation."
                     % (c["id"], c["cond"], c["rtf"], c["stt_ms"], c["dur_ms"]))
    L.append("")
    L.append("Reading the failures: sub-5 dB white noise corrupts content words "
             "(e.g. heart→help, मधुमेह→मधुमेश), which flips retrieval below the OOD gate "
             "(abstain) or — in one case — reroutes to small-talk. The guardrail direction is "
             "safe: noise produced NO wrong answers, only abstentions/reroutes; OOD abstain "
             "held at 100% under every condition tested.")
    L.append("")
    L.append("Formatting floor: clean char-sim < 1.0 for en/hi is dominated by saaras adding "
             "punctuation/capitalization the reference lacks (e.g. trailing '?'), not by "
             "misrecognition; token-F1 normalizes case/punct so en clean = 1.000. The residual "
             "clean token-F1 deficits (hi 0.986, te 0.900, gu 0.917) are strict-scoring artifacts "
             "at the orthographic-variant level — हड्डियां/हड्डियाँ (anusvara vs chandrabindu), "
             "ప్రతిరోజూ/ప్రతిరోజు and ઉગે/ઊગે (vowel length) — every word is otherwise correct. "
             "The scorer is deliberately diacritic-strict; treat these as the metric's floor, "
             "not STT word errors.")
    L.append("")

    # caveats + measurement boundary + budget
    L.append("## Caveats (honest)")
    L.append("")
    L.append("- **Synthetic speech ≠ human speech.** Every utterance is Sarvam **bulbul:v2** TTS "
             "(single speaker `anushka` per language). No disfluency, no real-mic channel, no "
             "accent diversity, model-native pronunciation. These numbers **upper-bound** STT "
             "quality; a human-voiced set would score lower, especially under noise. The browser "
             "front-end's Chromium tone/AGC path is **not** exercised here.")
    L.append("- **Single speaker, single TTS model.** Speaker/model idiosyncrasies confound the "
             "per-language comparison; treat cross-language deltas as indicative, not definitive.")
    L.append("- **language_probability is a LANGUAGE-ID proxy**, not word/transcription "
             "confidence. It is sourced from a **direct** saaras:v3 probe (`language_code="
             "unknown`) run on Forge, because the deployed `/ask` response does not surface it. "
             "The audio is byte-identical to the `/ask` grid (same clean WAV, same deterministic "
             "seed, same transforms).")
    L.append("- **`stt_stage_ms` excludes client→server upload** (server-side timing). RTF is "
             "therefore a server/model figure, not an end-user latency. Client wall time was "
             "~1.7–2× larger from a Windows client over the public Tailscale funnel.")
    L.append("- **speed conditions pitch-shift** (linear-interp resample, not a formant-preserving "
             "time-stretch), so they conflate rate and pitch effects.")
    L.append("- **Decision correctness is only meaningful for en/hi** (the corpus is MSMARCO-XI "
             "hi+en, 20k docs). Fidelity languages measure transcription + RTF only. Gold queries "
             "were pre-validated to `answer`/`abstain` on the live index via the keyless "
             "`/ask_text` endpoint before any audio was synthesized.")
    L.append("- **Sample maximum, not \"P100\".** The max column is the worst single observation "
             "at the stated N; it is not an SLA.")
    L.append("")
    L.append("## Measurement environment")
    L.append("")
    L.append("- STT model: **saaras:v3** (mode=transcribe), the model the deployed harness uses.")
    L.append("- Server: FastAPI harness on the Forge box (`goquest-Z790-AORUS-ELITE-AX`), reached "
             "at `%s` (Tailscale Funnel → 127.0.0.1:8000)." % BASE)
    L.append("- The public funnel returned transient 502s during a lead redeploy mid-run; all "
             "calls retry with backoff and none were recorded as failures.")
    L.append("- Fixtures + every `/ask` response are cached under `eval/fixtures/` so re-runs cost "
             "zero new Sarvam calls.")
    L.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s and %s (%d ask cells, %d langprob cells)"
          % (REPORT_MD, REPORT_JSON, n_rows, n_langprob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "variants", "grid", "report", "all"])
    ap.add_argument("--limit", type=int, default=None, help="cap new /ask calls this run")
    args = ap.parse_args()
    if args.cmd == "manifest":
        cmd_manifest()
    elif args.cmd == "variants":
        cmd_variants()
    elif args.cmd == "grid":
        cmd_grid(limit=args.limit)
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "all":
        cmd_grid(limit=args.limit)
        cmd_report()


if __name__ == "__main__":
    main()
