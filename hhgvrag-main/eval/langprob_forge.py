#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
langprob_forge.py -- RUN ON FORGE. Direct Sarvam /speech-to-text probe that recovers
`language_probability` (Sarvam's LANGUAGE-ID confidence proxy -- NOT word confidence), which the
harness consumes as asr_confidence but does NOT surface in the /ask response.

It regenerates BYTE-IDENTICAL noisy audio to what /ask received by importing the pure transform
functions + deterministic seed from noise_robustness.py and reading the SAME clean WAVs that were
synthesized here (~/tts_out). It calls saaras:v3 (mode=transcribe, language_code=unknown), exactly
the model the deployed harness uses, timing the Forge->Sarvam round-trip (which mirrors the
server-side stt-stage boundary). Writes langprob.json keyed "id|cond"; scp it back to
eval/fixtures/langprob.json.

Usage:
  ~/anaconda3/envs/hhgvrag/bin/python langprob_forge.py --clean-dir ~/tts_out \
      --out ~/tts_work/langprob.json --min-interval 1.4
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_robustness as nr  # noqa: E402

STT_URL = "https://api.sarvam.ai/speech-to-text"


def read_key(path="~/.config/hhgvrag.env"):
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SARVAM_API_KEY="):
                v = line.split("=", 1)[1].strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                return v
    raise SystemExit("SARVAM_API_KEY not found")


def wav_bytes(x, sr=16000):
    xi = np.clip(np.round(x), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(xi.tobytes())
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="~/tts_out")
    ap.add_argument("--out", default="~/tts_work/langprob.json")
    ap.add_argument("--env", default="~/.config/hhgvrag.env")
    ap.add_argument("--min-interval", type=float, default=1.4)
    args = ap.parse_args()

    import requests
    key = read_key(args.env)
    clean_dir = os.path.expanduser(args.clean_dir)
    out_path = os.path.expanduser(args.out)

    us = nr.utterances()
    # Build the babble pool per language EXACTLY as noise_robustness.cmd_variants does, from the
    # same clean WAVs (headroom-normalized), so regenerated audio is byte-identical.
    pool = {}
    for u in us:
        x, _ = nr.read_wav(os.path.join(clean_dir, u["id"] + ".wav"))
        pool.setdefault(u["lang"], []).append((u["id"], nr.headroom_normalize(x)))

    out = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            out = json.load(f)

    todo = []
    for u in us:
        for cond in u["langprob_conds"]:
            key_ = u["id"] + "|" + cond
            if key_ not in out or out[key_].get("language_probability") is None:
                todo.append((u, cond, key_))
    print("langprob: %d cells to probe (%d already cached)"
          % (len(todo), sum(len(u["langprob_conds"]) for u in us) - len(todo)), flush=True)

    session = requests.Session()
    n = 0
    for u, cond, key_ in todo:
        x, _ = nr.read_wav(os.path.join(clean_dir, u["id"] + ".wav"))
        x = nr.headroom_normalize(x)
        srcs = [xx for (i, xx) in pool[u["lang"]] if i != u["id"]] or \
               [xx for (i, xx) in pool[u["lang"]]]
        seed = (nr.SEED + nr.stable_seed(u["id"] + "|" + cond)) % (2 ** 32)
        y, _note = nr.make_variant(x, cond, seed, srcs)
        audio = wav_bytes(y, nr.SR)
        dur_ms = 1000.0 * len(y) / nr.SR
        lp, txt, code, ms, err = None, None, None, None, None
        for attempt in range(4):
            try:
                t0 = time.time()
                r = session.post(
                    STT_URL, headers={"api-subscription-key": key},
                    files={"file": ("a.wav", audio, "audio/wav")},
                    data={"model": "saaras:v3", "mode": "transcribe",
                          "language_code": "unknown"},
                    timeout=30)
                ms = round((time.time() - t0) * 1000, 1)
                if r.status_code == 200:
                    j = r.json()
                    lp = j.get("language_probability")
                    txt = j.get("transcript", "")
                    code = j.get("language_code")
                    break
                elif r.status_code == 429:
                    err = "429"
                    time.sleep(5 * (attempt + 1))
                else:
                    err = "HTTP %d: %s" % (r.status_code, (r.text or "")[:100])
                    time.sleep(2 * (attempt + 1))
            except Exception as e:  # noqa
                err = str(e)[:100]
                time.sleep(2 * (attempt + 1))
        out[key_] = {"id": u["id"], "cond": cond, "lang": u["lang"],
                     "language_probability": lp, "transcript_direct": txt,
                     "detected_code": code, "direct_ms": ms, "dur_ms": round(dur_ms, 1),
                     "err": err}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        n += 1
        print("  [%3d/%3d] %-18s %-9s lp=%s code=%s ms=%s %s"
              % (n, len(todo), u["id"], cond, lp, code, ms, err or ""), flush=True)
        time.sleep(args.min_interval)

    done = sum(1 for v in out.values() if v.get("language_probability") is not None)
    print("langprob done: %d cells with language_probability -> %s" % (done, out_path), flush=True)


if __name__ == "__main__":
    main()
