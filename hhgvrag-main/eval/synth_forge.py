#!/usr/bin/env python3
"""
synth_forge.py -- Sarvam TTS synthesis, RUN ON FORGE ONLY (the box that holds the key).

Reads SARVAM_API_KEY from ~/.config/hhgvrag.env (value never printed), takes a JSON manifest of
utterances on --manifest, writes one 16 kHz mono 16-bit WAV per item into --out, and prints a
per-item status line (id, ok, bytes, sample_rate) -- never the key, never the audio.

The reference TEXT that TTS speaks is the ground truth the eval scores transcripts against, so the
manifest is the single source of truth for both synthesis (here) and scoring (noise_robustness.py).

Usage (on Forge):
  ~/anaconda3/envs/hhgvrag/bin/python synth_forge.py --manifest m.json --out ~/tts_out
Throttle: --min-interval seconds between calls (default 1.4 -> ~42/min, under the 50/min tier cap).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import wave

TTS_URL = "https://api.sarvam.ai/text-to-speech"


def _read_key(path: str) -> str:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SARVAM_API_KEY="):
                v = line.split("=", 1)[1].strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                return v
    raise SystemExit("SARVAM_API_KEY not found in " + path)


def _to_16k_mono_pcm16(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Return (canonical_wav_bytes, sample_rate, n_frames). Sarvam already returns 16k mono PCM16
    when speech_sample_rate=16000; we re-wrap defensively and report the true header values."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        nch, sw, sr, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        frames = w.readframes(nfr)
    # We requested 16k mono PCM16; assert and pass through. If channels>1, keep as-is (report it).
    return wav_bytes, sr, nfr


def synth_one(session, key, item, param_name):
    import requests  # noqa
    body = {
        "text": item["text"],
        param_name: item["language_code"],
        "speaker": item.get("speaker", "anushka"),
        "model": item.get("model", "bulbul:v2"),
        "speech_sample_rate": 16000,
    }
    r = session.post(TTS_URL, headers={"api-subscription-key": key}, json=body, timeout=30)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="~/.config/hhgvrag.env")
    ap.add_argument("--min-interval", type=float, default=1.4)
    args = ap.parse_args()

    import requests
    key = _read_key(args.env)
    os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    with open(args.manifest, "r", encoding="utf-8") as f:
        items = json.load(f)

    session = requests.Session()
    # Auto-detect the correct TTS language param name once, using the first item.
    param_name = "target_language_code"
    if items:
        probe = synth_one(session, key, items[0], param_name)
        if probe.status_code >= 400:
            txt = (probe.text or "")[:200].lower()
            if "language" in txt or "field" in txt or "unexpected" in txt or probe.status_code == 422:
                param_name = "language_code"
        print("PARAM_NAME " + param_name, flush=True)

    results = []
    for i, item in enumerate(items):
        out_path = os.path.join(os.path.expanduser(args.out), item["id"] + ".wav")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 44:
            with wave.open(out_path, "rb") as w:
                sr, nfr = w.getframerate(), w.getnframes()
            print(json.dumps({"id": item["id"], "ok": True, "cached": True,
                              "bytes": os.path.getsize(out_path), "sr": sr, "frames": nfr}), flush=True)
            results.append({"id": item["id"], "ok": True})
            continue
        ok, err, sr, nfr, nbytes = False, None, 0, 0, 0
        for attempt in range(4):
            try:
                r = synth_one(session, key, item, param_name)
                if r.status_code == 200:
                    j = r.json()
                    audios = j.get("audios") or []
                    if not audios:
                        err = "no audios in response"
                        break
                    raw = base64.b64decode(audios[0])
                    canon, sr, nfr = _to_16k_mono_pcm16(raw)
                    with open(out_path, "wb") as fh:
                        fh.write(canon)
                    nbytes = len(canon)
                    ok = True
                    break
                elif r.status_code == 429:
                    err = "429 throttled"
                    time.sleep(5 * (attempt + 1))
                else:
                    err = "HTTP %d: %s" % (r.status_code, (r.text or "")[:120])
                    time.sleep(2 * (attempt + 1))
            except Exception as e:  # noqa
                err = str(e)[:120]
                time.sleep(2 * (attempt + 1))
        print(json.dumps({"id": item["id"], "ok": ok, "err": err, "bytes": nbytes,
                          "sr": sr, "frames": nfr, "lang": item["language_code"]}), flush=True)
        results.append({"id": item["id"], "ok": ok, "err": err})
        if i < len(items) - 1:
            time.sleep(args.min_interval)

    nok = sum(1 for r in results if r["ok"])
    print("SUMMARY %d/%d ok" % (nok, len(results)), flush=True)


if __name__ == "__main__":
    main()
