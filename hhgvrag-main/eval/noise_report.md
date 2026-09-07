# ASR Noise-Robustness Report — hhgvrag live voice pipeline

Endpoint: `POST https://goquest-z790-aorus-elite-ax.tail16e418.ts.net/ask` · STT: **Sarvam saaras:v3** (mode=transcribe, auto-detect) · TTS fixtures: **bulbul:v2 / anushka / 16 kHz mono**.

**Metric definitions (binding).** *token-F1* / *char-sim* (difflib ratio) compare the STT transcript to the REFERENCE TEXT that TTS spoke. Tokenization lowercases and strips only Unicode Punctuation/Symbol categories, keeping combining marks so Indic matra errors count (a naive `[^\w\s]` regex is diacritic-blind). *RTF* = `server_stt_stage_ms / audio_duration_ms`; we report **mean, P50, and the sample maximum** (never a "P100 SLA"). *language_probability* is Sarvam's **language-ID confidence proxy — NOT word/transcription confidence** — from a direct saaras:v3 probe (`language_code=unknown`) on Forge, because the `/ask` response does not surface it. *post-STT ms* = `trace.total_ms` (sum of non-STT stages; the <200 ms budget path). *decision-correct%*: en/hi in-corpus must `answer`; OOD must abstain.

**Measurement boundary.** `stt_stage_ms` is timed **server-side** around Sarvam's round-trip inside the harness; it **excludes** the client→server audio upload. Client wall time (recorded separately) ran ~1.7–2.0× the server STT time from a Windows client over the public Tailscale funnel and is **not** used for RTF.

**N.** en gold = 10 utterances × 7 conditions; hi gold = 10 × 7; OOD = 4 × 4; each fidelity language = 2 × 4. Every cell's N is shown in its table. language_probability N per cell is shown separately (bounded subset).

## Conditions

| condition | definition |
|---|---|
| clean | synthesized speech, no degradation |
| white20 / white10 / white5 | additive white Gaussian noise at SNR 20/10/5 dB |
| babble10 | sum of ≤4 other synthesized utterances (same language), SNR 10 dB |
| speed0.9 / speed1.1 | resample to 0.9× / 1.1× speed (pitch-shifting; caveat) |

SNR scaling: noise gain `k = rms(s) / (rms(n)·10^(SNR/20))` so `20·log10(rms(s)/rms(k·n)) = SNR`. 16 kHz mono 16-bit; clip-guarded by uniform down-scaling (SNR preserved).

## English (in-corpus)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N)  decision-correct% (N) |
|---|---|---|---|---|---|---|---|---|
| clean | 10 | 1.000 | 0.937 | 0.174 | 0.163 | 0.256 | 15.8 | 0.969 (10)  100.0 (10) |
| white20 | 10 | 1.000 | 0.937 | 0.157 | 0.156 | 0.212 | 17.6 | - (0)  100.0 (10) |
| white10 | 10 | 0.975 | 0.927 | 0.164 | 0.158 | 0.228 | 21.7 | 0.946 (10)  100.0 (10) |
| white5 | 10 | 0.891 | 0.885 | 0.162 | 0.152 | 0.240 | 19.6 | 0.944 (10)  90.0 (10) |
| babble10 | 10 | 1.000 | 0.937 | 0.162 | 0.167 | 0.243 | 16.4 | 0.973 (10)  100.0 (10) |
| speed0.9 | 10 | 1.000 | 0.937 | 0.140 | 0.135 | 0.190 | 16.4 | - (0)  100.0 (10) |
| speed1.1 | 10 | 1.000 | 0.937 | 0.173 | 0.172 | 0.212 | 15.4 | - (0)  100.0 (10) |

## Hindi (in-corpus)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N)  decision-correct% (N) |
|---|---|---|---|---|---|---|---|---|
| clean | 10 | 0.986 | 0.975 | 0.169 | 0.162 | 0.229 | 16.1 | 0.986 (10)  100.0 (10) |
| white20 | 10 | 0.969 | 0.968 | 0.177 | 0.165 | 0.270 | 18.1 | - (0)  100.0 (10) |
| white10 | 10 | 0.943 | 0.965 | 0.171 | 0.163 | 0.244 | 23.6 | 0.954 (10)  90.0 (10) |
| white5 | 10 | 0.796 | 0.886 | 0.245 | 0.165 | 0.902 | 39.7 | 0.830 (10)  60.0 (10) |
| babble10 | 10 | 0.983 | 0.976 | 0.194 | 0.159 | 0.516 | 14.3 | 0.985 (10)  100.0 (10) |
| speed0.9 | 10 | 0.969 | 0.973 | 0.169 | 0.144 | 0.318 | 14.5 | - (0)  100.0 (10) |
| speed1.1 | 10 | 0.986 | 0.975 | 0.180 | 0.175 | 0.258 | 14.8 | - (0)  100.0 (10) |

## OOD (must-abstain, en)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N)  decision-correct% (N) |
|---|---|---|---|---|---|---|---|---|
| clean | 4 | 1.000 | 0.944 | 0.123 | 0.127 | 0.143 | 68.6 | 0.971 (4)  100.0 (4) |
| white10 | 4 | 0.903 | 0.923 | 0.163 | 0.149 | 0.234 | 63.2 | 0.929 (4)  100.0 (4) |
| white5 | 4 | 0.914 | 0.907 | 0.240 | 0.236 | 0.389 | 63.4 | - (0)  100.0 (4) |
| babble10 | 4 | 1.000 | 0.944 | 0.126 | 0.128 | 0.146 | 68.7 | - (0)  100.0 (4) |

## Bengali (fidelity)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N) |
|---|---|---|---|---|---|---|---|---|
| clean | 2 | 1.000 | 1.000 | 0.097 | 0.097 | 0.105 | 55.0 | 0.996 (2) |
| white10 | 2 | 0.900 | 0.986 | 0.101 | 0.101 | 0.104 | 61.2 | - (0) |
| babble10 | 2 | 1.000 | 1.000 | 0.103 | 0.103 | 0.108 | 56.4 | - (0) |
| speed1.1 | 2 | 1.000 | 1.000 | 0.113 | 0.113 | 0.124 | 62.9 | - (0) |

## Tamil (fidelity)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N) |
|---|---|---|---|---|---|---|---|---|
| clean | 2 | 1.000 | 1.000 | 0.099 | 0.099 | 0.104 | 71.8 | 0.996 (2) |
| white10 | 2 | 0.917 | 0.986 | 0.103 | 0.103 | 0.103 | 94.6 | - (0) |
| babble10 | 2 | 1.000 | 1.000 | 0.098 | 0.098 | 0.107 | 80.1 | - (0) |
| speed1.1 | 2 | 1.000 | 1.000 | 0.111 | 0.111 | 0.112 | 67.4 | - (0) |

## Telugu (fidelity)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N) |
|---|---|---|---|---|---|---|---|---|
| clean | 2 | 0.900 | 0.988 | 0.105 | 0.105 | 0.113 | 57.5 | 0.982 (2) |
| white10 | 2 | 0.675 | 0.944 | 0.098 | 0.098 | 0.103 | 49.6 | - (0) |
| babble10 | 2 | 0.900 | 0.988 | 0.103 | 0.103 | 0.105 | 54.7 | - (0) |
| speed1.1 | 2 | 0.900 | 0.988 | 0.115 | 0.115 | 0.116 | 52.2 | - (0) |

## Marathi (fidelity)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N) |
|---|---|---|---|---|---|---|---|---|
| clean | 2 | 1.000 | 1.000 | 0.110 | 0.110 | 0.118 | 63.0 | 0.991 (2) |
| white10 | 2 | 1.000 | 1.000 | 0.193 | 0.193 | 0.281 | 39.9 | - (0) |
| babble10 | 2 | 1.000 | 1.000 | 0.100 | 0.100 | 0.102 | 38.5 | - (0) |
| speed1.1 | 2 | 1.000 | 1.000 | 0.147 | 0.147 | 0.181 | 42.9 | - (0) |

## Gujarati (fidelity)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N) |
|---|---|---|---|---|---|---|---|---|
| clean | 2 | 0.917 | 0.985 | 0.181 | 0.181 | 0.261 | 32.6 | 0.999 (2) |
| white10 | 2 | 0.833 | 0.970 | 0.126 | 0.126 | 0.135 | 43.1 | - (0) |
| babble10 | 2 | 0.833 | 0.970 | 0.118 | 0.118 | 0.121 | 47.4 | - (0) |
| speed1.1 | 2 | 0.917 | 0.985 | 0.138 | 0.138 | 0.143 | 34.8 | - (0) |

## Kannada (fidelity)

| condition | N | token-F1 | char-sim | RTF mean | RTF P50 | RTF max | post-STT ms (mean) | lang_prob (mean, N) |
|---|---|---|---|---|---|---|---|---|
| clean | 2 | 1.000 | 1.000 | 0.106 | 0.106 | 0.121 | 40.7 | 0.999 (2) |
| white10 | 2 | 1.000 | 1.000 | 0.101 | 0.101 | 0.115 | 34.6 | - (0) |
| babble10 | 2 | 1.000 | 1.000 | 0.105 | 0.105 | 0.121 | 33.2 | - (0) |
| speed1.1 | 2 | 1.000 | 1.000 | 0.104 | 0.104 | 0.111 | 35.8 | - (0) |

## Degradation narrative (computed from the grid)

- **English token-F1:** clean 1.000 · white10 0.975 · white5 0.891 (Δ -0.109) · babble10 1.000
- **Hindi token-F1:** clean 0.986 · white10 0.943 · white5 0.796 (Δ -0.190) · babble10 0.983

- **English RTF (mean):** clean 0.174 · white10 0.164 · white5 0.162 · speed0.9 0.140 · speed1.1 0.173 — additive noise leaves RTF ~flat (STT time tracks duration, not SNR); speed changes shift it via the duration denominator.
- **Hindi RTF (mean):** clean 0.169 · white10 0.171 · white5 0.245 · speed0.9 0.169 · speed1.1 0.180 — additive noise leaves RTF ~flat (STT time tracks duration, not SNR); speed changes shift it via the duration denominator.

- **English language_probability (language-ID proxy):** clean 0.969 · white10 0.946 · white5 0.944 (N per cell in the tables).
- **Hindi language_probability (language-ID proxy):** clean 0.986 · white10 0.954 · white5 0.830 (N per cell in the tables).

- **Decision robustness (answer% for in-corpus, abstain% for OOD):**
  - English in-corpus: clean 100%, white20 100%, white10 100%, white5 90%, babble10 100%, speed0.9 100%, speed1.1 100%
  - Hindi in-corpus: clean 100%, white20 100%, white10 90%, white5 60%, babble10 100%, speed0.9 100%, speed1.1 100%
  - OOD (abstain correct): clean 100%, white10 100%, white5 100%, babble10 100%

- **Fidelity languages, clean token-F1 (best→worst):** Bengali 1.000 · Tamil 1.000 · Marathi 1.000 · Kannada 1.000 · Gujarati 0.917 · Telugu 0.900

## Anomalies (every decision failure + RTF outlier, from the raw grid)

| cell | decision | token-F1 | reference | transcript |
|---|---|---|---|---|
| en_heart / white5 | small_talk | 0.80 | how does the heart work | How does the help work? |
| hi_liver / white10 | abstain_ood | 0.80 | यकृत का कार्य क्या है | यकुत का कार्य क्या है? |
| hi_diabetes / white5 | abstain_ood | 0.00 | मधुमेह क्या है | मधुमेश आहे. |
| hi_gravity / white5 | abstain_ood | 0.67 | गुरुत्वाकर्षण क्या है | गुरुस्वाकर्शन क्या है? |
| hi_bloodpressure / white5 | abstain_ood | 0.83 | उच्च रक्तचाप के कारण क्या हैं | उच्च रक्षा के कारण क्या हैं? |
| hi_liver / white5 | abstain_ungrounded | 0.60 | यकृत का कार्य क्या है | या क्या कार्य किया है? |

RTF outliers (> 0.4):
- `hi_diabetes / white5`: RTF 0.902 (stt 775 ms on 859 ms audio) — outliers concentrate on the shortest clips under heavy degradation.
- `hi_diabetes / babble10`: RTF 0.516 (stt 443 ms on 859 ms audio) — outliers concentrate on the shortest clips under heavy degradation.

Reading the failures: sub-5 dB white noise corrupts content words (e.g. heart→help, मधुमेह→मधुमेश), which flips retrieval below the OOD gate (abstain) or — in one case — reroutes to small-talk. The guardrail direction is safe: noise produced NO wrong answers, only abstentions/reroutes; OOD abstain held at 100% under every condition tested.

Formatting floor: clean char-sim < 1.0 for en/hi is dominated by saaras adding punctuation/capitalization the reference lacks (e.g. trailing '?'), not by misrecognition; token-F1 normalizes case/punct so en clean = 1.000. The residual clean token-F1 deficits (hi 0.986, te 0.900, gu 0.917) are strict-scoring artifacts at the orthographic-variant level — हड्डियां/हड्डियाँ (anusvara vs chandrabindu), ప్రతిరోజూ/ప్రతిరోజు and ઉગે/ઊગે (vowel length) — every word is otherwise correct. The scorer is deliberately diacritic-strict; treat these as the metric's floor, not STT word errors.

## Caveats (honest)

- **Synthetic speech ≠ human speech.** Every utterance is Sarvam **bulbul:v2** TTS (single speaker `anushka` per language). No disfluency, no real-mic channel, no accent diversity, model-native pronunciation. These numbers **upper-bound** STT quality; a human-voiced set would score lower, especially under noise. The browser front-end's Chromium tone/AGC path is **not** exercised here.
- **Single speaker, single TTS model.** Speaker/model idiosyncrasies confound the per-language comparison; treat cross-language deltas as indicative, not definitive.
- **language_probability is a LANGUAGE-ID proxy**, not word/transcription confidence. It is sourced from a **direct** saaras:v3 probe (`language_code=unknown`) run on Forge, because the deployed `/ask` response does not surface it. The audio is byte-identical to the `/ask` grid (same clean WAV, same deterministic seed, same transforms).
- **`stt_stage_ms` excludes client→server upload** (server-side timing). RTF is therefore a server/model figure, not an end-user latency. Client wall time was ~1.7–2× larger from a Windows client over the public Tailscale funnel.
- **speed conditions pitch-shift** (linear-interp resample, not a formant-preserving time-stretch), so they conflate rate and pitch effects.
- **Decision correctness is only meaningful for en/hi** (the corpus is MSMARCO-XI hi+en, 20k docs). Fidelity languages measure transcription + RTF only. Gold queries were pre-validated to `answer`/`abstain` on the live index via the keyless `/ask_text` endpoint before any audio was synthesized.
- **Sample maximum, not "P100".** The max column is the worst single observation at the stated N; it is not an SLA.

## Measurement environment

- STT model: **saaras:v3** (mode=transcribe), the model the deployed harness uses.
- Server: FastAPI harness on the Forge box (`goquest-Z790-AORUS-ELITE-AX`), reached at `https://goquest-z790-aorus-elite-ax.tail16e418.ts.net` (Tailscale Funnel → 127.0.0.1:8000).
- The public funnel returned transient 502s during a lead redeploy mid-run; all calls retry with backoff and none were recorded as failures.
- Fixtures + every `/ask` response are cached under `eval/fixtures/` so re-runs cost zero new Sarvam calls.

