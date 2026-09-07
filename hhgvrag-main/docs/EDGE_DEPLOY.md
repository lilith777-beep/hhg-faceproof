# Edge Deployment — hhgvrag on a laptop (i7 / 16GB / MX150 2GB)

The **fallback tier** of the two-backend failover: a CPU-only, GPU-free hhgvrag that answers
grounded questions locally, so the demo survives Forge being down. Lightweight by design —
vectorless PageIndex retrieval + extractive composition, no BGE-M3/reranker/7B in the path.

## What runs where
| Piece | Edge tier | Notes |
|---|---|---|
| Retrieval | PageIndex vectorless (inverted postings, CPU) | ~0.3–1ms |
| Answer | Extractive composer (CPU, verbatim + cited) | <2ms |
| Gate | calibrated lexical (CPU) | honest abstain preserved |
| STT | Sarvam API (needs net) OR whisper-tiny on-device | pass `--stt tiny` |
| Quality (optional) | Sarvam-1 / Qwen Q4 GGUF via llama-cpp | streams, seconds |
| Corpus | a **subset** JSONL (fits RAM) | export step below |

## One-time setup

### 1. Export a corpus subset from Forge (do while Forge is up)
The full 876k index is too big for 16GB RAM; export a subset (start ~50k leaves):
```bash
# on Forge (or wherever the built Qdrant lives):
# BALANCED (recommended): even per-language quota so every language is searchable on edge.
# A plain first-N scroll is Hindi-heavy (51% hi; bn/or/pa/mr/sa/ne get a handful) because
# the collection's scroll order isn't language-balanced.
python src/edge.py --export-corpus edge_corpus.jsonl --limit 50000 \
    --qdrant-url http://localhost:6333 --collection msmarco_xi_val14_73ca3e90__passage \
    --languages hi,bn,ta,te,kn,ml,mr,gu,pa,or,ur,en,sa,ne
# then copy edge_corpus.jsonl to the laptop
# NOTE: mr(137)/sa(138)/ne(20) are thin in the SOURCE corpus (MSMARCO-XI) itself — even
# the full Forge tier is limited there; the quota just takes all available.
```

### 2. On the laptop (Windows/Linux): deps
Needs Python ≥3.8 (on the target laptop: `py -3.11`; system default 3.7 is too old).
```bash
py -3.11 -m venv .venv-edge          # Windows; on Linux: python3 -m venv .venv-edge
.venv-edge/Scripts/pip install fastapi uvicorn numpy qdrant-client python-multipart
# python-multipart is REQUIRED — the shared /ask audio endpoint uses Form/File uploads
# optional offline STT:  pip install faster-whisper
# optional CPU LLM polish:  pip install llama-cpp-python   (+ a Q4 GGUF file)
```
(No torch/CUDA needed for the vectorless+extractive path.)

### 3. Run the edge server
```bash
.venv-edge/Scripts/python src/edge.py --corpus edge_corpus.jsonl --port 8010
# 8010 because :8000 is often taken on dev laptops; pick any free port
# VOICE: set SARVAM_API_KEY in the environment -> Sarvam saaras:v3 (best Indic quality).
#        The laptop is online (it runs the tunnel), so Sarvam — a plain HTTP API — is the
#        right choice on edge too; whisper is only the offline fallback. To use Sarvam:
#          [Environment]::SetEnvironmentVariable('SARVAM_API_KEY', <key>, 'User')  # once
#        then launch WITHOUT --stt (an explicit --stt wins over the env key). With neither
#        key nor --stt, voice returns a typed error (never a 500, which would dead-mark the
#        backend). --stt tiny -> faster-whisper int8 on-device (offline, romanizes Indic).
# with a CPU LLM polisher: --polisher path/to/qwen2.5-0.5b-instruct-q4.gguf
```
Boot is ~15–30 s (50k-leaf PageIndex build), then it serves the same API as Forge:
`/health`, `/ask`, `/ask_text`, `/status`. On Windows probe `http://127.0.0.1:8010`
explicitly — `localhost` may resolve to IPv6 first and stall each request ~2 s.

Verified on the target laptop (i7 / 16 GB, 2026-08-17): EN answer · Hindi answer in
Devanagari (553 chars, 3 citations) · off-corpus abstain · 4–16 ms trace latency.

### 4. Give it a public URL (so the Vercel frontend can reach it)
```bash
# cloudflared quick tunnel (no account needed):
cloudflared tunnel --url http://localhost:8000
# -> prints https://<random>.trycloudflare.com  — this is the laptop's public URL
```

### 5. Wire it into the failover
In the site's **advanced · backends** panel, put the cloudflared URL in the **edge** slot
(priority 2). Forge stays priority 1 (full). Done — the frontend now:
- uses **Forge** when it's up (full fidelity, "full" badge),
- **auto-fails-over to the laptop** when Forge is down ("edge" badge),
- falls back to **Forge** if the laptop is off.

## Honest limits
- Edge answers are **lower fidelity** (lexical/vectorless vs hybrid+rerank) on a **corpus
  subset** — the tier badge shows "edge" so it's transparent.
- The laptop must be **on + tunneled** to serve as a fallback; if it's off, only Forge answers
  (which is the failover working correctly).
- The Indic tokenizer fix (`textnorm.WORD_RE`) is what makes the lexical path work on
  Devanagari — it's already in `src/`, so the edge tier inherits it.

## The pitch this unlocks
"A voice RAG that answers grounded questions in 14 Indian languages — **and keeps working on a
2GB-GPU laptop, offline, when the server is down.** No single point of failure." That's the
resilience story tonight's Forge outage makes concrete.
