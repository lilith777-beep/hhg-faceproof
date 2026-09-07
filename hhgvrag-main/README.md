# hhgvrag — Voice-Enabled RAG

Voice → Sarvam STT → hybrid retrieval over MSMARCO-XI → grounded answer, **<200 ms** retrieval→output.
Built for **Hacker House Goa 2026 Task #2** · **#RAGInGoa**

## Pipeline

```
🎙️ Voice ──→ Sarvam STT ──→ Normalize (R3) ──→ Route (R2) ──→ Cache check (R5a)
                                                                     │
                              ┌──────────────────────────────────────┘
                              ↓
                     Embed ──→ Retrieve (Qdrant hybrid) ──→ Rerank (R1)
                              ↓
                     OOD gate ──→ Generate (extractive | LLM quality) ──→ Grounding
                              ↓
                         Answer | Abstain | Refuse
```

**Stack:** Sarvam STT · BGE-M3 (dense+sparse) · Qdrant (hybrid, HNSW) · RAPTOR tree ·
semantic response cache · session context · extractive default / Qwen-2.5-3B quality mode ·
FastAPI on Modal (A10G) · static frontend on Vercel.

## Local Dev

```bash
cd hhgvrag
python -m venv .venv --python=python3.11
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# run all 85 tests (CPU, no keys needed)
.venv/Scripts/python tests/test_pipeline.py
.venv/Scripts/python tests/test_features.py
.venv/Scripts/python tests/test_raptor_cag.py
.venv/Scripts/python tests/test_chunking.py
.venv/Scripts/python tests/test_build_eval_api.py
.venv/Scripts/python tests/test_stress.py
.venv/Scripts/python tests/test_ingest.py

# local API server (mocked models, in-memory Qdrant)
uvicorn src.api:app --reload --port 8000
# then open web/index.html, set backend to http://localhost:8000
```

## Deploy (Modal + Vercel)

```bash
pip install modal
modal token new
modal secret create sarvam SARVAM_API_KEY=<your-sarvam-key>
modal secret create hf HF_TOKEN=<your-hf-token>

# 1. Build the index (offline, ~30 min on A10G)
modal run src/modal_app.py::build_index

# 2. Deploy the GPU service (prints the Modal URL)
modal deploy src/modal_app.py

# 3. Point the frontend at the Modal URL, deploy on Vercel
#    In web/index.html, save the Modal URL via the "Backend" input
#    Deploy web/ on Vercel (or any static host)
```

## Architecture

| Module | Role |
|---|---|
| `src/stt.py` | Sarvam STT (saaras:v3, WebM/Opus, auto-detect language) |
| `src/normalize.py` | R3: noisy-ASR repair (fillers, homophones, phonetic keys, confidence-aware) |
| `src/router.py` | R2: Indic language detection + intent triage (qa/chitchat/meta) |
| `src/reranker.py` | R1: BGE-reranker-v2-m3 (GPU) / LexicalReranker (CPU) |
| `src/raptor.py` | R4: RAPTOR tree (cluster→summarize→embed, multi-level retrieval) |
| `src/cache.py` | R5: semantic response cache + session context |
| `src/chunking.py` | 6 strategies (fixed, recursive, sentwin, passage, semantic, hierarchical) |
| `src/embeddings.py` | BGE-M3 (GPU) / HashEmbedder (CPU tests) |
| `src/retrieval.py` | Qdrant hybrid dense+sparse (RRF) |
| `src/guardrails.py` | Safety + OOD gate (dense+lexical) + grounding (NLI+negation) |
| `src/generation.py` | ExtractiveGenerator (default) / LocalLLMGenerator (quality mode) |
| `src/harness.py` | Typed orchestrator: staged, timed, retried, deadline-enforced |
| `src/api.py` | FastAPI: /ask_text, /ask (voice), /health |
| `src/topics.py` | Topic classification: keyword seeds + embedding clustering |
| `src/modal_app.py` | Modal GPU deployment (A10G, co-located everything) |
| `eval/` | Chunking A/B eval, latency P50/P70/P100, OOD calibration |
| `web/index.html` | Hold-to-talk mic + text, live latency breakdown |

## Dataset

`ai4bharat/MSMARCO-XI` — cross-lingual Indian-language passage retrieval.
Indexed: English + Hindi passages, dedup'd, ≤50k docs. RAPTOR tree built on top.

## Guardrails (demoable)

- **Off-topic** → abstain ("I don't have grounded information on that")
- **Unsafe** → refuse
- **Ungrounded** → abstain (grounding check + negation guard)
- **Greeting/meta** → intentional small-talk (no wasted retrieval)
