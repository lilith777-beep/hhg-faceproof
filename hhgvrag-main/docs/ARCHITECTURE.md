# ARCHITECTURE — hhgvrag (Voice-Enabled RAG)

Locked stack (2026-08-14): **Sarvam** STT · **BGE-M3** embeddings · **Qdrant** hybrid vector DB ·
**local 3B** generation (Qwen2.5-3B / Llama-3.2-3B-Instruct) · **FastAPI on Modal** · static
frontend on Vercel. Everything on the <200 ms path is co-located on one Modal GPU box.

## Request flow (live path)
```
[browser mic] --audio--> FastAPI /ask
   1. Sarvam STT            audio -> query text            (upstream of the 200ms budget)
   ┌─────────────── 200 ms budget starts (retrieval -> final output) ───────────────┐
   2. embed query           BGE-M3 dense+sparse            ~10-30 ms
   3. retrieve              Qdrant hybrid + metadata filter ~10-40 ms
   4. guardrail: OOD gate   top-k score threshold          ~1 ms   -> maybe ABSTAIN
   5. generate              local 3B, grounded prompt      ~60-120 ms
   6. guardrail: grounding  answer ⊆ context (entailment/citations) ~5-20 ms -> maybe ABSTAIN
   └──────────────────────────────── < 200 ms ─────────────────────────────────────┘
-> {answer, citations[], abstained, per-stage latency trace}
```
Input safety (unsafe/inappropriate) is checked on the query text right after STT, in parallel
with embedding, so it doesn't add to the critical path.

## Offline index build (the "vast chunking" showcase)
A reproducible script (`src/index_build.py`, runs on Modal GPU) that:
1. Loads a documented slice of `ai4bharat/MSMARCO-XI` (record exact languages + row count).
2. Runs it through **every** chunking strategy in `src/chunking.py` (see that file's docstring).
3. Embeds each strategy's chunks with BGE-M3, builds a Qdrant collection per strategy.
4. **Evaluates** strategies on a held-out query set (recall@k, MRR, nDCG) → `eval/chunking_report.md`.
5. Promotes the winning strategy's collection to the live path; keeps the rest as evidence.

## The harness (`src/harness.py`)
NOT prompt-in/text-out. A typed, staged orchestrator:
- **Typed I/O** — every stage takes/returns a Pydantic model (`src/schemas.py`).
- **Stages as tools** — `embed`, `retrieve`, `safety`, `ood_gate`, `generate`, `ground_check`;
  each is independently timed, retried (bounded backoff), and has a timeout + fallback.
- **Error recovery** — any stage failure degrades gracefully (e.g. retrieval empty → abstain;
  generation timeout → extractive fallback from top passage).
- **Trace** — a `QueryTrace` with per-stage latency + decisions is returned and logged; the
  latency analytics (P50/P70/P100) are computed from these traces over a test-query set.

## Guardrails (`src/guardrails.py`) — "knows when NOT to answer"
1. **Input safety** — Llama-Guard-class check (or lightweight classifier) on the query;
   unsafe → refuse with a safe message.
2. **Off-topic / OOD gate** — if top-k retrieval scores are all below τ, the corpus doesn't
   cover the question → abstain ("I don't have grounded information on that").
3. **Grounding / anti-hallucination** — the generated answer must be supported by the retrieved
   context: citation-span overlap + a fast entailment check; if unsupported → abstain or return
   only the grounded portion. Every non-abstain answer carries citations to passage-ids.

## Latency analytics (`eval/latency.py`)
Replays N (≥100) test queries through the live harness, collects `QueryTrace` timings, reports
**P50 / P70 / P100** for the total retrieval→output path and per stage → `eval/latency_report.md`.

## Repo layout
```
hhgvrag/
├── CLAUDE.md · docs/ARCHITECTURE.md · docs/TASKS.md
├── src/  config.py schemas.py chunking.py embeddings.py retrieval.py
│         guardrails.py generation.py harness.py index_build.py api.py modal_app.py
├── eval/ chunking_report.md latency.py latency_report.md
├── web/  (static mic frontend -> Vercel)
├── tests/  test_chunking.py test_harness.py test_guardrails.py
└── requirements.txt · README.md
```
