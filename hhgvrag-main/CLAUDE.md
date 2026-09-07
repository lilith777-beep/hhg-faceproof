# CLAUDE.md — hhgvrag (Task #2)

Task-scoped context for **Hacker House Goa 2026 · Task #2 — Voice-Enabled RAG**.
Root rules in `../CLAUDE.md` still apply. Read this top-to-bottom before touching code.

## The task in one line
A user **speaks** a question → we **transcribe** it → **retrieve** grounded context from the
provided dataset → **generate** an answer — end to end, with a real **harness** and real
**guardrails**, and the retrieval→output path under **200 ms**.

Pipeline: `Voice → Speech-to-text → Chunking / Retrieval (vector DB) → Answer generation`

## Hard requirements (from the official brief — do not drift)
1. **STT:** use **Sarvam _or_ ElevenLabs** (pick exactly one).
2. **Chunking:** must be *vast* — NOT a single naive fixed-size split. Show real thought:
   multiple strategies, overlap handling, semantic vs fixed-size, metadata-aware, hierarchical.
3. **Latency:** chunking + vector-DB retrieval + everything through to **final output < 200 ms**.
4. **Latency analytics:** report **P50 / P70 / P100** across a reasonable number of test queries
   (not a single best-case run).
5. **Harness:** structured orchestration around the model — tool calls, retries, structured
   input/output, error recovery — NOT a raw prompt-in/text-out call.
6. **Guardrails:** off-topic handling, unsafe/inappropriate input handling, hallucination /
   grounding checks. The system must know **when NOT to answer**, not just how to answer.

## Dataset
**`ai4bharat/MSMARCO-XI`** (HuggingFace) — cross-lingual (Indian languages) MS-MARCO passage
retrieval. It ships as passages + queries; treat passages as the corpus. NOTE: verify size and
language splits before indexing — we likely index a curated/sampled subset for the demo so the
<200 ms budget and hosting hold, and document exactly what we indexed.

## Deliverables & deadline
- **GitHub repo** + **live working link** + **2 videos**: (1) team/process, 90 s; (2) end-to-end demo.
- **Promotion (mandatory):** BOTH videos posted by **every team member** on **Instagram, X, and
  LinkedIn** (≥1 IG public), every post tagged **`#RAGInGoa`**.
- **Deadline: 2026-08-22 23:59. NO resubmissions — submit only when final.**
- Submission form: https://forms.gle/MNvCjcv23Hn2Eeu58

## The crux: the < 200 ms budget
200 ms for retrieval→**final output** (i.e. *including generation*) is the whole engineering
story. STT is upstream of this budget (its own API latency). Indicative budget:

| Stage | Target |
|---|---|
| Query embedding | ~10–30 ms |
| Vector retrieval (HNSW, hybrid dense+sparse) | ~10–40 ms |
| Grounding gate / guardrail | ~5–20 ms |
| Answer generation (SHORT grounded answer) | ~60–120 ms |
| **Total** | **< 200 ms** |

The killer is **generation network RTT**. A remote LLM API (Groq/Cerebras in the US) can cost
100–250 ms of round-trip alone from an India-hosted backend — that blows the budget. Reliable
<200 ms almost certainly needs a **small local generation model co-located** with the embedder +
vector DB (zero network hops), with a fast hosted LLM as an optional higher-quality (over-budget)
mode we measure and report honestly.

## Recommended architecture (PENDING human confirmation — see the questions)
- **STT:** Sarvam (Indian-language-first, matches the ai4bharat corpus + the judges) — or ElevenLabs.
- **Embeddings:** BGE-M3 (multilingual; dense + sparse + ColBERT in one) → enables genuine hybrid
  retrieval, which is half the "vast chunking/retrieval" story.
- **Vector DB:** Qdrant (embedded/self-hosted, hybrid search, metadata filters, HNSW) co-located
  with the backend.
- **Chunking (the showcase):** build an offline pipeline that produces MULTIPLE indexes and we
  A/B them on retrieval metrics (recall@k, MRR): (a) fixed-size + overlap baseline, (b) passage-
  aware (respect MS-MARCO passage boundaries), (c) semantic/embedding-boundary chunking,
  (d) hierarchical parent-child, (e) metadata-aware (language, passage-id, source). Pick the
  winner for the live path; keep the comparison as evidence.
- **Generation:** small local instruct model (e.g. Qwen2.5-3B / Llama-3.2-3B) on GPU for the
  <200 ms path; Cerebras/Groq (Llama-3.3-70B) as an optional quality mode.
- **Harness:** a custom **typed** orchestrator (Pydantic I/O), staged tool-calls
  (embed→retrieve→guard→generate→ground-check), per-stage timeouts + retries + fallbacks, and a
  per-query trace object feeding the latency analytics.
- **Guardrails:** (a) input safety filter (Llama-Guard-class or lightweight classifier),
  (b) off-topic / OOD gate via retrieval-score threshold, (c) grounding/entailment check on the
  answer vs retrieved context → graceful **abstain** ("not enough grounded context to answer").
- **Stack:** Python + FastAPI backend (async); lightweight web frontend (mic capture → transcript
  + answer + sources + live latency breakdown). Host the backend where the live link stays <200 ms.

## Build discipline (same spirit as Task 1)
- **Measure, don't claim.** Every latency number comes from the harness trace over N real queries.
- **Guardrails are demoable.** The abstain path must be shown, not asserted.
- **Reproducible index build.** Offline chunk/index step is a script with a documented dataset slice.
- **One decision locked before core build** (below), recorded here.

## Decisions — LOCKED 2026-08-14 (by the human)
1. **STT = Sarvam** (Saarika) — Indian-language-first, matches the corpus + judges.
2. **<200 ms generation = local small model** — Qwen2.5-3B / Llama-3.2-3B-Instruct on a
   co-located GPU, zero network hop. Hosted 70B (Cerebras/Groq) only as an out-of-budget
   "quality mode" we measure honestly if we add it.
3. **Retrieval = Qdrant + BGE-M3** — hybrid dense+sparse, metadata filters (language,
   passage-id, source), HNSW; co-located with the backend.
4. **Backend = Python + FastAPI on Modal** (GPU + Qdrant + local 3B on ONE box for the
   <200 ms path); **frontend = static on Vercel** (mic capture → transcript + answer +
   sources + live latency breakdown).

## Build order (staged — do not start a stage while the prior is red)
`scaffold → data+index (offline, multi-strategy chunking + eval) → RAG harness (typed,
guardrailed) → latency analytics (P50/P70/P100) → API + frontend → Modal/Vercel deploy → verify`.
Full design in `docs/ARCHITECTURE.md`; live task board in `docs/TASKS.md`.
