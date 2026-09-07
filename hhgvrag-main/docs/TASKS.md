# TASKS — hhgvrag build board

## OWNERSHIP CLAIMS (production plan v4 — exclusive, do not cross)
- **Opus UI agent**: `web/index.html`, `tests/e2e/**`, playwright config.
  P0.2 Track 1 CLOSED 2026-08-16 (58/58 e2e, axe 0, streaming card handed off).
  Next claim (only after data artifacts signed): `web/data.html` + display JS + its tests.
- **Opus Data agent**: `src/index_build.py`, `src/chunking.py`, `src/embeddings.py`,
  `src/raptor.py`, `src/corpus_spec.py`, `eval/dry_run_build.py` + data tests.
  P0.4 protocol machinery DELIVERED (plan_split_multi, sealed_test.py, sizing). Idle —
  next claim: data_stats.json/plots after the versioned build verifies.
- **Ops agent**: `bin/**`, `ops/**` unit files, `docs/RUNBOOK.md`. Package staged on Forge;
  WAITING-HUMAN: `sudo bash ~/hhgvrag/ops/install.sh` + reboot drill.
- **Audio agent**: `src/stt.py`, audio parts of `src/api.py` (contract lead-approved). Idle.
- **Integration lead (ACTIVE)**: `src/config.py`, `src/harness.py`, `src/retrieval.py`,
  `src/generation.py`, `src/build_real_index.py`, `src/modal_app.py`, `src/api.py` contract,
  manifest signature, promotion, deployment, final claims.
  IN FLIGHT: versioned build `msmarco_xi_val14_73ca3e90` — plan realized on Forge
  (integrity all-zero), manifests at `eval/builds/73ca3e90/`, AWAITING-HUMAN signature.
  2026-08-16 evening: scope APPROVED by human (validation-SAMPLE wording binding; never
  "full validation"); signature withheld pending consolidated P0 report. Build script now
  phase-durable (markers, resume, guards). Systemd stack INSTALLED (human sudo): keys out
  of ps, linger on, funnel green. Drills GREEN: controlled-orphan sweep exact-kill,
  judge-now public path (EN/HI answer, OOD abstain). Reboot drill PERMANENTLY WAIVED —
  shared production host, owner directive; claim only "service-restart drill passed".
  Fresh full regression at candidate: 280 passed / 0 failed / 0 skipped / 199.6s.
- Plan of record: v4 (delegated-baking-treehouse) + the 2026-08-16 execution directive.
- Corpus contract note: full-validation = 97,941 rows/shard (measured) ⇒ ~16M chunks ⇒ 50h+,
  fails the P90≤8h sizing gate even leaf-only. Max-fit spec: first 6000 rows/shard,
  query-aligned across all 14 shards. Claims must say "validation-split sample", never "full".

Staged. Deadline **2026-08-22 23:59**. `[x]` done · `[~]` code-done, real-run pending keys · `[ ]` todo.
**Verification: 99/99 local tests green** (8 chunking + 15 pipeline + 13 RAPTOR/CAG + 10 features
+ 6 build/eval/api + 29 stress + 18 ingest). Three-way adversarial review 2026-08-15 (me + 2
independent agents): 2 BLOCKERs + 6 HIGHs found and fixed, each with a regression test.
Real GPU numbers from Modal (blocked on keys). **First deploy: default 20k docs, measure,
then scale deliberately.**

## S0 — Scaffold  ✅
- [x] Locked stack · `CLAUDE.md` · `docs/ARCHITECTURE.md`
- [x] `src/chunking.py` — 6 strategies, Indic-aware — **8/8 tests**
- [x] `src/schemas.py` typed contract · `src/config.py` · `requirements.txt`

## S1 — Data + offline index build  ✅ code / [~] real run
- [x] `src/embeddings.py` — BGE-M3 backend + `HashEmbedder` (CPU-testable)
- [x] `src/index_build.py` — `build_all_strategies` (6 collections) + `build_raptor_index` (7th) +
      MSMARCO-XI loader — **tested on synthetic corpus**
- [x] `eval/eval_chunking.py` — recall@k / MRR / nDCG per strategy + winner picker — **tested**
- [~] Real: `modal run src/modal_app.py::build_index` over the MSMARCO-XI slice → reports

## S2 — RAG harness (typed, guardrailed)  ✅
- [x] `src/retrieval.py` — Qdrant hybrid dense+sparse (RRF) + metadata filters — **tested (in-mem)**
- [x] `src/guardrails.py` — safety + OOD gate (dense+lexical) + grounding (NLI+negation) — **tested**
- [x] `src/generation.py` — ExtractiveGenerator (default <200ms) + LocalLLMGenerator (quality mode)
- [x] `src/harness.py` — staged tool-calls, timing, retries, deadline-enforced, fallback, `QueryTrace`,
      cache (R5a), session context (R5c), quality mode (H8) — **end-to-end tested**

## S3 — Latency analytics  ✅ code / [~] real numbers
- [x] `eval/latency.py` — P50/P70/P100 total + per-stage → `eval/latency_report.md` — **tested**
- [x] `eval/calibrate_ood.py` — M7 threshold calibration from labeled in-domain/OOD scores
- [~] Real P50/P70/P100 from the Modal box — must show P50 < 200 ms (extractive path)

## S4 — API + frontend  ✅
- [x] `src/api.py` — FastAPI `/ask_text` + `/ask` (audio) + `/health`; session_id + quality_mode — **tested**
- [x] `web/index.html` — hold-to-talk mic + text, quality mode toggle, session context, signal chips
      (cache hit, quality mode, repaired query, intent, rerank score, gate signal), live latency breakdown

## S5 — Deploy  [~] written, blocked on keys
- [x] `src/modal_app.py` — GPU box: BGE-M3 + Qdrant + ExtractiveGenerator (default) + 3B quality mode +
      RAPTOR tree + semantic cache + session context + reranker + router + normalizer; `build_index` fn
      now builds 6 strategy collections + RAPTOR
- [ ] **Human:** `modal token new` · secrets `sarvam` (SARVAM_API_KEY) + `hf` (HF_TOKEN) ·
      `modal run …::build_index` · `modal deploy` · put the Modal URL in `web/` · deploy `web/` on Vercel
- [ ] Verify the live link answers voice→grounded end-to-end; capture real P50/P70/P100

## S6 — Submit
- [x] `README.md` — architecture, local dev, deploy, test summary
- [ ] Demo recording (voice→answer + abstain + quality mode) · repo public-ready
- [ ] **Videos**: team/process 90 s + demo. Posts on IG/X/LinkedIn by every member — **#RAGInGoa**
- [ ] Submit https://forms.gle/MNvCjcv23Hn2Eeu58 (repo + live link + videos). No resubmissions.

## H — Hardening (adversarial review, 2026-08-14)
Two-pass review (mine + independent agent). **All confirmed findings fixed + verified**,
each with a behavioral regression test that fails on the old code.
- [x] **B1 (BLOCKER) RecursiveChunker char-soup** — `chunking.py`
- [x] **H1 data loader** — real MSMARCO-XI schema. `index_build.py`
- [x] **H2 STT** — `saaras:v3`; magic-byte mime sniff; `language_probability`. `stt.py`
- [x] **H3 mic** — mono/16k + noise suppression. `web/`
- [x] **H4 enforce stage timeouts** — `future.result(timeout)` → fallback. `harness.py`
- [x] **H5 OOD scale-mismatch** — dense top + lexical rescue. `guardrails.py`
- [x] **M3 grounding** — NLI + negation contradiction guard
- [x] **M4 voice guard** (STT error → ERROR, not 500) · **M9 nDCG≤1**
- [x] **H6 cumulative trace** · **H7 lazy api** · **m1 top_k** · **m2 empty-query** · **m4 spans** · **m5 verify_ready**
- [x] **M7 τ calibration** — `eval/calibrate_ood.py` written (local: τ=0.36 F1=1.0; run on Modal for BGE-M3 τ)
- [x] **H8 live path = grounded-extractive** — `ExtractiveGenerator` default (<200ms),
      `LocalLLMGenerator` = opt-in `quality_mode` (separately timed). Wired in harness + API + Modal + frontend.
      Cache skips quality mode (different output). Tests verify extractive default and quality override.
- [x] **minors** — M10 mooted by `_norm_key` dedup; m6 safety is inside the 200ms budget (correct);
      abstain-label clear; Qdrant single-replica is Modal default; `config` module name no conflict (all tests pass)

## R — Robustness v2 (features) — the audio-first RAG
**R1–R5 + H8 ALL DONE — 50/50 tests, zero regressions.**
- [x] **R1 reranker** (`reranker.py`) — BGEReranker + LexicalReranker; answerability signal
- [x] **R2 query router** (`router.py`) — Indic language + intent; small-talk; cross-lingual open
- [x] **R3 noisy-ASR** (`normalize.py`) — fillers + homophones + phonetic + confidence-aware
- [x] **R4 RAPTOR** (`raptor.py`) — cluster→summarize→embed tree; multi-level retrieval
- [x] **R5 CAG** (`cache.py`) — semantic response cache + session context
- [x] **H8 extractive default** — LLM = quality mode, separately timed

## I — Ingest pipeline hardening (2026-08-15)
- [x] **Batch embeddings** — `BGEM3Embedder` batches at 64 with progress logging (OOM prevention)
- [x] **Per-passage language detection** — `_detect_lang()` uses Unicode script blocks on actual
      text content, not MSMARCO config labels (English passages were being tagged as the config lang)
- [x] **Quality filtering** — `_passage_quality()` drops <5 words, >70% repetition, boilerplate
      (cookie/subscribe/URLs). Logs dropped count at ingest.
- [x] **RAPTOR LLM summaries** — Modal `build_index` now loads Qwen2.5-3B for real abstractive
      cluster summaries instead of extractive longest-sentence picks. Cleans up GPU memory after.
- [x] **Topic classification** — `topics.py`: `KeywordTopicClassifier` (seed vocabularies, CPU) +
      `EmbeddingTopicClassifier` (k-means on dense vectors, labels clusters by dominant keywords).
      Applied to every chunk at `build_all_strategies`. Modal uses embedding classifier (n=12 topics).
      Topic label stored in `chunk.extra.topic` → Qdrant payload for metadata filtering.
- [x] **Stress test** — `test_stress.py`: 18 tests covering concurrent load (16 threads × 10 queries),
      edge cases (empty/emoji/injection/null-bytes/5000-char), guardrail coverage, cache/session
      behavior, decision coverage, API stress, latency distribution
- [x] **Bug fixes found by stress test** — cache hit now records session turns; safety pattern matches
      "methamphetamine"; router detects "hi there"/"hey there" as chitchat

## A — Adversarial review round 2 (2026-08-15, me + 2 independent agents)
All fixed + regression-tested (99/99):
- [x] **BLOCKER: quality mode DOA** — 150ms generate cap killed every 3B call (3-8s) into a
      silent fallback that still claimed quality_mode=True, orphaning generation on the GPU.
      Now: own `generate_quality` stage (10s cap, outside the 200ms budget), honest flag on fallback.
- [x] **BLOCKER: quadratic RAPTOR seeding** — farthest-first recomputed the full matmul per seed
      (O(n·d·k²) ≈ hours at 50k leaves). Now incremental (one matvec/seed) + duplicate early-stop.
- [x] **HIGH: cross-session cache poisoning** — session-expanded answers were cached under the bare
      query embedding. Context-dependent turns now bypass cache get AND put; leak test added.
- [x] **HIGH: 200ms budget enforced end-to-end** — request deadline caps EVERY budget-scoped stage
      (incl. previously-untimed embed/route/cache) at remaining budget with a 15ms floor.
- [x] **HIGH: unbounded RAPTOR cluster → context overflow** — prompt capped (12 members / 6k chars).
- [x] **HIGH: semantic chunker per-doc GPU calls** — excluded from the GPU build by default
      (still in the local A/B eval); documented degeneracy on short passages.
- [x] **MED: eval/live corpus divergence** — quality filter moved INSIDE `_load_msmarco`
      (pre-doc-id), so qrels describe the shipped index. "javascript" removed from kill-list.
- [x] **MED: cache key ignores top_k** — key now includes eff_k; "xx" language tag → config fallback;
      topic lifted to top-level payload (filterable); async /ask → sync (event-loop freeze);
      meta-router word cap ("help me understand X" reaches retrieval); prefetch ≥ top_k;
      Modal `max_containers=1` + `concurrent(4)` (sessions/cache are per-container memory);
      GPU warmup in `@enter`; HF_HOME on the Volume; safety patterns broadened; STT 8s cap.
- [x] **Honest posture** — in-process Qdrant = exact scan (HNSW only with a real server, noted in
      config); first deploy defaults to `max_docs=20_000`; scale after measuring.

## P — Forge + PageIndex + 7B (2026-08-15, human-directed pivot)
- [x] **Forge-first hosting** (i9/64GB/RTX A6000 48GB, $0) — `src/server.py` (real-backend uvicorn,
      same wiring as Modal), `src/build_real_index.py` (on-prem index build), `requirements-gpu.txt`,
      `docs/FORGE.md` (venv → build → serve → Cloudflare Tunnel → latency evidence).
      Modal = built fallback (scale-to-zero, no min_containers).
- [x] **PageIndex vectorless retrieval** (`pageindex.py`) — beam descent over the RAPTOR tree by
      lexical scoring, ZERO embeddings at query time; `Query.retrieval_mode="pageindex"` toggle,
      frontend checkbox + 🌲 chip, cache isolated per mode, OOD/rerank/grounding unchanged.
      HYBRID STAYS DEFAULT (brief names a vector DB). Optional `llm_navigate` for quality mode.
      Local A/B (`eval/eval_retrieval_modes.py`): hit@3 8/8 both modes; retrieval 0.16ms vs 6.9ms.
- [x] **Quality mode → Qwen2.5-7B on vLLM** (`VLLMGenerator`, prefix-cached SYSTEM_V2 contextual
      prompt: grounded-only, forced [n] citations, language-matched answers, abstain string,
      1-3 sentences). HF `LocalLLMGenerator` = automatic fallback. Extractive <200ms path unchanged.
- [x] 10 new pageindex tests — **109/109 total**

## L — LIVE ON FORGE (2026-08-15) 🎯
**Public URL (PERMANENT, Tailscale Funnel): https://goquest-z790-aorus-elite-ax.tail16e418.ts.net**
(cloudflared quick tunnel stays as backup). **VOICE LIVE**: WAV→Sarvam STT 260ms perfect
transcript → pipeline 79.8ms.
**MEASURED P50/P70/P90/P100 = 51.8 / 55.9 / 60.3 / 78.7 ms** over 43 live queries (25 EN +
10 HI + 8 OOD) — zero over 200ms — with the FULL stack: BGE-M3 (11ms) → Qdrant-server HNSW
(7ms) → BGE-reranker-v2-m3 cross-encoder (34ms) → extractive+grounding (<1ms).
- [x] Forge deploy over SSH/Tailscale: conda py3.13 env, vLLM 0.27 (CUDA 12.4 + no-flashinfer
      sampler), Qdrant server in docker (HNSW), cloudflared tunnel
- [x] 20k-doc real index: 149k chunks / 6 collections + RAPTOR w/ 4,800 Qwen-3B batched
      abstractive summaries (55.5 min build); PageIndex tree 25,964 nodes
- [x] Real-run fixes: parquet shard loader (datasets streaming crashes on nested passages);
      vLLM fork-order + BaseException fallback + ninja/CUDA-12.4 PATH; transformers-native
      reranker (FlagEmbedding broke on transformers 5); HF generator template fix;
      setproctitle ghost-engine kill hygiene (nvidia-smi PID, not pkill)
- [x] M7 LIVE calibration: τ=0.5852 (F1 0.923, 40 real + 15 OOD) — crypto leak fixed;
      NEW cross-encoder VETO (rerank_top<0.03 → abstain even if dense passed) — fired live
- [x] Quality mode on vLLM: ~3.5s grounded answers, separately timed, honest fallback;
      reply-language forced from router detection
- [x] Hindi voice-corpus queries answer IN HINDI at ~53ms (cross-lingual live)
- Tradeoff (documented): PageIndex vectorless = ~20-35ms on term-anchored queries; abstains
  on paraphrase-style (lexical descent limit) — hybrid stays default
- [x] Voice path LIVE (Sarvam key wired): WAV → saaras:v3 260ms perfect transcript → 79.8ms pipeline
- [x] **Frontend LIVE: https://hhgvrag.vercel.app** (Vercel, project `hhgvrag`) → permanent
      Funnel backend. Judges can click both links today.
- [ ] Polish: systemd units (server+funnel survive Forge reboot) · README refresh with live
      numbers · demo-script uses extractive for EN (quality mode answers in corpus language)

## Blocked-on-human
Modal account (`modal token new`) · **Sarvam API key** · HF token · deploy · videos · submit.
Everything else is built, tested (50/50), and green.
