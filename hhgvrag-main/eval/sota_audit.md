# SOTA-RAG Audit — hhgvrag vs the 2025-2026 landscape

_Auditor pass, 2026-08-16. Scope: deep web research on the 2025-2026 SOTA RAG landscape, then an
audit of OUR pipeline against it, with an exact, budgeted plan to reach SOTA under a HARD
retrieval→answer budget of **200 ms on one RTX A6000**. Deadline 2026-08-22, code freeze 08-20._

**Sourcing legend.** Every non-obvious claim is tagged **[SOURCED]** (a cited URL supports it),
**[MEASURED]** (our own eval artifact in this repo), or **[INFERRED]** (synthesis/engineering
judgment from sourced+measured facts). URLs are collected in the appendix and inline at first use.

---

## 0. TL;DR — the verdict and the five moves that matter

Our pipeline is **already strong and, on the two axes the brief scores hardest (latency and
"knows when not to answer"), close to or at SOTA.** Live A6000: **P50 51.8 ms / P100 78.7 ms**,
0/43 over budget [MEASURED, `docs/TASKS.md`]. Extractive-verbatim composition + a calibrated
abstain gate is exactly the 2025-2026 faithfulness playbook (VerbatimRAG, LongCite, selective-QA)
[SOURCED]. BGE-reranker-v2-m3 is the current **MIRACL multilingual accuracy leader** [SOURCED],
so our core model choices are right.

The gap to SOTA is **not** a missing exotic component. It is **four cheap, high-confidence tuning
/ correctness fixes** plus **one measurement-discipline upgrade**, all inside budget:

| # | Move | Expected gain | ms cost | Days | Confidence |
|---|---|---|---|---|---|
| **1** | Fix cross-lingual fusion: ship **dense-only** (or heavily down-weighted score-fusion), retire equal-weight RRF | **up to +0.21 recall@10** cross-lingual | **−3 to −5 ms** | 0.5 | High (dir.) / Med (mag.) |
| **2** | Spend rerank headroom: **deepen pool top-24→48/64** and/or **ONNX/TensorRT fp16-int8** (1.5-1.8×) | +0.02-0.05 recall@8 (est.) | ~0 (fp16) / +20-40 ms (depth) | 1-2 | High |
| **3** | Replace **lexical grounding gate → small multilingual NLI** entailment (in-budget) | fewer false-grounded answers; better abstain precision | +8-12 ms | 1-2 | High |
| **4** | Make the abstain threshold **risk-coverage / split-conformal**, per-language, from a bigger calib set | principled coverage@risk; likely +coverage at equal risk | ~0 (O(1)) | 1-2 | High (method) |
| **5** | Run the **ablation + tuning grid** (below) on the 500-query calib split, then **one sealed 40k run** | locks the above with CIs | offline | 2-3 | — |

Everything else (a ColBERT arm, a tiny-LM query rewriter, adaptive depth routing) is **optional
upside to test in the grid, not a prerequisite for SOTA** — and several famous "advanced RAG"
ideas (HyDE, RAG-Fusion, Self-RAG, Cohere rerank, 9B gemma reranker) are **out-of-budget or
against our no-server constraint** and belong on the DO-NOT-DO list (§7).

---

## 1. Our pipeline, as verified in code (not as described)

Ground truth from the source, because a few live parameters differ from the brief's prose:

| Stage | Implementation (file) | Key params (verified) | Note |
|---|---|---|---|
| STT | Sarvam `saaras:v3` (`stt.py`) | 260 ms, RTF 0.17 | **outside** the 200 ms budget |
| Normalize | `normalize.py` `QueryNormalizer` | fillers/homophones/phonetic; **confidence→(k, ood_relax): ≥0.75→(k,0); 0.5-0.75→(1.5k,0.03); <0.5→(2k,0.06)** | already an adaptive-depth hook |
| Route | `router.py` `HeuristicRouter` | script-detect (10 Indic + en) + intent(qa/chitchat/meta); **`language_filter=False`** | cross-lingual left open |
| Embed | `embeddings.py` `BGEM3Embedder` | dense 1024 + learned sparse; **`return_colbert_vecs=False`** | **ColBERT vecs are NOT computed** (stronger than "discarded") |
| Cache | `cache.py` `SemanticResponseCache` | cos **τ=0.92**, 256 entries, key=(eff_k,mode), **ANSWER-only store** | session rewrite = **naive concat** `"{prev} {q}"` |
| Retrieve | `retrieval.py` `Retriever` | Qdrant dense + sparse, **Python-side RRF, `rrf_k=60`, EQUAL weight**, prefetch 40 | dense cosine kept for OOD |
| Leaf-expand | `retrieval.expand_to_leaves` | RAPTOR summaries → leaf descendants | live collection = `msmarco_xi__raptor` |
| Rerank | `reranker.py` `BGEReranker` | **bge-reranker-v2-m3**, fp16, `max_length=512`, batch 32, sigmoid; **pool `rerank_candidates=24` → top eff_k=8** | 34 ms, the dominant stage |
| OOD gate | `guardrails.ood_gate` | **primary = max dense cosine ≥ τ=0.585**; secondary lexical ≥0.34; **rerank rescue ≥`rerank_min`0.35 / veto <`rerank_veto`0.03** | τ from F1-opt on 55 queries |
| Compose | `generation.ExtractiveGenerator` | **MMR**, `max_support=5`, `support_floor=0.26`, `novelty_lambda=0.55`, `max_len_chars=1300`; per-segment `[n]`; verbatim | (brief said ~900 chars; code = 1300) |
| Ground gate | `guardrails.grounding_check` | **lexical support ≥0.5** + negation-contradiction guard; NLI optional (**quality mode only**) | |
| Quality (opt) | `generation.VLLMGenerator` Qwen2.5-7B | ~3.5 s, streamed | honestly out-of-budget |
| Alt mode | `pageindex.py` | vectorless idf-postings over leaves, beam 5 | per-query toggle |

### Measured baseline (the numbers to beat)
- **Latency, live A6000, 43 queries** [MEASURED, `docs/TASKS.md`]: P50/P70/P90/P100 =
  **51.8 / 55.9 / 60.3 / 78.7 ms**; stage means **embed 11 / retrieve 7 / rerank 34 /
  extract+ground <1 ms**. Stress (`eval/stress_report.md`): sustained P95 70.6 ms (mixed),
  **non-cache P95 90.5 ms**; gates A/B/C pass; **30-way cold burst → 100% ERROR** (embed/safety
  deadline; admission-control semaphore added). **Headroom for accuracy spend ≈ 110-148 ms.**
- **Retrieval quality, 20k calib index, N=500, single run, no CI** [MEASURED,
  `eval/retrieval_metrics.md`]:
  - Chunking winner **passage** nDCG@10 0.3859 / recall@10 0.4876 (raptor 0.3824).
  - **Rerank on/off @k8: recall 0.4331 → 0.7255 (+0.2924), MRR +0.1077.** ← biggest lever, deployed.
  - **Hybrid vs dense-only @passage top-10: hybrid 0.4876 vs dense-only 0.7021, Δ = −0.2145**
    (hybrid loses); win/loss/tie 19/228/253. ← the standout finding.
  - PageIndex vectorless recall 0.1589 (keep as alt only).
- **ASR robustness** [MEASURED, `eval/noise_report.md`]: clean token-F1 ~1.0 en/hi; degrades
  under ≤5 dB noise but **degrades to abstain, never to a wrong answer**; OOD abstain 100% under
  every condition. Guardrail direction is safe.

---

## 2. The 2025-2026 SOTA landscape (synthesized, cited)

**Two-stage cascade is the settled architecture**: recall-oriented first stage (ANN, top-100 to
top-1000) → precision cross-encoder rerank of the top 30-50 → 5-10 passages to the answerer;
hybrid + rerank buys +25-40% precision over naive RAG at modest cost [SOURCED,
digitalapplied.com hybrid-search 2026; redis.io reranking guide]. **We already run this shape.**

**Hybrid RRF (k=60) is the production default — but that advice is monolingual-centric.** The
generic "always hybrid, sparse adds +26% recall" figure is an English/BM25 result [SOURCED,
digitalapplied.com]. In **cross-lingual** retrieval the sparse arm collapses: BGE-M3's own paper
reports **MKQA cross-lingual R@100 Dense 75.1 / Sparse 45.3 / Dense+Sparse 75.3 (+0.2 only)** vs
**MIRACL monolingual Dense 69.2 / Sparse 53.9 / Dense+Sparse 70.4 (+1.2)** [SOURCED,
arxiv.org/abs/2402.03216]. Independent 2025 work: BM25 cross-lingual R@100 **10.4% vs dense
82.6%**, ~0.0 nDCG on cross-script pairs [SOURCED, arxiv.org/abs/2511.19324]. So our measured
hybrid loss is **direction-expected**; its unusually large magnitude points at **equal-weight RRF
over-weighting a near-useless sparse arm at recall@10** (§3, W1).

**RRF vs learned fusion.** Canonical RRF is **unweighted**; hand-tuned RRF weights **overfit and
don't transfer OOD** ("nDCG swings wildly", Bruch/Pinecone TOIS 2023) [SOURCED,
arxiv.org/abs/2210.11934]. **Convex Combination with one learned α (TMM-normalized) beats RRF on
all 9 tested datasets by ~3-7% nDCG** and converges with <5% of training data [SOURCED, same].
Weaviate's relativeScoreFusion is +6% recall over rankedFusion and is their default [SOURCED,
weaviate.io]. Takeaway: **either dense-only (cross-lingual) or a single learned-α score fusion —
not weighted RRF.**

**Late interaction / ColBERT.** MaxSim rerank is **not** a latency problem: ~2 ms for top-100 even
at heavy ColPali shapes on A100 (Flash-MaxSim 2026) [SOURCED, arxiv 2605.29517]; Qdrant 1.10+ has
native multivector (MAX_SIM, HNSW disabled, rerank-only) [SOURCED, qdrant.tech]. The tax is
**storage** — BGE-M3 colbert vecs are 1024-dim ≈ **100 GB/1M docs int8** — but **our corpus is
20-50k docs, so ≈2-5 GB: trivial** [INFERRED from SOURCED per-doc math]. Quality lift is **~+1
nDCG@10 over dense+sparse** on MIRACL (multi-vec 70.5 vs dense 69.2); authors' fusion weights
dense 1.0 / sparse 0.3 / colbert 1.0 [SOURCED, arxiv 2402.03216]. **Feasible for us, but small,
and likely redundant on top of a strong cross-encoder** — a grid candidate, not a must.

**Rerankers.** **BGE-reranker-v2-m3 is the MIRACL leader at 69.32 nDCG@10**, ahead of
Qwen3-Reranker-4B (67.52) and jina-reranker-v3 (66.50) [SOURCED, arxiv 2509.25085]. Its only weak
spot is English BEIR (56.51, ~5 behind jina-v3). **We keep it.** Free speed: sentence-transformers
ONNX/OpenVINO backends give **1.5× (fp16) to 1.8× (int8/TensorRT)** on GPU short-text [SOURCED,
sbert.net; nvidia TensorRT blog]. jina-reranker-v2-base-multilingual (278M, FlashAttention-2, ~2×
throughput, MIRACL 63.65) is the speed-first fallback [SOURCED, jina.ai]. **Avoid** mxbai-rerank-v2
(weak MIRACL 57.94), bge-reranker-v2.5-gemma2 (9B, too heavy at depth), Cohere v3.5 (API-only).

**Corrective / self-RAG under budget.** Only the **evaluator idea** is extractable: CRAG's
retrieval evaluator is a T5-large (0.77B) confidence gate [SOURCED, arxiv 2401.15884]; its
corrective action (web search + decompose-recompose) is out-of-budget. Self-RAG's reflection loop
**is** the generator (fine-tuned LM, segment beam search) — not a cheap side-gate [SOURCED, arxiv
2310.11511]. RAG-Fusion adds a multi-query LLM call, **~1.7× slower** [SOURCED, arxiv 2402.03367].
Cheap, no-LLM retrieval-quality signals do exist: Qdrant's June-2026 study "Predicting Weak
Retrieval Without an LLM" (max_score, dense_variance, evidence_coverage, retriever_divergence,
dense_agreement; AUC 0.73-0.77; Youden threshold; sub-ms) [SOURCED, qdrant.tech]. **These are the
right substrate for adaptive routing** (§6). Caveat: retrieval-side gates fire too eagerly — use
for escalation, not answer-correctness (TASR, arxiv 2606.13814) [SOURCED].

**Extractive composition + citations.** Our design is on-trend, not legacy. Verbatim-extractive
RAG is a recognized 2025 anti-hallucination pattern (**VerbatimRAG**, ACL BioNLP 2025) [SOURCED,
arxiv 2605.21102]; sentence-level `[n]` citations match **LongCite / LongBench-Cite** [SOURCED,
arxiv 2409.02897]; **MMR is still the production default** and the baseline every 2024-2026
diversity paper compares to; it is beaten only slightly, on factuality metrics, by cubic DPP
log-det selection [SOURCED, arxiv 2503.09249, 2608.03655]. **The one cheap upgrade the literature
flags: our grounding gate.** Purely lexical overlap is the exact "shallow token-matching" failure
mode — up to **55-57% false grounding on keyword-sharing distractors** (Wallat et al. 2024)
[SOURCED, arxiv 2412.18004]; the standard fix is a small **NLI/entailment** verifier (AutoAIS:
T5-TRUE, AttrScore) [SOURCED].

**Calibration / selective-QA.** A single F1-optimal τ is one point on a **risk-coverage curve**
with no guarantee; report **AURC** (or the 2024 fix **AUGRC**) and set the threshold by **split
conformal** for a distribution-free coverage guarantee at **O(1)** inference (CONFLARE calibrates
exactly a retrieval-similarity cutoff — our gate's shape) [SOURCED, arxiv 2407.01032, 2404.04287].
**Temperature/Platt scaling is monotonic → it cannot improve our abstain ordering or AURC**; it
only makes 0.585 read as a probability [SOURCED, arxiv 2402.05806]. Keep answerability (OOD) and
answer-correctness (rerank-veto) as **two separate axes** [SOURCED, arxiv 2607.08456] — which we
already do.

**Query understanding under budget (small-LM, HyDE, doc2query, translation).** The single most
important number here is **batch-1 decode latency on A6000**, and it is widely mis-cited: the
official Qwen2.5 speed sheet (0.5B 47 tok/s, 1.5B 40 tok/s) is **HuggingFace-eager**, ~15× below
the memory-bandwidth ceiling, and must not be used for a hot-path decision [SOURCED,
qwen.readthedocs.io]. On an **optimized engine** (llama.cpp/vLLM/TRT-LLM, int4, CUDA graphs) the
measured A6000 anchor is 138.7 tok/s for a 7B-Q4 [SOURCED, knightli.com], and batch-1 decode is
memory-bound (arXiv 2605.30571). Extrapolated [INFERRED from that anchor]: a **0.5B-Q4 doing ~30
tokens ≈ 43 ms, ~60 tokens ≈ ~86 ms**; **1.5B-Q4 ~30-45 tokens ≈ 75-110 ms**; a full 60-token 1.5B
rewrite (~150 ms) **overshoots** our headroom. Consequences:
- A tiny-LM **router** (decode 1-5 tokens) fits trivially (**<10-15 ms**) — but see §6, the
  cheapest robust router is not even a generative model, it's a **logistic / cosine-to-prototype
  head on the BGE-M3 query embedding we already compute (~0 ms marginal)** [SOURCED,
  Lightweight Query Routing arxiv 2604.03455; Adaptive-RAG NAACL 2024 arxiv 2403.14403; Wang et al.
  EMNLP 2024 arxiv 2407.01219, where a BERT retrieve-or-not router raised score 0.428→0.443 **and**
  cut latency 29%].
- A tiny-LM **rewriter** fits **only with a hard `max_new_tokens ≈ 24-48` cap on int4** — never
  HF-eager. Useful for W7 (session rewrite), optional elsewhere.
- **HyDE does NOT fit** a 200 ms budget: it generates a whole ~100-200-token hypothetical document
  before retrieval (~150-500 ms on A6000); the RAG best-practices paper measures HyDE at 2.8-4.3×
  end-to-end and **drops it from its efficiency config** [SOURCED, arxiv 2407.01219, 2212.10496].
  Gate it behind a low-confidence fallback, never default it.
- **doc2query-- is the one free lunch**: document expansion is **offline/index-time**, ~zero query
  latency, and with relevance filtering gives **+16% effectiveness, −23% query time, −33% index**
  [SOURCED, arxiv 2301.03266; docTTTTTquery]. It augments *documents*, so it needs a re-index — but
  that is offline and can land before freeze (see the opportunity note in §5).
- **Multilingual: native BGE-M3 cross-lingual dense beats query-translation-at-retrieval** for
  supported languages (MKQA 75.1 with no translation, one ~2-15 ms forward pass) and its sparse head
  gives exact-term matching for free; translation costs an extra MT decode (~75-150 ms) and only
  wins for terminology-heavy/exact-match or weak-alignment low-resource languages (the one
  translation-wins paper used a weak 2020 encoder + a tuned reranker vs zero-shot dense — confounded)
  [SOURCED, arxiv 2402.03216; NTCIR-18 2025]. This confirms the DO-NOT-DO entry (§7).

---

## 3. RANKED WEAKNESS LIST (deliverable 1)

Ranked by (impact × confidence ÷ cost). Each: what, evidence, expected accuracy gain, ms cost,
implementation days.

### W1 — Cross-lingual fusion is misconfigured: equal-weight RRF ships hybrid where dense-only wins **[RANK 1]**
- **What.** `config.use_sparse=True` + `retrieval._rrf` equal-weight (`rrf_k=60`) ships hybrid,
  yet our own paired A/B has **dense-only beating hybrid by +0.2145 recall@10** (0.7021 vs 0.4876),
  228/500 queries better, 19 worse [MEASURED, `eval/retrieval_metrics.md`].
- **Evidence.** BGE-M3 paper: cross-lingual sparse adds +0.2 R@100 but is near-noise alone (45.3);
  monolingual sparse adds +1.2 [SOURCED, 2402.03216]. BM25 cross-lingual catastrophic [SOURCED,
  2511.19324]. Equal-weight RRF gives a near-random sparse ranking the **same** rank-weight as a
  strong dense ranking; at **top-10** there is no room to absorb the injected junk, so fusion goes
  **negative**, not just flat — a config bug, not a law [INFERRED, grounded]. Weighted RRF overfits
  [SOURCED, 2210.11934]; CC/relative-score fusion with one learned α is the transferable alternative
  (+3-7%) [SOURCED].
- **Fix (pick per grid result):** (a) **dense-only** for cross-lingual — simplest, matches our win,
  **removes a Qdrant query**; (b) keep sparse only for **same-language** queries (route by
  detected-lang == corpus-lang), where +1.2 is real; (c) if keeping a fused arm anywhere, use
  **learned-α score fusion (TMM)**, never equal-weight RRF.
- **Gain:** up to **+0.21 recall@10** (must re-confirm the interaction *with* the cross-encoder —
  rerank may recover part of hybrid's first-stage deficit, so the net after-rerank delta is the
  number that ships). **ms: −3 to −5 ms** (one fewer arm). **Days: 0.5.** Confidence: **high
  direction, medium magnitude** (single run, no CI — the grid fixes this).

### W2 — Rerank pool is shallow and un-optimized while 110-148 ms of budget sits idle **[RANK 2]**
- **What.** `rerank_candidates=24`, 34 ms, only ~17% of budget; `max_length=512` (MSMARCO passages
  are ~50-100 tokens, so most of that window is unused headroom, not compute) and no ONNX/TensorRT.
- **Evidence.** Rerank is already our biggest lever (+0.2924 recall@8) [MEASURED]. Cross-encoder
  latency is ~linear in candidate count; **top-24→top-64/100 with bge-m3 stays under 200 ms
  (~120-140 ms est.)** and directly rescues first-stage recall misses [SOURCED/INFERRED, arxiv
  2509.25085 + aimultiple anchors]. **ONNX/TensorRT fp16 ~1.5× / int8 ~1.8×** → top-24 in ~19-23 ms
  or deeper rerank at constant latency [SOURCED, sbert.net; nvidia].
- **Fix.** Grid `rerank_candidates ∈ {8,16,24,32,48,64}`; export bge-reranker to ONNX/TensorRT
  fp16; pick the depth that maximizes recall@8 subject to A6000 P95 < 200 ms.
- **Gain:** **+0.02-0.05 recall@8** (est., deeper pool) — bounded because rerank already recovers
  most of first-stage recall. **ms: ~0 with fp16; +20-40 ms if pure depth.** **Days: 1-2.**
  Confidence: **high**.

### W3 — Grounding gate is lexical; SOTA is NLI entailment (and it fits the budget) **[RANK 3]**
- **What.** `grounding_check` uses lexical support ≥0.5 + a negation guard; NLI is wired but
  **only in quality mode** (`nli` unset on the extractive path).
- **Evidence.** Lexical overlap is precisely the failure the faithfulness literature warns about
  (**55-57% false grounding** on keyword-sharing distractors) [SOURCED, arxiv 2412.18004]; NLI is
  the standard AutoAIS verifier and also handles negation/paraphrase properly [SOURCED]. Because our
  answers are **verbatim**, fabrication is already impossible — the residual risk is
  relevance/negation mismatch, exactly NLI's strength [SOURCED, arxiv 2605.21102].
- **Fix.** Add a distilled **multilingual NLI cross-encoder** (e.g. mDeBERTa-v3-base-xnli class) as
  the extractive-path grounding decision over the ~5-8 cited (premise=context, hypothesis=answer)
  pairs.
- **Gain:** fewer false-grounded answers + better abstain precision — a **guardrail-quality** win on
  the exact axis the brief scores ("knows when NOT to answer"). **ms: +8-12 ms** (a cross-encoder
  over ~5-8 pairs; scaled from our measured 34 ms/24-pair reranker) [INFERRED]. **Days: 1-2.**
  Confidence: **high**.

### W4 — Abstain threshold is a single F1-point from 55 queries; SOTA is risk-coverage / conformal **[RANK 4]**
- **What.** τ=0.5852 is F1-optimal over 40 in-domain + 15 OOD [MEASURED, `eval/calibrate_ood_live.py`],
  a single global point; the loop even picks the **largest** F1-optimal τ (conservative → over-abstains).
- **Evidence.** Selective-QA best practice: report **risk-coverage + AURC/AUGRC**, set the cutoff by
  **split conformal** for a distribution-free guarantee at O(1) inference; CONFLARE does this for a
  retrieval-similarity gate [SOURCED, arxiv 2407.01032, 2404.04287]. Temperature scaling won't help
  discrimination [SOURCED, 2402.05806]. Kamath: a proper calibrator answers 56% @80% acc vs 48%
  raw under shift [SOURCED, arxiv 2006.09462].
- **Fix.** Grow the labeled calib set (≥150-300 in/OOD, per-language buckets); compute the RC curve;
  set τ **per detected language** by target risk via split conformal; report AURC/AUGRC. Keep the
  rerank-veto as the separate answer-correctness axis.
- **Gain:** principled coverage@risk; likely **+coverage at equal risk** (relaxes over-abstention).
  **ms: ~0** (precomputed-quantile compare). **Days: 1-2.** Confidence: **high (method)**.

### W5 — No ColBERT/late-interaction arm (BGE-M3's third output is switched off) **[RANK 5, optional]**
- **What.** `return_colbert_vecs=False`; the multi-vector signal BGE-M3 already computes is discarded.
- **Evidence.** ~**+1 nDCG@10** over dense+sparse on MIRACL; **MaxSim ~2 ms/top-100**; storage
  **~2-5 GB at our corpus size** (trivial); Qdrant 1.10 native multivector [SOURCED, arxiv
  2402.03216, 2605.29517, qdrant.tech]. **But** the lift is small and **likely redundant on top of a
  strong cross-encoder** that already does full query-doc attention [INFERRED].
- **Fix (only if the grid shows lift):** re-index with `return_colbert_vecs=True` (token-pool ×2 +
  int8), Qdrant multivector rerank-arm, fusion dense 1.0/sparse 0.3/colbert 1.0.
- **Gain:** **~+1 nDCG@10** at best, possibly ~0 on top of the cross-encoder. **ms: +2-5 ms
  (MaxSim), offline re-embed.** **Days: 2-3.** Confidence: **medium (small, possibly redundant)**.

### W6 — Adaptive retrieval depth is keyed only on ASR confidence, not query difficulty **[RANK 6, optional]**
- **What.** `normalize.confidence_params` widens k / relaxes OOD by `asr_confidence` — a good hook,
  but the **query-difficulty** signal (dense margin, retriever divergence) is unused for depth.
- **Evidence.** Cheap no-LLM weakness signals (AUC 0.73-0.77) route easy→shallow, hard→deep; the
  cascade pattern captures long-tail wins without regressing easy queries [SOURCED, qdrant.tech;
  arxiv]. Caveat: fires eagerly → escalation only [SOURCED, 2606.13814].
- **Fix.** §6 config-routing: after first-stage, branch rerank depth on `ood.top_score` margin +
  divergence. **Gain:** latency saved on easy queries (more headroom) + recall on hard. **ms:
  net-neutral/savings. Days: 2-3.** Confidence: **medium**.

### W7 — Multi-turn "session rewrite" is naive string concatenation **[RANK 7, optional]**
- **What.** `SessionContext.rewrite` returns `"{prev_q} {q}"` — no coreference resolution, no
  ellipsis handling; can dilute the current query's embedding.
- **Fix.** Better heuristic (append only prev **nouns/entities**) or a tiny-LM rewrite. **Gain:**
  better follow-up turns (only if multi-turn is demoed/scored). **ms:** heuristic ~0; a tiny-LM
  rewrite fits **only** as a **0.5B-Q4 with `max_new_tokens ≈ 24-48` on an optimized engine
  (~43-86 ms)** — never HF-eager (~4-17× slower) [SOURCED, §2 small-LM]. **Days: 1** (heuristic)
  **/ 2-3** (tiny-LM). Confidence: **medium; scope-dependent** — do the heuristic first; only add
  the tiny-LM if multi-turn is actually scored.

### W8 — RAPTOR is the live collection but its retrieval edge over `passage` is within noise **[RANK 8, simplify]**
- **What.** Live collection = `msmarco_xi__raptor` (leaves+summaries), adding ~4.8k summary vectors
  and a `leaf_expand` stage (extra latency + a failure surface), yet **raptor 0.3824 ≈ passage
  0.3859 nDCG@10** [MEASURED].
- **Fix.** Consider promoting **`passage`** as the live collection and keeping RAPTOR strictly for
  the summary-navigation / PageIndex path. **Gain:** ~0 accuracy, **simpler + faster + fewer failure
  modes.** **ms: −(leaf_expand). Days: 0.5.** Confidence: **medium** (verify no recall regression on
  broad queries in the grid).

### W9 — Cache τ=0.92 untuned; answer-only store causes the Gate-D measurement artifact **[RANK 9, minor]**
- **What.** `cache_similarity_threshold=0.92` never swept; answer-only caching skews the under-load
  answer-rate metric (documented, not a real regression) [MEASURED, `eval/stress_report.md`].
- **Fix.** Sweep τ ∈ {0.88,0.92,0.95,0.97} for false-hit rate vs hit-rate; document the cache-slice
  metric so it isn't misread. **Gain:** avoids a wrong-answer-from-cache tail. **ms: 0. Days: 0.5.**

### W10 — (folds into W1) If any same-language slice keeps sparse, use learned-α score fusion, not RRF
- CC/relative-score fusion (+3-7%, single learned α, transfers) beats RRF where both arms are
  calibrated [SOURCED, 2210.11934, weaviate.io]. Only relevant if a monolingual slice retains sparse.

---

## 4. ABLATION + TUNING GRID (deliverable 2)

Run on the **500-query calibration split** (the existing `eval/eval_chunking.evaluate_strategies`
+ `eval/eval_retrieval_modes` harness, real BGE-M3 + BGEReranker + live Qdrant), then freeze and do
**one** sealed 40k run via the already-built `eval/sealed_test.py` (mechanically once-only, records
git SHA + collection hashes + seeds; outcomes never feed tuning).

### 4.1 Factors and levels (parameter → real config field)

| Factor | Config field | Levels | Primary? |
|---|---|---|---|
| **Fusion arm** | `use_sparse` + new `fusion_mode` | `dense_only`, `rrf60_equal` (current), `rrf_weighted{w_sparse∈0.3,0.5}`, `cc_alpha_TMM{α learned}` | **P0** |
| **RRF k** | `Retriever.rrf_k` | {10, 30, 60, 100} (only within hybrid arms) | P1 |
| **Rerank depth** | `rerank_candidates` | {8, 16, 24, 32, 48, 64} | **P0** |
| **Answer depth** | `top_k` / eff_k (top_n) | {5, 8, 10} | P1 |
| **Reranker runtime** | new `rerank_backend` | `hf_fp16` (current), `onnx_fp16`, `trt_int8` | P1 (latency) |
| **ColBERT arm** | `return_colbert_vecs` + Qdrant multivector | {off (current), on: dense1/sparse0.3/colbert1} | **P0 (go/no-go)** |
| **Rerank admit** | `rerank_min` | {0.20, 0.35, 0.50} | P1 |
| **Rerank veto** | `rerank_veto` | {0.0, 0.03, 0.06} | P1 |
| **OOD τ** | `ood_score_threshold` | RC-curve sweep + **per-language**; conformal α∈{0.05,0.10,0.15} | **P0 (guardrail)** |
| **OOD relax** | `normalize.confidence_params` | {current, ±0.03} | P2 |
| **Cache τ** | `cache_similarity_threshold` | {0.88, 0.92, 0.95, 0.97} | P2 |
| **Composer floor** | `support_floor` | {0.20, 0.26, 0.35} | P1 |
| **Composer novelty** | `novelty_lambda` | {0.40, 0.55, 0.70} | P1 |
| **Composer support** | `max_support` | {3, 5} | P2 |
| **Answer length** | `max_len_chars` | {900, 1300} | P2 |
| **Grounding** | new `nli` on extractive path | {lexical (current), +mDeBERTa-NLI} | **P0 (guardrail)** |
| **Live collection** | `live_strategy` | {`raptor` (current), `passage`} | P1 |
| **Tiny-LM router** | new `use_query_rewriter` | {off, on} | P2 (see §2, §6) |

### 4.2 The interaction cells that MUST be run (not one-factor-at-a-time)
1. **`fusion_mode × rerank_candidates`** — the decisive untested cell. Our rerank ablation used a
   **hybrid** first-stage pool; if dense-only is +0.21 better *before* rerank, `dense_only × {24,48}`
   is the config most likely to be the new SOTA. **This is the single most important run.**
2. **`ColBERT_arm × rerank_candidates`** — does the colbert arm add anything *on top of* the
   cross-encoder, or is it redundant? (go/no-go for W5.)
3. **`OOD τ × language`** — per-language RC curves; en vs hi may want different cutoffs.
4. **`grounding(lexical vs NLI) × abstain-decision`** — measured on a labeled faithful/unfaithful set
   (inject keyword-sharing distractors per Wallat) to catch the 55-57% false-grounding mode.

### 4.3 Measurement protocol (paired, seeded, CIs)
- **Paired, per-query.** Every arm scores the **same 500 queries**; report **Δ + win/loss/tie**
  (as `eval/retrieval_metrics.md` already does), not just means.
- **Seeded.** Fix seeds for sampling, any k-means (RAPTOR), and cache state; record them in the
  sealed-run metadata (`sealed_test.build_metadata` already captures seeds + git SHA + collection
  hashes + GPU).
- **Confidence intervals.** **Bootstrap 95% CI** (≥1000 resamples over the paired per-query deltas)
  for every headline metric — the calibration reports explicitly lack CIs; this closes that gap.
- **Significance.** Paired **bootstrap** or **Wilcoxon signed-rank** on per-query deltas; treat a
  result as real only if the CI excludes 0.
- **Primary metrics.** **recall@8** (matches deployed eff_k) and **nDCG@10**; distinct-doc credit
  (already implemented, caps nDCG≤1). **Guardrail metrics:** answer-rate, **OOD F1 + AURC/AUGRC**,
  grounding false-accept rate on the distractor set. **Latency gate:** re-measure P50/P95 on A6000
  per config; **reject any config with P95 ≥ 200 ms** (`eval/latency.py` already emits P50/P70/P100).
- **Multiple-comparisons hygiene.** The grid is large; use the calib split to **select**, then the
  **sealed 40k** run (once) to **confirm** the frozen config. Never tune on the sealed outcome
  (enforced by `sealed_test.py`).

---

## 5. Expected end-state after the grid
- Ship **dense-only (or same-language-gated fusion) × rerank depth 32-48 × ONNX-fp16 reranker ×
  NLI grounding × per-language conformal τ**. Projected: **recall@8 ~0.75→~0.78-0.80** [INFERRED
  from the +0.21 first-stage headroom partially absorbed by rerank + deeper pool], guardrail
  precision up, **P95 still < 200 ms** (fp16 reranker buys back the NLI + depth cost). ColBERT arm
  and tiny-LM router ship **only if** their grid cells clear their CI.

**Offline opportunity (near-free, before freeze): doc2query-- document expansion.** The only
query-understanding technique with **~zero query-time cost** — it augments *documents* at index
time. With relevance filtering it gives **+16% effectiveness, −23% query time, −33% index size**
[SOURCED, arxiv 2301.03266]. It needs a re-index (offline, ~the RAPTOR-build cost) and would
strengthen the sparse/lexical arm in the *same-language* slice where we keep it. Worth a single
offline A/B against the current `passage` index if there is build time before 08-20; strictly
additive, no hot-path latency. [INFERRED value; SOURCED technique]

---

## 6. CONFIG-ROUTING design (deliverable 3)

**Goal:** a per-query adaptive policy that spends the 110-148 ms headroom where it helps (hard
queries) and saves it where it doesn't (easy queries), using signals **already in the trace** — no
new model on the hot path. Cheapest robust implementation = **a threshold policy over existing
signals**, not a learned router.

### 6.1 What to key on (all already available)
**Pre-retrieval (free):**
- **Query token length** (`len(clean.split())`) — short factoid vs long descriptive.
- **Language relation** — `router.detect_language(query)` vs corpus language ⇒ **cross-lingual flag**
  (Latin→"en" default already captures the Hindi-query-English-corpus case). Drives the fusion arm
  (W1): cross-lingual ⇒ dense-only; same-language ⇒ allow sparse.
- **Intent** (`route.intent`) — chitchat/meta already short-circuit; qa proceeds.
- **`asr_confidence`** — already widens k / relaxes OOD (`normalize.confidence_params`). Keep.

**Mid-flight (≤1 extra Qdrant query or free from the first stage):**
- **Dense top-1 margin** (`ood.top_score`) — high margin ⇒ easy ⇒ shallow rerank; low margin ⇒ hard.
- **Retriever divergence / dense variance** — Qdrant's no-LLM weakness signals (AUC 0.73-0.77)
  [SOURCED, qdrant.tech]. Reuse the query vector; sub-ms.
- **`rerank_top`** — post-rerank confidence, already the answerability signal.

### 6.2 The policy (cheap, robust)
```
route(query):
  lang, intent = detect_language, HeuristicRouter        # existing
  if intent in {chitchat, meta}: small_talk              # existing
  cross_lingual = (lang != corpus_lang)
  fusion = dense_only if cross_lingual else same_lang_fusion   # W1
  k, ood_relax = confidence_params(asr_confidence)             # existing hook

  hits = retrieve(fusion, pool=base_pool)                # base_pool = 24
  margin = max_dense_cosine(hits); div = dense_sparse_divergence(hits)
  # difficulty branch (new): easy -> shallow, hard -> deep
  if margin >= τ_easy and div low:      depth = 16       # easy: save ~10-15 ms
  elif margin < τ_hard or div high:     depth = 48       # hard: spend headroom
  else:                                  depth = 32
  reranked = rerank(hits[:depth]) ; gate + compose + NLI-ground   # W2/W3
```
- **τ_easy / τ_hard** are set on the calib split by the **Youden point** of the weakness signals
  (per Qdrant's recipe), per language.
- **Escalation tier (optional, off the hot path):** if `rerank_top` and margin are both very low,
  flag the query for the **quality mode** (7B) or a CRAG-style corrective retry — **asynchronously**,
  returning the honest abstain now. Never block the 200 ms answer on it.
- **Implementation cost:** ~2-3 days. It is additive to `harness._answer_admitted` (one new
  post-first-stage branch) and reuses `normalize.confidence_params`. No new model, no new hot-path
  latency beyond one optional Qdrant query already budgeted.

### 6.3 The router substrate: a free logistic head on the query embedding — not a generative LM
The 2026 routing literature is explicit that the **cheapest robust router is not an LLM call**: a
**logistic-regression / cosine-to-prototype head over the query embedding you already compute for
dense retrieval** is the most economical robust option, ~**0 ms marginal**, and it beats putting a
generative model on the hot path [SOURCED, Lightweight Query Routing arxiv 2604.03455; Adaptive-RAG
NAACL 2024 arxiv 2403.14403]. So the recommended implementation of §6.2 is: keep the transparent
threshold policy for the depth branch (debuggable, per-language), and — if a single scalar margin
proves too noisy — train a **tiny logistic head on the BGE-M3 query vector** (labels = "easy/hard"
from the calib split's per-query rerank outcome) rather than adding any generative router. Reserve a
generative 0.5B only for a capped **rewrite** (W7), never for routing. The literature's caution that
retrieval-side gates fire eagerly [SOURCED, 2606.13814] is why the escalation tier is async and the
default answer still returns in-budget.

---

## 7. DO-NOT-DO list (deliverable 4) — SOTA-paper ideas that do NOT fit 200 ms / one A6000 / 4 days

| Idea | Why it's tempting | Why NOT here | Source |
|---|---|---|---|
| **HyDE** (hypothetical-doc embeddings) | strong zero-shot recall gains | requires a full LLM **generation of a ~100-200-token doc before retrieval** — ~150-500 ms on A6000; the RAG best-practices paper measures 2.8-4.3× and drops it from its efficiency config | arxiv 2212.10496; 2407.01219 |
| **RAG-Fusion** (multi-query) | catches long-tail phrasings | extra generative LLM call, **~1.7× slower**, in the retrieval path | arxiv 2402.03367 |
| **Self-RAG** reflection tokens | best selective-gen numbers | the reflection loop **IS** the generator (fine-tuned LM + segment beam search); not a side-gate | arxiv 2310.11511 |
| **CRAG corrective action** | fixes weak retrieval | action = **web search + decompose-recompose** (external I/O + LLM); only the *evaluator* is extractable | arxiv 2401.15884 |
| **bge-reranker-v2.5-gemma2-lightweight** | MIRACL 77.3 (best accuracy) | **9B LLM reranker**; even at −60% FLOPs, too heavy for top-24+ under 200 ms on one A6000 | huggingface BAAI card |
| **Cohere Rerank v3.5 / any API reranker** | SOTA-claimed, easy | **API-only** → violates "no server in the hot path / client-side" + adds network RTT | docs.cohere.com |
| **mxbai-rerank-v2** | strong English BEIR | **weak multilingual (MIRACL 57.94)** — a downgrade for an Indic pipeline | arxiv 2509.25085 |
| **ColBERT as FIRST-stage retriever** | ~100× faster than cross-encoder | storage/ops tax at scale + **redundant with our cross-encoder**; only worth it as an optional rerank arm to *test* (W5) | qdrant.tech; arxiv 2205.09707 |
| **DPP log-det answer selection** | small factuality win over MMR | **~O(n³)**; offline/quality-only, not the hot path | arxiv 2608.03655 |
| **Temperature / Platt scaling for the gate** | "calibrate the threshold" | **monotonic → cannot change abstain ordering or AURC**; only relabels 0.585 as a probability | arxiv 2402.05806 |
| **Query-translation-at-retrieval** for BGE-M3-supported langs | intuitive for cross-lingual | native BGE-M3 cross-lingual dense already wins for supported langs (MKQA 75.1, one forward pass); MT adds a ~75-150 ms decode + error. Reserve only for terminology-heavy/low-resource-alignment cases | arxiv 2402.03216; NTCIR-18 2025 |
| **Three-way (dense+sparse+colbert) equal-weight fusion** | "use all signals" | our cross-lingual data says sparse **hurts** at equal weight (W1); only weighted/gated fusion is safe | arxiv 2402.03216 + [MEASURED] |

---

## 8. Source appendix (primary URLs)

**Fusion / cross-lingual:** BGE-M3 paper https://arxiv.org/abs/2402.03216 (·v5
https://arxiv.org/html/2402.03216v5) · Bruch/Pinecone fusion (CC>RRF, weighted-RRF overfit)
https://arxiv.org/html/2210.11934v2 · cross-lingual ranking / BM25 collapse
https://arxiv.org/abs/2511.19324 · Elastic RRF k=60
https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion · Weaviate
fusion https://weaviate.io/blog/hybrid-search-fusion-algorithms

**ColBERT / multivector:** PLAID https://arxiv.org/abs/2205.09707 · Flash-MaxSim (2026)
https://arxiv.org/html/2605.29517v1 · Qdrant 1.10 multivector
https://qdrant.tech/blog/qdrant-1.10.x/ + tutorial
https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/ ·
Answer.AI token pooling https://www.answer.ai/posts/colbert-pooling.html · ColBERT-serve
https://arxiv.org/abs/2504.14903 · Arctic-Embed 2.0 (MIRACL/CLEF table)
https://arxiv.org/abs/2412.04506

**Rerankers:** jina-reranker-v3 paper (standings table) https://arxiv.org/html/2509.25085v1 ·
bge-reranker-v2-m3 card https://huggingface.co/BAAI/bge-reranker-v2-m3 · jina-reranker-v2
https://jina.ai/news/jina-reranker-v2-for-agentic-rag-ultra-fast-multilingual-function-calling-and-code-search/
· mxbai-rerank-v2 https://www.mixedbread.com/blog/mxbai-rerank-v2 · sbert efficiency (ONNX/OpenVINO)
https://sbert.net/docs/cross_encoder/usage/efficiency.html · NVIDIA TensorRT int8
https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/
· Redis reranking guide https://redis.io/blog/top-reranking-models-rag-accuracy/ · aimultiple
reranker latency https://aimultiple.com/rerankers

**Corrective / control loops:** CRAG https://arxiv.org/html/2401.15884v2 · Self-RAG
https://arxiv.org/abs/2310.11511 · RAG-Fusion https://arxiv.org/html/2402.03367v2 · Qdrant
"Predicting Weak Retrieval Without an LLM" https://qdrant.tech/articles/predicting-weak-retrieval/
· TASR (eager-firing caveat) https://arxiv.org/pdf/2606.13814

**Extractive composition / citations:** VerbatimRAG https://arxiv.org/abs/2605.21102 · LongCite
https://arxiv.org/abs/2409.02897 · citation-faithfulness (55-57% false)
https://arxiv.org/abs/2412.18004 · MMR/DL-MMR https://arxiv.org/pdf/2503.09249 · DPP log-det
faithful summarization https://arxiv.org/html/2608.03655 · submodular (Lin & Bilmes)
https://aclanthology.org/P13-1100.pdf · ALCE https://arxiv.org/abs/2305.14627

**Calibration / selective-QA:** AURC→AUGRC (NeurIPS 2024) https://arxiv.org/abs/2407.01032 ·
selective-QA / AURC survey https://arxiv.org/abs/2410.15361 · split conformal / TRAQ
https://arxiv.org/abs/2307.04642 · CONFLARE https://arxiv.org/abs/2404.04287 · Conformal Abstention
https://arxiv.org/abs/2405.01563 · temperature-scaling monotonicity https://arxiv.org/html/2402.05806
· Kamath selective-QA https://arxiv.org/abs/2006.09462 · SQuAD 2.0 https://arxiv.org/pdf/1806.03822 ·
two abstention axes https://arxiv.org/pdf/2607.08456 · semantic-entropy probes
https://arxiv.org/pdf/2406.15927

**Query understanding / small-LM / routing:** HyDE https://arxiv.org/abs/2212.10496 · Best
Practices in RAG (HyDE latency, retrieve-or-not router) https://aclanthology.org/2024.emnlp-main.981/
(arxiv 2407.01219) · docTTTTTquery https://github.com/castorini/docTTTTTquery · Doc2Query--
https://arxiv.org/abs/2301.03266 · Adaptive-RAG (NAACL 2024) https://arxiv.org/abs/2403.14403 ·
Lightweight Query Routing / RAGRouter-Bench (2026) https://arxiv.org/abs/2604.03455 · Qwen2.5 speed
sheet (HF-eager caveat) https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html ·
batch-1 decode roofline https://arxiv.org/html/2605.30571v1 · llama.cpp A6000 anchor
https://knightli.com/en/2026/04/23/llama-cpp-gpu-benchmark-cuda-rocm-vulkan-scoreboard/ ·
translation-vs-multilingual (NTCIR-18, Jun 2025) https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings18/pdf/ai_cup/03-AICUP-AICUP-ChiuY.pdf

**Landscape:** hybrid/BM25/rerank 2026 reference https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026
· advanced RAG techniques 2026 https://atlan.com/know/advanced-rag-techniques/

_All five research clusters (ColBERT/multivector · fusion/advanced-RAG · rerankers ·
extractive/calibration · small-LM/query-understanding) are integrated above. Claims are tagged
[SOURCED]/[MEASURED]/[INFERRED]; A6000 small-model tok/s and any "days" estimates are [INFERRED] and
should be validated empirically (e.g. `llama-bench`/vLLM on the actual build) before locking the
freeze plan._
