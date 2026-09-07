# Retrieval metrics — live 20k index (calibration-side)

_Generated 2026-08-15 19:51 UTC by `eval/run_chunking_eval_live.py` on Forge (host `goquest-Z790-AORUS-ELITE-AX`, GPU NVIDIA RTX A6000), against the live Qdrant server `http://localhost:6333` with the real BGE-M3 embedder and real MSMARCO-XI qrels._

## What this is (and is not)

- **Real qrels.** Labels are MSMARCO-XI `is_selected==1` passages — the dataset's own relevance judgements, not synthetic. Cross-lingual setting: the query is the Hindi `query` field; relevant passages are the selected English/translated passages that were actually indexed.
- **Standard IR metrics.** recall@k / MRR / nDCG@k, distinct-doc credit (a relevant doc gains once at its first rank, so fine-grained chunkers can't inflate past 1.0).
- **Honest denominator.** recall denominator = every `is_selected` positive for the query; the pre-score assertion (below) confirms those positives resolve in the index.
- **Calibration-side, single run, no CIs.** This is the interim **20k** index, one run, one box. **No confidence intervals** — bootstrap CIs come with the sealed **40k** eval (`eval/sealed_test.py`, untouched here). Numbers are point estimates.

## Pre-score assertion — qrel/index alignment

The eval set was rebuilt with `index_build.build_eval_set(languages=('hi',), max_docs=20000, include_english=True, include_translated=True)` — the **same `_load_msmarco` code path** (via `load_msmarco_slice`) that `build_real_index.py` used to build the live `msmarco_xi__*` collections on 2026-08-15, so the sequential `{lang}-N` doc_ids line up by construction. Verified empirically before scoring:

- **Sampled (20 random qrel doc_ids):** 20/20 = **100%** resolve as payload `doc_id` in `msmarco_xi__passage` (gate: ≥90%). PASS.
- **Full (all 1066 unique positive doc_ids):** **100.00%** resolve in `msmarco_xi__passage`.

Per-collection resolution of the positive doc_ids (justifies the recall denominator on each collection):

| collection | points | distinct doc_ids | positives resolved |
|---|---|---|---|
| msmarco_xi__fixed | 22281 | — | 100.00% |
| msmarco_xi__recursive | 22482 | — | 100.00% |
| msmarco_xi__sentwin | 33296 | — | 100.00% |
| msmarco_xi__passage | 21164 | — | 100.00% |
| msmarco_xi__hierarchical | 49668 | — | 100.00% |
| msmarco_xi__raptor | 25964 | — | 100.00% |

- Eval corpus rebuilt: **20000** docs, **500** candidate queries, **500** with ≥1 positive (scored), **1066** unique positive doc_ids, **1066** total (query, positive) pairs (avg **2.13** positives/query).

## 1. Chunking strategy A/B

Real BGE-M3 hybrid retrieval (dense+sparse RRF) via `eval_chunking.evaluate_strategies`, top_k=10, N=500 queries. Sorted by nDCG@10.

| strategy | n_chunks | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| passage | 21164 | 0.4876 | 0.5057 | 0.3859 |
| raptor | 25964 | 0.4836 | 0.5019 | 0.3824 |
| recursive | 22482 | 0.4743 | 0.5061 | 0.3815 |
| fixed | 22281 | 0.4768 | 0.5034 | 0.3809 |
| sentwin | 33296 | 0.4596 | 0.4903 | 0.3683 |
| hierarchical | 49668 | 0.3992 | 0.4507 | 0.3271 |

> **raptor** caveat: `msmarco_xi__raptor` mixes 25964 points (~21k leaves + ~4.8k summaries). This raw A/B scores `retriever.search` output directly, so summary hits occupy top slots and are never credited (their doc_ids aren't leaf positives) — the LIVE harness applies leaf-expansion (summaries navigate, leaves are evidence) which this strategy-level A/B deliberately does not. Read raptor here as 'raw tree-collection retrieval', not the served path.

## 2. Ablation (a) — cross-encoder rerank on/off (passage)

Paired, same N=500 queries. First stage: hybrid top-24 pool on `msmarco_xi__passage` (deployed `rerank_candidates=24`). Rerank arm: `src/reranker.BGEReranker` (BAAI/bge-reranker-v2-m3) re-scores the pool and keeps top-k. no-rerank arm: first-stage RRF order, top-k. k=8 is the deployed `eff_k`.

| k | recall no-rr | recall +rerank | Δrecall | MRR no-rr | MRR +rerank | ΔMRR | win/loss/tie |
|---|---|---|---|---|---|---|---|
| 8 | 0.4331 | 0.7255 | +0.2924 | 0.5021 | 0.6098 | +0.1077 | 297/9/194 |
| 10 | 0.4876 | 0.7624 | +0.2748 | 0.5057 | 0.6124 | +0.1067 | 278/4/218 |

win/loss/tie = per-query recall of (+rerank) vs (no-rerank). Single run, no CI.

## 3. Ablation (b) — hybrid dense+sparse vs dense-only (passage)

Paired, same N=500 queries, `msmarco_xi__passage`, top_k=10. Dense-only = `Retriever(use_sparse=False)` (dense arm only); hybrid = dense+sparse RRF.

| arm | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| hybrid (dense+sparse) | 0.4876 | 0.5057 | 0.3859 |
| dense-only | 0.7021 | 0.5307 | 0.5093 |

Δrecall@10 (hybrid − dense-only) = **-0.2145**; hybrid win/loss/tie vs dense-only = **19/228/253**. Single run, no CI.

## 4. Ablation (c) — PageIndex vectorless vs hybrid

Paired, same N=500 queries, top_k=10. PageIndex = `PageIndexTree.from_qdrant(msmarco_xi__raptor)` — the current inverted-index (idf-weighted postings over leaves) scorer, **zero embeddings at query time**; tree = 25964 nodes, 169 roots, 21164 leaves. Hybrid = BGE-M3 dense+sparse on `msmarco_xi__passage`.

| arm | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| pageindex (vectorless) | 0.1589 | 0.1364 | 0.1084 |
| hybrid (dense+sparse) | 0.4876 | — | — |

Δrecall@10 (hybrid − pageindex) = **+0.3288**; hybrid win/loss/tie vs pageindex = **295/4/201**. PageIndex is offered as an additional vectorless mode (per-query toggle), not the default — it trades recall for no query-time embeddings. Single run, no CI.

## Honest summary

**Measured:** real MSMARCO-XI qrels (is_selected), standard recall@k/MRR/nDCG@k with distinct-doc credit, on the live 20k Qdrant index with the real BGE-M3 embedder and the real BGEReranker; three paired ablations (rerank, hybrid vs dense, pageindex vs hybrid); explicit qrel/index alignment assertion before any score.

**NOT measured / caveats:** no confidence intervals (single run — CIs come with the sealed 40k eval); calibration-side on the **interim 20k** index, not the sealed 40k; single box, single run; cross-lingual Hindi→English/Indic setting only (`languages=('hi',)`); raptor row is raw tree-collection retrieval, not the leaf-expanded served path; PageIndex scored on raptor-tree leaves while hybrid is scored on the passage collection (same underlying passages).

_Wall time: 1.5 min._
