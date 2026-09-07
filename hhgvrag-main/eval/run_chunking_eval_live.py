# -*- coding: utf-8 -*-
"""
run_chunking_eval_live.py — RESEARCH-GRADE retrieval metrics on the LIVE 20k Qdrant index.

Runs ON the Forge GPU box against the live Qdrant server (localhost:6333) with the REAL
BGE-M3 embedder and REAL MSMARCO-XI qrels (is_selected labels). Calibration-side evidence on
the 20k interim index — NOT the sealed 40k eval (that lives in eval/sealed_test.py, untouched).

Produces eval/retrieval_metrics.md:
  1. Strategy A/B table (recall@10, MRR, nDCG@10, N) over all 6 collections via
     eval_chunking.evaluate_strategies with the real BGE Retriever.
  2. Ablation (a): passage WITH cross-encoder rerank vs WITHOUT (BGEReranker on top-24 pool),
     paired on the same queries, at k=8 (deployed eff_k) and k=10.
  3. Ablation (b): hybrid dense+sparse vs dense-only (Retriever use_sparse flag), paired.
  4. Ablation (c): PageIndexTree vectorless (from msmarco_xi__raptor) vs hybrid, paired.

CRITICAL — the pre-score assertion. build_eval_set params MUST match the live build
(languages=("hi",), max_docs=20000, include_english=True, include_translated=True — the
build_real_index.py / load_msmarco_slice path that produced msmarco_xi__* on 2026-08-15). The
script rebuilds the eval set, then verifies qrel doc_ids resolve as payload doc_id in
msmarco_xi__passage. If the sampled resolution is <90% it STOPS and reports the divergence
instead of publishing false recall.

Single run, no confidence intervals — bootstrap CIs come with the sealed 40k eval. Stated
honestly in the report.

Run (on Forge, in the GPU venv):
    ~/anaconda3/envs/hhgvrag/bin/python eval/run_chunking_eval_live.py \
        --qdrant-url http://localhost:6333
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)  # so eval_chunking resolves regardless of invocation cwd

import eval_chunking  # noqa: E402  (same dir)
import index_build  # noqa: E402  (src)

STRATS = ("fixed", "recursive", "sentwin", "passage", "hierarchical", "raptor")
PASSAGE = "msmarco_xi__passage"
RAPTOR = "msmarco_xi__raptor"


# ---- percentiles / scoring helpers (match eval_chunking's distinct-doc logic exactly) ----
def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score_hits(hits, rel, k):
    """Return (recall, reciprocal_rank, ndcg) for one query.

    Distinct-doc credit: a relevant doc gains once, at its first rank — identical to
    eval_chunking.evaluate_strategies, so a fine-grained chunker that returns many chunks of one
    doc cannot inflate the metric past 1.0."""
    seen, gains, first_rank = set(), [], None
    for i, h in enumerate(hits[:k]):
        did = h.payload.get("doc_id")
        if did in rel and did not in seen:
            seen.add(did)
            gains.append(1.0)
            first_rank = first_rank or (i + 1)
        else:
            gains.append(0.0)
    recall = len(seen) / len(rel)
    rr = 1.0 / first_rank if first_rank else 0.0
    ideal = _dcg([1.0] * min(len(rel), k)) or 1.0
    ndcg = min(_dcg(gains) / ideal, 1.0)
    return recall, rr, ndcg


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def paired(arm_a, arm_b):
    """Win/loss/tie counts for a paired metric list (per-query arm_b - arm_a)."""
    win = sum(1 for a, b in zip(arm_a, arm_b) if b > a + 1e-9)
    loss = sum(1 for a, b in zip(arm_a, arm_b) if b < a - 1e-9)
    tie = len(arm_a) - win - loss
    return win, loss, tie


# ---- memoized embedder: embed each query ONCE, reuse across strategies + ablations --------
class MemoEmbedder:
    """Wrap the real BGE-M3 embedder and cache embed_query by text. evaluate_strategies calls
    search(coll, q) once per (strategy, query); memoizing the query embedding keeps that to one
    BGE-M3 forward per unique query (and spares the shared live GPU)."""

    def __init__(self, inner):
        self._inner = inner
        self.dim = inner.dim
        self._cache = {}

    def embed_docs(self, texts):
        return self._inner.embed_docs(texts)

    def embed_query(self, text):
        er = self._cache.get(text)
        if er is None:
            er = self._inner.embed_query(text)
            self._cache[text] = er
        return er

    def get_offset_tokenizer(self):
        return self._inner.get_offset_tokenizer()


def collect_doc_ids(client, coll):
    """Full payload-only scroll of one collection -> set of doc_id. No payload index needed,
    and it yields the EXACT resolution denominator (not just a sample)."""
    ids = set()
    offset = None
    while True:
        try:
            pts, offset = client.scroll(coll, limit=2048, offset=offset,
                                        with_payload=["doc_id"], with_vectors=False)
        except Exception:
            # older client: fall back to full payload
            pts, offset = client.scroll(coll, limit=2048, offset=offset,
                                        with_payload=True, with_vectors=False)
        for p in pts:
            d = (p.payload or {}).get("doc_id")
            if d:
                ids.add(d)
        if offset is None:
            break
    return ids


def score_collection(retriever, coll, queries, qrels, k):
    """recall@k / MRR / nDCG@k over a collection, using the real hybrid retriever. Returns
    (metrics dict, per-query recall list) so ablations can compute paired win/loss/tie."""
    recalls, rrs, ndcgs = [], [], []
    for q in queries:
        rel = qrels.get(q, set())
        if not rel:
            continue
        hits = retriever.search(coll, q, top_k=k)
        r, rr, nd = score_hits(hits, rel, k)
        recalls.append(r)
        rrs.append(rr)
        ndcgs.append(nd)
    return ({"recall@%d" % k: round(avg(recalls), 4), "mrr": round(avg(rrs), 4),
             "ndcg@%d" % k: round(avg(ndcgs), 4), "n": len(recalls)}, recalls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--max-queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank-candidates", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "retrieval_metrics.md"))
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from embeddings import BGEM3Embedder
    from pageindex import PageIndexTree
    from reranker import BGEReranker
    from retrieval import Retriever

    t_start = time.time()
    print(f"[1/7] Rebuilding eval set (MUST match live build): languages=('hi',), "
          f"max_docs={args.max_docs}, include_english=True, include_translated=True ...")
    docs, queries, qrels = index_build.build_eval_set(
        languages=("hi",), max_docs=args.max_docs, max_queries=args.max_queries,
        include_english=True, include_translated=True)
    scored_queries = [q for q in queries if qrels.get(q)]
    all_pos = sorted({d for s in qrels.values() for d in s})
    n_pos_total = sum(len(qrels[q]) for q in scored_queries)
    print(f"      docs={len(docs)}  queries={len(queries)}  scored_queries={len(scored_queries)}"
          f"  unique_positive_doc_ids={len(all_pos)}  positives(total)={n_pos_total}")

    client = QdrantClient(url=args.qdrant_url)

    # ---- pre-score assertion: qrel doc_ids must resolve in msmarco_xi__passage ----
    print("[2/7] Pre-score assertion: collecting doc_ids from every collection ...")
    coll_doc_ids = {}
    coll_counts = {}
    for s in STRATS:
        coll = f"msmarco_xi__{s}"
        coll_doc_ids[s] = collect_doc_ids(client, coll)
        coll_counts[s] = client.count(coll).count
        print(f"      {coll}: points={coll_counts[s]}  distinct_doc_ids={len(coll_doc_ids[s])}")

    passage_ids = coll_doc_ids["passage"]
    random.seed(13)
    sample = random.sample(all_pos, min(20, len(all_pos)))
    sample_resolved = sum(1 for d in sample if d in passage_ids)
    sample_rate = sample_resolved / len(sample) if sample else 0.0
    full_resolved = sum(1 for d in all_pos if d in passage_ids)
    full_rate = full_resolved / len(all_pos) if all_pos else 0.0
    print(f"      SAMPLE(20) resolution in {PASSAGE}: {sample_resolved}/{len(sample)} = "
          f"{sample_rate:.1%}")
    print(f"      FULL resolution in {PASSAGE}: {full_resolved}/{len(all_pos)} = {full_rate:.2%}")

    if sample_rate < 0.90:
        msg = (f"STOP: qrel doc_id resolution {sample_rate:.0%} < 90% in {PASSAGE}. The rebuilt "
               f"eval set diverges from the live index — publishing recall would be false. "
               f"Investigate build param / shard drift before scoring.")
        print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"# Retrieval metrics — HALTED (divergence)\n\n{msg}\n\n"
                    f"- sampled resolution: {sample_resolved}/{len(sample)} = {sample_rate:.1%}\n"
                    f"- full resolution: {full_resolved}/{len(all_pos)} = {full_rate:.2%}\n")
        sys.exit(2)

    # resolution across every scored collection (justifies the recall denominator per collection)
    resolution = {s: (sum(1 for d in all_pos if d in coll_doc_ids[s]) / len(all_pos))
                  for s in STRATS}

    print("[3/7] Loading BGE-M3 (real embedder) ...")
    emb = MemoEmbedder(BGEM3Embedder())
    retr = Retriever(client, emb, use_sparse=True)

    # ---- 1) MAIN STRATEGY TABLE via eval_chunking.evaluate_strategies ----
    print("[4/7] Strategy A/B (evaluate_strategies) over 6 collections ...")
    strat_report = {s: {"collection": f"msmarco_xi__{s}", "n_chunks": coll_counts[s]}
                    for s in STRATS}
    main_tbl = eval_chunking.evaluate_strategies(strat_report, retr, scored_queries, qrels,
                                                 k=args.k)

    # ---- 2) ABLATION (a): rerank on/off on passage, paired ----
    print("[5/7] Ablation (a): cross-encoder rerank on/off (BGEReranker, top-24 pool) ...")
    reranker = BGEReranker("BAAI/bge-reranker-v2-m3")
    abl_a = {}
    for kk in (8, 10):
        rec_norr, rec_rr, rr_norr, rr_rr = [], [], [], []
        for q in scored_queries:
            rel = qrels[q]
            cand = retr.search(PASSAGE, q, top_k=args.rerank_candidates)  # first-stage RRF pool
            r0, rr0, _ = score_hits(cand[:kk], rel, kk)
            reranked = reranker.rerank(q, list(cand), top_n=kk)
            r1, rr1, _ = score_hits(reranked, rel, kk)
            rec_norr.append(r0); rec_rr.append(r1); rr_norr.append(rr0); rr_rr.append(rr1)
        w, l, t = paired(rec_norr, rec_rr)
        abl_a[kk] = {
            "n": len(rec_norr),
            "norr_recall": round(avg(rec_norr), 4), "rr_recall": round(avg(rec_rr), 4),
            "norr_mrr": round(avg(rr_norr), 4), "rr_mrr": round(avg(rr_rr), 4),
            "delta_recall": round(avg(rec_rr) - avg(rec_norr), 4),
            "delta_mrr": round(avg(rr_rr) - avg(rr_norr), 4),
            "win": w, "loss": l, "tie": t,
        }
        print(f"      k={kk}: recall {abl_a[kk]['norr_recall']} -> {abl_a[kk]['rr_recall']} "
              f"(Δ{abl_a[kk]['delta_recall']:+}), win/loss/tie={w}/{l}/{t}")

    # ---- 3) ABLATION (b): hybrid dense+sparse vs dense-only on passage, paired ----
    print("[6/7] Ablation (b): hybrid dense+sparse vs dense-only ...")
    retr_dense = Retriever(client, emb, use_sparse=False)
    hy_m, hy_rec = score_collection(retr, PASSAGE, scored_queries, qrels, args.k)
    de_m, de_rec = score_collection(retr_dense, PASSAGE, scored_queries, qrels, args.k)
    wb, lb, tb = paired(de_rec, hy_rec)
    abl_b = {"hybrid": hy_m, "dense_only": de_m,
             "delta_recall": round(hy_m[f"recall@{args.k}"] - de_m[f"recall@{args.k}"], 4),
             "win": wb, "loss": lb, "tie": tb}  # win = hybrid beats dense-only
    print(f"      hybrid recall@{args.k}={hy_m[f'recall@{args.k}']} vs dense-only "
          f"{de_m[f'recall@{args.k}']} (Δ{abl_b['delta_recall']:+}), "
          f"hybrid win/loss/tie vs dense={wb}/{lb}/{tb}")

    # ---- 4) ABLATION (c): PageIndex vectorless vs hybrid, paired ----
    print("[7/7] Ablation (c): PageIndexTree vectorless (from raptor) vs hybrid ...")
    tree = PageIndexTree.from_qdrant(client, RAPTOR, beam=5)
    pi_rec, pi_rr, pi_nd = [], [], []
    hy2_rec = []
    for q in scored_queries:
        rel = qrels[q]
        pih = tree.search(q, top_k=args.k)
        r, rr, nd = score_hits(pih, rel, args.k)
        pi_rec.append(r); pi_rr.append(rr); pi_nd.append(nd)
        hyh = retr.search(PASSAGE, q, top_k=args.k)
        hr, _, _ = score_hits(hyh, rel, args.k)
        hy2_rec.append(hr)
    wc, lc, tc = paired(pi_rec, hy2_rec)  # win = hybrid beats pageindex
    abl_c = {
        "pageindex": {"recall@%d" % args.k: round(avg(pi_rec), 4), "mrr": round(avg(pi_rr), 4),
                      "ndcg@%d" % args.k: round(avg(pi_nd), 4), "n": len(pi_rec),
                      "tree_nodes": tree.size, "tree_roots": len(tree._roots),
                      "tree_leaves": tree._n_leaves},
        "hybrid": {"recall@%d" % args.k: round(avg(hy2_rec), 4), "n": len(hy2_rec)},
        "delta_recall_hybrid_minus_pi": round(avg(hy2_rec) - avg(pi_rec), 4),
        "hybrid_win": wc, "hybrid_loss": lc, "tie": tc}
    print(f"      pageindex recall@{args.k}={abl_c['pageindex'][f'recall@{args.k}']} vs hybrid "
          f"{abl_c['hybrid'][f'recall@{args.k}']} "
          f"(Δ{abl_c['delta_recall_hybrid_minus_pi']:+} hybrid-pi)")

    elapsed = time.time() - t_start
    write_report(args, docs, queries, scored_queries, all_pos, n_pos_total, coll_counts,
                 sample_resolved, len(sample), sample_rate, full_rate, resolution,
                 main_tbl, abl_a, abl_b, abl_c, elapsed)
    print(f"\nDONE in {elapsed/60:.1f} min. Report: {args.out}")


def write_report(args, docs, queries, scored_queries, all_pos, n_pos_total, coll_counts,
                 sample_resolved, sample_n, sample_rate, full_rate, resolution,
                 main_tbl, abl_a, abl_b, abl_c, elapsed):
    k = args.k
    N = len(scored_queries)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    gpu = _gpu_name()
    L = []
    A = L.append
    A("# Retrieval metrics — live 20k index (calibration-side)")
    A("")
    A(f"_Generated {now} by `eval/run_chunking_eval_live.py` on Forge "
      f"(host `{platform.node()}`, GPU {gpu}), against the live Qdrant server "
      f"`{args.qdrant_url}` with the real BGE-M3 embedder and real MSMARCO-XI qrels._")
    A("")
    A("## What this is (and is not)")
    A("")
    A("- **Real qrels.** Labels are MSMARCO-XI `is_selected==1` passages — the dataset's own "
      "relevance judgements, not synthetic. Cross-lingual setting: the query is the Hindi "
      "`query` field; relevant passages are the selected English/translated passages that were "
      "actually indexed.")
    A("- **Standard IR metrics.** recall@k / MRR / nDCG@k, distinct-doc credit (a relevant doc "
      "gains once at its first rank, so fine-grained chunkers can't inflate past 1.0).")
    A(f"- **Honest denominator.** recall denominator = every `is_selected` positive for the "
      f"query; the pre-score assertion (below) confirms those positives resolve in the index.")
    A("- **Calibration-side, single run, no CIs.** This is the interim **20k** index, one run, "
      "one box. **No confidence intervals** — bootstrap CIs come with the sealed **40k** eval "
      "(`eval/sealed_test.py`, untouched here). Numbers are point estimates.")
    A("")
    A("## Pre-score assertion — qrel/index alignment")
    A("")
    A(f"The eval set was rebuilt with `index_build.build_eval_set(languages=('hi',), "
      f"max_docs={args.max_docs}, include_english=True, include_translated=True)` — the **same "
      f"`_load_msmarco` code path** (via `load_msmarco_slice`) that `build_real_index.py` used to "
      f"build the live `msmarco_xi__*` collections on 2026-08-15, so the sequential `{{lang}}-N` "
      f"doc_ids line up by construction. Verified empirically before scoring:")
    A("")
    A(f"- **Sampled (20 random qrel doc_ids):** {sample_resolved}/{sample_n} = "
      f"**{sample_rate:.0%}** resolve as payload `doc_id` in `{PASSAGE}` (gate: ≥90%). "
      f"{'PASS' if sample_rate >= 0.9 else 'FAIL'}.")
    A(f"- **Full (all {len(all_pos)} unique positive doc_ids):** **{full_rate:.2%}** resolve in "
      f"`{PASSAGE}`.")
    A("")
    A("Per-collection resolution of the positive doc_ids (justifies the recall denominator on "
      "each collection):")
    A("")
    A("| collection | points | distinct doc_ids | positives resolved |")
    A("|---|---|---|---|")
    for s in STRATS:
        A(f"| msmarco_xi__{s} | {coll_counts[s]} | — | {resolution[s]:.2%} |")
    A("")
    A(f"- Eval corpus rebuilt: **{len(docs)}** docs, **{len(queries)}** candidate queries, "
      f"**{N}** with ≥1 positive (scored), **{len(all_pos)}** unique positive doc_ids, "
      f"**{n_pos_total}** total (query, positive) pairs "
      f"(avg **{n_pos_total/max(N,1):.2f}** positives/query).")
    A("")
    A("## 1. Chunking strategy A/B")
    A("")
    A(f"Real BGE-M3 hybrid retrieval (dense+sparse RRF) via "
      f"`eval_chunking.evaluate_strategies`, top_k={k}, N={N} queries. Sorted by nDCG@{k}.")
    A("")
    A(f"| strategy | n_chunks | recall@{k} | MRR | nDCG@{k} |")
    A("|---|---|---|---|---|")
    for s, v in sorted(main_tbl.items(), key=lambda kv: -kv[1].get(f"ndcg@{k}", 0)):
        A(f"| {s} | {v['n_chunks']} | {v.get(f'recall@{k}')} | {v.get('mrr')} | "
          f"{v.get(f'ndcg@{k}')} |")
    A("")
    A(f"> **raptor** caveat: `msmarco_xi__raptor` mixes {coll_counts['raptor']} points "
      f"(~21k leaves + ~4.8k summaries). This raw A/B scores `retriever.search` output directly, "
      f"so summary hits occupy top slots and are never credited (their doc_ids aren't leaf "
      f"positives) — the LIVE harness applies leaf-expansion (summaries navigate, leaves are "
      f"evidence) which this strategy-level A/B deliberately does not. Read raptor here as "
      f"'raw tree-collection retrieval', not the served path.")
    A("")
    A("## 2. Ablation (a) — cross-encoder rerank on/off (passage)")
    A("")
    A(f"Paired, same N={abl_a[8]['n']} queries. First stage: hybrid top-"
      f"{args.rerank_candidates} pool on `{PASSAGE}` (deployed `rerank_candidates=24`). "
      f"Rerank arm: `src/reranker.BGEReranker` (BAAI/bge-reranker-v2-m3) re-scores the pool and "
      f"keeps top-k. no-rerank arm: first-stage RRF order, top-k. k=8 is the deployed `eff_k`.")
    A("")
    A("| k | recall no-rr | recall +rerank | Δrecall | MRR no-rr | MRR +rerank | ΔMRR | "
      "win/loss/tie |")
    A("|---|---|---|---|---|---|---|---|")
    for kk in (8, 10):
        a = abl_a[kk]
        A(f"| {kk} | {a['norr_recall']} | {a['rr_recall']} | {a['delta_recall']:+} | "
          f"{a['norr_mrr']} | {a['rr_mrr']} | {a['delta_mrr']:+} | "
          f"{a['win']}/{a['loss']}/{a['tie']} |")
    A("")
    A("win/loss/tie = per-query recall of (+rerank) vs (no-rerank). Single run, no CI.")
    A("")
    A("## 3. Ablation (b) — hybrid dense+sparse vs dense-only (passage)")
    A("")
    A(f"Paired, same N={abl_b['hybrid']['n']} queries, `{PASSAGE}`, top_k={k}. Dense-only = "
      f"`Retriever(use_sparse=False)` (dense arm only); hybrid = dense+sparse RRF.")
    A("")
    A(f"| arm | recall@{k} | MRR | nDCG@{k} |")
    A("|---|---|---|---|")
    A(f"| hybrid (dense+sparse) | {abl_b['hybrid'][f'recall@{k}']} | {abl_b['hybrid']['mrr']} | "
      f"{abl_b['hybrid'][f'ndcg@{k}']} |")
    A(f"| dense-only | {abl_b['dense_only'][f'recall@{k}']} | {abl_b['dense_only']['mrr']} | "
      f"{abl_b['dense_only'][f'ndcg@{k}']} |")
    A("")
    A(f"Δrecall@{k} (hybrid − dense-only) = **{abl_b['delta_recall']:+}**; hybrid win/loss/tie "
      f"vs dense-only = **{abl_b['win']}/{abl_b['loss']}/{abl_b['tie']}**. Single run, no CI.")
    A("")
    A("## 4. Ablation (c) — PageIndex vectorless vs hybrid")
    A("")
    pi = abl_c["pageindex"]
    A(f"Paired, same N={pi['n']} queries, top_k={k}. PageIndex = `PageIndexTree.from_qdrant("
      f"msmarco_xi__raptor)` — the current inverted-index (idf-weighted postings over leaves) "
      f"scorer, **zero embeddings at query time**; tree = {pi['tree_nodes']} nodes, "
      f"{pi['tree_roots']} roots, {pi['tree_leaves']} leaves. Hybrid = BGE-M3 dense+sparse on "
      f"`{PASSAGE}`.")
    A("")
    A(f"| arm | recall@{k} | MRR | nDCG@{k} |")
    A("|---|---|---|---|")
    A(f"| pageindex (vectorless) | {pi[f'recall@{k}']} | {pi['mrr']} | {pi[f'ndcg@{k}']} |")
    A(f"| hybrid (dense+sparse) | {abl_c['hybrid'][f'recall@{k}']} | — | — |")
    A("")
    A(f"Δrecall@{k} (hybrid − pageindex) = **{abl_c['delta_recall_hybrid_minus_pi']:+}**; "
      f"hybrid win/loss/tie vs pageindex = "
      f"**{abl_c['hybrid_win']}/{abl_c['hybrid_loss']}/{abl_c['tie']}**. PageIndex is offered as "
      f"an additional vectorless mode (per-query toggle), not the default — it trades recall for "
      f"no query-time embeddings. Single run, no CI.")
    A("")
    A("## Honest summary")
    A("")
    A("**Measured:** real MSMARCO-XI qrels (is_selected), standard recall@k/MRR/nDCG@k with "
      "distinct-doc credit, on the live 20k Qdrant index with the real BGE-M3 embedder and the "
      "real BGEReranker; three paired ablations (rerank, hybrid vs dense, pageindex vs hybrid); "
      "explicit qrel/index alignment assertion before any score.")
    A("")
    A("**NOT measured / caveats:** no confidence intervals (single run — CIs come with the "
      "sealed 40k eval); calibration-side on the **interim 20k** index, not the sealed 40k; "
      "single box, single run; cross-lingual Hindi→English/Indic setting only (`languages=('hi',)`); "
      "raptor row is raw tree-collection retrieval, not the leaf-expanded served path; PageIndex "
      "scored on raptor-tree leaves while hybrid is scored on the passage collection (same "
      "underlying passages).")
    A("")
    A(f"_Wall time: {elapsed/60:.1f} min._")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _gpu_name():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        return out.strip().split("\n")[0]
    except Exception:
        return "unknown-GPU"


if __name__ == "__main__":
    main()
