"""
eval_chunking.py — A/B the chunking strategies on retrieval quality (recall@k / MRR / nDCG),
so promoting one to the live path is evidence-based, not a guess.

`qrels`: {query_text: set(relevant_doc_id)}. A retrieved chunk counts as relevant if its
payload doc_id is in that query's relevant set.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate_strategies(strategy_report: dict, retriever, eval_queries, qrels: dict,
                        k: int = 10) -> dict:
    out = {}
    for sname, info in strategy_report.items():
        coll = info["collection"]
        recalls, rrs, ndcgs = [], [], []
        for q in eval_queries:
            rel = qrels.get(q, set())
            if not rel:
                continue
            hits = retriever.search(coll, q, top_k=k)
            # score DISTINCT relevant docs (a relevant doc gains once, at its first rank) — else
            # a fine-grained chunker that returns many chunks of one doc inflates nDCG past 1.0
            seen, gains, first_rank = set(), [], None
            for i, h in enumerate(hits):
                did = h.payload.get("doc_id")
                if did in rel and did not in seen:
                    seen.add(did)
                    gains.append(1.0)
                    first_rank = first_rank or (i + 1)
                else:
                    gains.append(0.0)
            recalls.append(len(seen) / len(rel))
            rrs.append(1.0 / first_rank if first_rank else 0.0)
            ideal = _dcg([1.0] * min(len(rel), k)) or 1.0
            ndcgs.append(min(_dcg(gains) / ideal, 1.0))
        out[sname] = {
            "collection": coll, "n_chunks": info["n_chunks"],
            "recall@%d" % k: round(_avg(recalls), 4),
            "mrr": round(_avg(rrs), 4), "ndcg@%d" % k: round(_avg(ndcgs), 4),
        }
    return out


def pick_winner(results: dict, metric_key_suffix: str = "ndcg@10") -> str:
    return max(results, key=lambda s: results[s].get(metric_key_suffix, 0.0)) if results else ""


def write_report(results: dict, path: str, winner: str = "") -> None:
    keys = ["recall@10", "mrr", "ndcg@10"]
    L = ["# Chunking A/B report", "",
         f"Winner (promoted to live): **{winner}**" if winner else "", "",
         "| strategy | n_chunks | " + " | ".join(keys) + " |",
         "|---|---|" + "|".join(["---"] * len(keys)) + "|"]
    for s, v in sorted(results.items(), key=lambda kv: -kv[1].get("ndcg@10", 0)):
        row = " | ".join(str(v.get(k, "")) for k in keys)
        star = " ⭐" if s == winner else ""
        L.append(f"| {s}{star} | {v['n_chunks']} | {row} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0
