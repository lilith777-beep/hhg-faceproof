"""
calibrate_decisions.py — Stages B/C of calibration (CALIBRATION partitions only).

Stage B (record): run the FULL in-corpus + absent calibration partitions through
embed -> search (Stage-A winner config) -> rerank ONCE, recording exactly the features the
production gate consumes (guardrails.ood_gate + the harness rerank-veto):
    dense_top   — max dense cosine over retrieved (the gate's PRIMARY signal)
    lexical_top — query-token coverage fallback (gate's SECONDARY signal, fixed 0.34 floor)
    rerank_top / rerank_margin — cross-encoder top score and top1-top2 margin (veto signal)
    topk_sdids  — reranked top-k stable_doc_ids (citation-evidence proxy)
    hit         — any positive in topk (in-corpus only)

Stage C (sweep, offline — no model calls): replay decision(tau, veto) over the recorded
features exactly as production decides:
    ANSWER iff (dense_top >= tau OR lexical_top >= 0.34) AND rerank_top >= veto
per (tau, veto) report:
    in-corpus coverage, false-answer proxy rate (ANSWER with zero positive in topk),
    absent answer rate (any ANSWER on a held-out-evidence query), abstain precision/recall.
Selection: max in-corpus coverage subject to
    false_answer_rate <= bound  AND  absent_answer_rate <= bound   (default 0.02)
Bootstrap 95% CIs (seeded) at the chosen operating point. Proposal only — nothing is
applied to live config; dev partition confirms before freeze; sealed never opened here.

    python eval/calibrate_decisions.py --manifest 73ca3e90 --qdrant-url http://localhost:6333

Artifact: eval/builds/<m8>/decision_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def bootstrap_ci(flags, n_boot=2000, seed=20260817):
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choice(flags) for _ in range(n)) / n for _ in range(n_boot))
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4))


def record(qdict, coll, emb, retr, rr, depth, k, tag, t0):
    from guardrails import ood_gate
    rows = []
    qids = sorted(qdict)
    for i in range(0, len(qids), 64):
        batch = qids[i:i + 64]
        ers = emb.embed_docs([qdict[q]["text"] for q in batch])
        for q, er in zip(batch, ers):
            hits = retr.search(coll, er, top_k=depth)
            ranked = rr.rerank(qdict[q]["text"], hits, top_n=k)
            g = ood_gate(hits, threshold=999.0, query=qdict[q]["text"])  # thr sentinel: we
            # only want its computed signals; decisions replay offline in Stage C
            scores = [getattr(c, "rerank_score", 0.0) or 0.0 for c in ranked]
            pos = set(qdict[q].get("positives", ()))
            sdids = [(c.payload or {}).get("stable_doc_id") for c in ranked]
            rows.append({
                "qid": q, "lang": q.split(":", 1)[0],
                "dense_top": max((h.dense_score or 0.0) for h in hits) if hits else 0.0,
                "lexical_top": g.lexical_top or 0.0,
                "rerank_top": scores[0] if scores else 0.0,
                "rerank_margin": round(scores[0] - scores[1], 4) if len(scores) > 1 else 0.0,
                "hit": bool(pos and set(sdids) & pos),
            })
        if i % 1920 == 0:
            print(f"  [{tag}] {i + len(batch)}/{len(qids)} at {(time.time()-t0)/60:.1f}min")
    return rows


def decide(row, tau, veto):
    admits = row["dense_top"] >= tau or row["lexical_top"] >= 0.34
    return admits and row["rerank_top"] >= veto


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--false-answer-bound", type=float, default=0.02)
    ap.add_argument("--depth", type=int, default=0, help="0 = Stage-A winner")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--fusion", default="", help="'' = Stage-A winner")
    ap.add_argument("--record-only", action="store_true")
    args = ap.parse_args()

    build_dir = os.path.join(HERE, "builds", args.manifest)
    real = _load(os.path.join(build_dir, "realization.json"))
    ex = os.path.join(build_dir, "exhaustive_verification.json")
    if not (os.path.exists(ex) and _load(ex).get("result") == "GREEN"):
        print("REFUSED: exhaustive verification not GREEN")
        return 2
    coll = f"{real['spec']['collection_prefix']}_{args.manifest}__passage"

    winner = {}
    rep_p = os.path.join(build_dir, "calibration_report.json")
    if os.path.exists(rep_p):
        winner = _load(rep_p).get("winner", {})
    depth = args.depth or winner.get("depth", 24)
    k = args.k or winner.get("k", 8)
    fusion = args.fusion or winner.get("fusion", "hybrid")

    rec_p = os.path.join(build_dir, "decision_features.json")
    t0 = time.time()
    if os.path.exists(rec_p):
        feats = _load(rec_p)
        print(f"reusing recorded features ({len(feats['in_corpus'])}+{len(feats['absent'])})")
    else:
        from qdrant_client import QdrantClient
        from embeddings import BGEM3Embedder
        from reranker import BGEReranker
        from retrieval import Retriever
        client = QdrantClient(url=args.qdrant_url, timeout=60)
        emb = BGEM3Embedder(batch_size=16)
        rr = BGEReranker()
        retr = Retriever(client, emb, use_sparse=(fusion == "hybrid"))
        cal_in = _load(os.path.join(build_dir, "queries_incorpus_calibration.json"))
        cal_abs = _load(os.path.join(build_dir, "queries_absent_calibration.json"))
        feats = {"config": {"depth": depth, "k": k, "fusion": fusion},
                 "in_corpus": record(cal_in, coll, emb, retr, rr, depth, k, "in", t0),
                 "absent": record(cal_abs, coll, emb, retr, rr, depth, k, "abs", t0)}
        with open(rec_p, "w", encoding="utf-8") as f:
            json.dump(feats, f, ensure_ascii=False)
        print(f"features recorded -> {rec_p} ({(time.time()-t0)/60:.1f}min)")
    if args.record_only:
        return 0

    # ---- Stage C: offline sweep ---------------------------------------------------------
    inr, absr = feats["in_corpus"], feats["absent"]
    sweep, best = [], None
    for tau_i in range(40, 76):
        tau = tau_i / 100.0
        for veto in (0.02, 0.03, 0.05, 0.08):
            ans = [decide(r, tau, veto) for r in inr]
            cov = sum(ans) / len(inr)
            false_flags = [1 if (a and not r["hit"]) else 0
                           for a, r in zip(ans, inr) if a]
            fr = sum(false_flags) / max(1, sum(ans))
            abs_ans = sum(decide(r, tau, veto) for r in absr) / len(absr)
            row = {"tau": tau, "veto": veto, "coverage": round(cov, 4),
                   "false_rate": round(fr, 4), "absent_answer_rate": round(abs_ans, 4)}
            sweep.append(row)
            ok = fr <= args.false_answer_bound and abs_ans <= args.false_answer_bound
            if ok and (best is None or cov > best["coverage"]):
                best = dict(row)
    if best:
        tau, veto = best["tau"], best["veto"]
        ans = [decide(r, tau, veto) for r in inr]
        best["false_rate_ci"] = bootstrap_ci(
            [1 if (a and not r["hit"]) else 0 for a, r in zip(ans, inr) if a])
        best["coverage_ci"] = bootstrap_ci([1 if a else 0 for a in ans])
        best["absent_ci"] = bootstrap_ci([1 if decide(r, tau, veto) else 0 for r in absr])
        by_lang = {}
        for r, a in zip(inr, ans):
            d = by_lang.setdefault(r["lang"], {"n": 0, "ans": 0, "false": 0})
            d["n"] += 1
            d["ans"] += int(a)
            d["false"] += int(a and not r["hit"])
        best["per_language"] = by_lang

    out = {"manifest8": args.manifest, "config": feats["config"],
           "n": {"in_corpus": len(inr), "absent": len(absr)},
           "false_answer_bound": args.false_answer_bound,
           "chosen": best, "sweep": sweep,
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=os.path.join(HERE, "..")).stdout.strip()}
    with open(os.path.join(build_dir, "decision_calibration.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({"chosen": best}, indent=1)[:900])
    print(f"\nDECISION CALIBRATION written ({(time.time()-t0)/60:.1f}min) — proposal only; "
          f"dev confirms; promotion manual.")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
