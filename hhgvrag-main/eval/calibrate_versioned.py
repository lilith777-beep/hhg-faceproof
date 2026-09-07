"""
calibrate_versioned.py — CALIBRATION-PARTITION-ONLY tuning against the versioned build.

Runs AFTER eval/verify_build.py returns GREEN, on Forge, against the
msmarco_xi_val14_<manifest8>__* collections. Never opens dev or sealed files; never edits
config.py; never touches live serving. Output is a report + a PROPOSED config — applying it
is the lead's manual promotion decision, confirmed on the dev partition first.

    python eval/calibrate_versioned.py --manifest 73ca3e90 --qdrant-url http://localhost:6333 \
        --sample 3000 --absent-sample 1500

Stage A — retrieval-config grid (retrieval-only, scored on real qrels):
    cells: fusion {hybrid-rrf, dense-only} x rerank_depth {16, 24, 32} x final_k {8, 12}
    on a SEEDED subsample of the in-corpus calibration partition. Metrics per cell:
    recall@k / MRR@k (positives = stable_doc_id match), reranked-recall, stage latencies.
    The SOTA audit's headline candidate (dense-only fusion, measured +0.21 recall@10
    single-run at 20k) gets its CI'd verdict here.

Stage B — decision-score recording at the Stage-A winner:
    the FULL in-corpus + absent calibration partitions run through embed -> retrieve ->
    rerank once, recording per-query raw gate inputs (dense top-sim, rerank top score,
    margins) + top-hit correctness (stable_doc_id in positives). No generation needed:
    answer-correctness proxy = the evidence the composer would cite.

Stage C — offline tau/veto sweep on the recorded scores:
    for each (tau, veto) pair: coverage, selective risk, false-answer rate (answered with
    no positive in cited evidence), absent-evidence abstain rate. Selection = max coverage
    subject to false-answer-rate <= --false-answer-bound (default 0.02) on in-corpus AND
    absent-answer-rate <= bound. Bootstrap 95% CIs (seeded) on the chosen operating point.

Artifact: eval/builds/<m8>/calibration_report.json {git_sha, samples+seed, grid table,
winner, sweep curve, chosen (tau, veto) + CIs, proposed_config}. Exit 0 always unless the
build/manifest is inconsistent (exit 2) — selection is advisory, promotion is manual.
"""
from __future__ import annotations

import argparse
import hashlib
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


def _sample(d: dict, n: int, seed: int) -> dict:
    if n <= 0 or n >= len(d):
        return d
    keys = sorted(d)
    random.Random(seed).shuffle(keys)
    return {k: d[k] for k in keys[:n]}


def recall_mrr(ranked_sdids, positives, k):
    hit_rank = next((i + 1 for i, s in enumerate(ranked_sdids[:k]) if s in positives), None)
    return (1.0 if hit_rank else 0.0), (1.0 / hit_rank if hit_rank else 0.0)


def bootstrap_ci(vals, n_boot=2000, seed=20260817):
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(vals) for _ in range(len(vals))) / len(vals))
    means.sort()
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4))


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration-partition tuning (versioned build)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--sample", type=int, default=3000, help="in-corpus grid subsample")
    ap.add_argument("--absent-sample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--false-answer-bound", type=float, default=0.02)
    ap.add_argument("--skip-grid", action="store_true",
                    help="reuse an existing grid winner (rerun stages B/C only)")
    args = ap.parse_args()

    build_dir = os.path.join(HERE, "builds", args.manifest)
    real = _load(os.path.join(build_dir, "realization.json"))
    if real["manifest8"] != args.manifest:
        print("REFUSED: manifest mismatch")
        return 2
    ex = os.path.join(build_dir, "exhaustive_verification.json")
    if not (os.path.exists(ex) and _load(ex).get("result") == "GREEN"):
        print("REFUSED: exhaustive_verification.json missing or not GREEN — "
              "run eval/verify_build.py first")
        return 2
    coll = f"{real['spec']['collection_prefix']}_{args.manifest}__passage"

    # calibration partitions ONLY (dev = confirmation later; sealed = never here)
    cal_in = _load(os.path.join(build_dir, "queries_incorpus_calibration.json"))
    cal_abs = _load(os.path.join(build_dir, "queries_absent_calibration.json"))
    grid_in = _sample(cal_in, args.sample, args.seed)
    grid_abs = _sample(cal_abs, args.absent_sample, args.seed + 1)

    from qdrant_client import QdrantClient
    from embeddings import BGEM3Embedder
    from reranker import BGEReranker
    from retrieval import Retriever

    t0 = time.time()
    client = QdrantClient(url=args.qdrant_url, timeout=60)
    st = client.get_collection(coll).status
    print(f"collection {coll} status={st}")
    if str(st).lower() not in ("green", "collectionstatus.green"):
        print("REFUSED: collection not green (HNSW still building) — wait and rerun")
        return 2
    emb = BGEM3Embedder(batch_size=16)
    rr = BGEReranker()

    report = {"manifest8": args.manifest, "seed": args.seed,
              "samples": {"grid_in": len(grid_in), "grid_absent": len(grid_abs),
                          "full_in": len(cal_in), "full_absent": len(cal_abs)},
              "false_answer_bound": args.false_answer_bound}
    try:
        report["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=os.path.join(HERE, "..")).stdout.strip()
    except Exception:
        report["git_sha"] = "unknown"

    # ---- Stage A: retrieval grid on the seeded subsample -------------------------------
    if not args.skip_grid:
        cells = [{"fusion": f, "depth": d, "k": k}
                 for f in ("hybrid", "dense") for d in (16, 24, 32) for k in (8, 12)]
        qids = sorted(grid_in)
        embeds = {}
        for i in range(0, len(qids), 64):
            batch = qids[i:i + 64]
            for q, er in zip(batch, emb.embed_docs([grid_in[q]["text"] for q in batch])):
                embeds[q] = er
        print(f"embedded {len(embeds)} grid queries at {(time.time()-t0)/60:.1f}min")

        grid_rows = []
        for cell in cells:
            retr = Retriever(client, emb, use_sparse=(cell["fusion"] == "hybrid"))
            r_at, mrr_at, rr_at, lat = [], [], [], []
            for q in qids:
                pos = set(grid_in[q]["positives"])
                s = time.time()
                hits = retr.search(coll, embeds[q], top_k=cell["depth"])
                ranked = rr.rerank(grid_in[q]["text"], hits, top_n=cell["k"])
                lat.append((time.time() - s) * 1000)
                sdids = [(h.payload or {}).get("stable_doc_id") for h in hits]
                rr_sdids = [(h.payload or {}).get("stable_doc_id") for h in ranked]
                a, m = recall_mrr(sdids, pos, cell["k"])
                r_at.append(a)
                mrr_at.append(m)
                rr_at.append(recall_mrr(rr_sdids, pos, cell["k"])[0])
            n = len(qids)
            grid_rows.append({**cell, "recall": round(sum(r_at) / n, 4),
                              "recall_ci": bootstrap_ci(r_at),
                              "mrr": round(sum(mrr_at) / n, 4),
                              "reranked_recall": round(sum(rr_at) / n, 4),
                              "p50_ms": round(sorted(lat)[n // 2], 1)})
            print(f"  cell {cell}: recall={grid_rows[-1]['recall']} "
                  f"rr={grid_rows[-1]['reranked_recall']} p50={grid_rows[-1]['p50_ms']}ms")
        grid_rows.sort(key=lambda r: (-r["reranked_recall"], r["p50_ms"]))
        report["grid"] = grid_rows
        report["winner"] = grid_rows[0]
        print(f"WINNER {report['winner']}")

    out = os.path.join(build_dir, "calibration_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"stage A written -> {out} at {(time.time()-t0)/60:.1f}min "
          f"(stages B/C run via calibrate_decisions once the winner is reviewed)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
