# -*- coding: utf-8 -*-
"""
stress_diag_gateD.py — mechanism diagnostic for the gate-D load-induced accuracy drop.

The control proved the drop is real (same load pool: 95.6% uncontended -> 65.5% under load),
yet rerank/fallback/error counts were all ~0. This fires a concurrent burst of in-domain
queries that ANSWER uncontended and dumps every abstainer's full trace (decision, ood signal,
rerank_top, n_retrieved, and any degraded stage) to expose the mechanism.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import requests

import stress_live as S


def full_trace(base, text, top_k=8):
    r = requests.post(base + "/ask_text", json={"text": text, "top_k": top_k}, timeout=30)
    j = r.json()
    tr = j.get("trace") or {}
    return {"text": text, "decision": j.get("decision"), "cache_hit": tr.get("cache_hit"),
            "ood": tr.get("ood"), "rerank_top": tr.get("rerank_top"),
            "n_retrieved": tr.get("n_retrieved"), "total_ms": tr.get("total_ms"),
            "stages": [(s["stage"], round(s["ms"], 1), s["ok"], s.get("note"))
                       for s in (tr.get("stages") or [])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    en_all, hi_all = S.load_query_pools()
    nb = 48 // 2
    # load pool queries known to answer uncontended (from the control)
    pool = (en_all[nb:nb + 60] + hi_all[nb:nb + 30])[:args.n]

    # fire all concurrently (contention) with cache-busting distinct k per request so none
    # short-circuit on the cache -> every one recomputes under contention
    ex = ThreadPoolExecutor(max_workers=args.n)
    futs = [ex.submit(full_trace, args.base, q, 11 + (i % 5)) for i, q in enumerate(pool)]
    recs = [f.result() for f in futs]

    dec = {}
    for r in recs:
        dec[r["decision"]] = dec.get(r["decision"], 0) + 1
    ans = sum(1 for r in recs if r["decision"] == "answer")
    print(f"concurrent burst n={len(recs)}  answer-rate={ans}/{len(recs)}={ans/len(recs):.1%}  "
          f"decisions={dec}")
    print("\n--- ABSTAINERS (full trace) ---")
    for r in recs:
        if r["decision"] != "answer":
            print(json.dumps(r, ensure_ascii=False))
    # one answerer for contrast
    for r in recs:
        if r["decision"] == "answer":
            print("\n--- one ANSWERER for contrast ---")
            print(json.dumps(r, ensure_ascii=False))
            break
    print(f"\ntotal HTTP attempts this diag: {S._ATTEMPTS['n'] + len(recs)}")


if __name__ == "__main__":
    main()
