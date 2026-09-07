# -*- coding: utf-8 -*-
"""
stress_gated_control.py — gate-D confound control.

The main stress run measured the ANSWER-rate "drop" as sequential-baseline-pool (95.8%) vs
under-load-compute-pool (65.5%). Those are DISJOINT query samples, so the gap conflates
sample difficulty with any load effect. This control removes the confound: it runs the SAME
load-pool queries SEQUENTIALLY and UNCONTENDED, with a cache-busting top_k (9 instead of the
stress run's 8 -> different cache key -> forced recompute), so we get the load pool's
uncontended compute ANSWER-rate. Comparing that to the under-load 65.5% isolates load from
sample.

Also re-checks the baseline pool at top_k=9 as a positive control (cache-bust must not move a
pool's answer-rate).

Run (on Forge, after the main stress run, server idle):
    ~/anaconda3/envs/hhgvrag/bin/python eval/stress_gated_control.py --base http://localhost:8000
"""
from __future__ import annotations

import argparse

import stress_live as S


def rate(recs):
    a = sum(1 for r in recs if r["decision"] == "answer")
    return a, len(recs), (a / len(recs) if recs else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--top-k", type=int, default=9, help="cache-busting k (stress used 8)")
    args = ap.parse_args()

    en_all, hi_all = S.load_query_pools()
    nb = 48 // 2
    base_en, base_hi = en_all[:nb], hi_all[:nb]
    load_en, load_hi = en_all[nb:nb + 60], hi_all[nb:nb + 30]
    print(f"pools: base={len(base_en)+len(base_hi)} load={len(load_en)+len(load_hi)} "
          f"(cache-bust top_k={args.top_k})")

    # load pool, sequential + uncontended + cache-busted -> uncontended compute answer-rate
    load_recs = [S.ask(args.base, q, None, False, args.top_k, "control_load", "control")
                 for q in (load_en + load_hi)]
    la, ln, lr = rate(load_recs)
    load_cache_hits = sum(1 for r in load_recs if r.get("cache_hit"))

    # base pool positive control
    base_recs = [S.ask(args.base, q, None, False, args.top_k, "control_base", "control")
                 for q in (base_en + base_hi)]
    ba, bn, br = rate(base_recs)
    base_cache_hits = sum(1 for r in base_recs if r.get("cache_hit"))

    # decision breakdown for the load pool (why non-answers happen)
    dec = {}
    for r in load_recs:
        dec[r["decision"]] = dec.get(r["decision"], 0) + 1

    print("\n==== GATE-D CONTROL ====")
    print(f"LOAD pool  uncontended compute ANSWER-rate: {la}/{ln} = {lr:.1%}  "
          f"(cache_hits={load_cache_hits}/{ln})")
    print(f"BASE pool  uncontended compute ANSWER-rate: {ba}/{bn} = {br:.1%}  "
          f"(cache_hits={base_cache_hits}/{bn})  [positive control ~ 95.8%]")
    print(f"LOAD pool decisions: {dec}")
    print(f"\nUnder-load compute ANSWER-rate was 65.5%. LOAD-pool uncontended = {lr:.1%}.")
    print(f"=> real load-induced drop (same pool) = {(lr-0.655)*100:+.1f}pp is the sample vs "
          f"load separation; if uncontended ~= under-load, the apparent 30pp gap is SAMPLE, "
          f"not load.")
    print(f"total HTTP attempts this control: {S._ATTEMPTS['n']}")


if __name__ == "__main__":
    main()
