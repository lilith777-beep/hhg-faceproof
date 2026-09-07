"""
latency.py — the P50 / P70 / P100 analytics (requirement #4).

Replays a query set through the live harness, reads each `QueryTrace`, and reports percentiles
for the total retrieval->output path (the brief's <200ms budget, STT excluded) and per stage.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import Query  # noqa: E402


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    if p >= 100:
        return s[-1]
    k = (p / 100.0) * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def measure_latency(harness, queries, warmup: int = 3) -> dict:
    for q in queries[:warmup]:
        harness.answer(Query(text=q))
    totals, per_stage, decisions = [], {}, {}
    for q in queries:
        resp = harness.answer(Query(text=q))
        tr = resp.trace
        totals.append(tr.retrieval_to_output_ms)
        decisions[tr.decision.value] = decisions.get(tr.decision.value, 0) + 1
        for s in tr.stages:
            if s.stage == "stt":
                continue
            per_stage.setdefault(s.stage, []).append(s.ms)
    report = {
        "n_queries": len(queries),
        "budget_ms": harness.budget_ms,
        "total_ms": {"p50": round(_pct(totals, 50), 2), "p70": round(_pct(totals, 70), 2),
                     "p100": round(_pct(totals, 100), 2)},
        "per_stage_ms": {k: {"p50": round(_pct(v, 50), 2), "p70": round(_pct(v, 70), 2),
                             "p100": round(_pct(v, 100), 2)} for k, v in per_stage.items()},
        "decisions": decisions,
        "under_budget_p50": round(_pct(totals, 50), 2) < harness.budget_ms,
    }
    return report


def write_report(report: dict, path: str) -> None:
    L = ["# Latency report", "",
         f"- queries: **{report['n_queries']}**  ·  budget: **{report['budget_ms']} ms** "
         f"(retrieval→output, STT excluded)",
         f"- **P50 {report['total_ms']['p50']} ms · P70 {report['total_ms']['p70']} ms · "
         f"P100 {report['total_ms']['p100']} ms**",
         f"- decisions: {report['decisions']}", "",
         "| stage | P50 | P70 | P100 |", "|---|---|---|---|"]
    for stage, v in report["per_stage_ms"].items():
        L.append(f"| {stage} | {v['p50']} | {v['p70']} | {v['p100']} |")
    L.append(f"| **TOTAL** | **{report['total_ms']['p50']}** | "
             f"**{report['total_ms']['p70']}** | **{report['total_ms']['p100']}** |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
