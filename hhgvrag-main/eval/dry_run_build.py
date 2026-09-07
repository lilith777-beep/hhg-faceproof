"""
dry_run_build.py — the --dry-run build command (P0.4 / deliverable 8).

Prints the frozen manifest, the versioned collection names it WOULD build, the realized
leakage-safe split counts + integrity assertions, and a chunk/build-time projection — WITHOUT
touching Qdrant, a model, or any live prefix. This is what runs at the P0 review to confirm the
split is realizable before the one real Forge build is authorized.

    # local (no dataset): synthetic rows
    python eval/dry_run_build.py --synthetic 2000 --indexed 800 --heldout 200 --tolerance 0.05
    # Forge, ALL 14 languages, whole-corpus-minus-holdout (pin the revision):
    python eval/dry_run_build.py \
        --languages as,bn,gu,hi,kn,ml,mr,ne,or,pa,sa,ta,te,ur --split validation \
        --max-rows 3000 --dataset-revision <sha> --indexed all --heldout 10000 --json

`--max-rows` caps rows PER SHARD. `--indexed all` (or 0) = whole-corpus-minus-holdout: indexed
:= every family not held out; the tolerance band applies to the HOLDOUT only (maximizes qrel
realizability). Multi-language runs build ONE unified family graph with disjoint per-shard
row/query namespaces — English passages repeating across shards collapse into one family.

Never builds. Never promotes. Never writes config. Exit code 2 if the split is UNREALIZABLE
(BuildConstraintError) so CI/the review can gate on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import corpus_spec as cspec           # noqa: E402
import index_build as ib              # noqa: E402


def parse_indexed(value: str) -> int:
    """'all' -> 0 (the whole-corpus-minus-holdout sentinel); else a positive int target."""
    v = str(value).strip().lower()
    if v == "all":
        return 0
    return int(v)


def build_projection(indexed_docs: int, *, chunk_factor: float = 1.15,
                     embed_rate: float = 350.0, raptor_ratio: float = 4.4,
                     summary_rate: float = 9.0, overhead: float = 1.75,
                     fixed_min: float = 12.0) -> dict:
    """Chunk-count + wall-time projection for the versioned build (ONE chunking strategy +
    RAPTOR). Constants are A6000 empirics from the 2026-08-15 20k build (55.5 min, 149k chunks
    across 6 strategies, 4,800 batched Qwen-3B summaries): ~350 chunk-embeds/s, ~9 summaries/s,
    leaves ~= docs x 1.15 (passage strategy), summaries ~= leaves / 4.4; x1.75 covers chunking,
    clustering, topic k-means, and Qdrant upserts; +12 min fixed for model loads."""
    leaves = int(round(indexed_docs * chunk_factor))
    summaries = int(round(leaves / raptor_ratio))
    embed_s = (leaves + summaries) / embed_rate
    gen_s = summaries / summary_rate
    total_h = ((embed_s + gen_s) * overhead + fixed_min * 60.0) / 3600.0
    return {
        "leaf_chunks": leaves,
        "raptor_summaries": summaries,
        "total_points": leaves + summaries,
        "embed_hours": round(embed_s / 3600.0, 2),
        "summary_gen_hours": round(gen_s / 3600.0, 2),
        "projected_build_hours_1strategy_plus_raptor": round(total_h, 2),
        "assumptions": ("A6000 empirics (2026-08-15 20k build): 350 chunk-embeds/s, "
                        "9 summaries/s batched Qwen-3B, chunks=1.15x docs, "
                        "summaries=leaves/4.4, x1.75 overhead, +12min model loads; "
                        "every EXTRA strategy collection adds ~docs*1.15/350s embed time"),
    }


def _report_from_plan(plan, spec: cspec.CorpusBuildSpec) -> dict:
    return {
        "manifest8": spec.manifest8(),
        "manifest_full": spec.manifest_full(),
        "spec": spec.to_canonical_dict(),
        "collections_would_build": plan.collection_names(
            ("passage", "fixed", "recursive", "sentwin", "hierarchical", "raptor")),
        "realization": plan.realization.to_dict(),
        "integrity": plan.integrity,
        "in_corpus_partition": plan.incorpus_partition.counts(),
        "absent_evidence_partition": plan.absent_partition.counts(),
        "excluded_reasons": plan.realization.notes.get("excluded_reasons", {}),
        "query_groups": plan.realization.notes.get("query_groups", {}),
        "projection": build_projection(plan.realization.indexed_docs),
    }


def build_report_multi(shards, spec: cspec.CorpusBuildSpec) -> dict:
    """Plan the multi-shard split and assemble the printable report (pure; no side effects).
    `shards` = sequence of (cfg, rows_iterable) with disjoint per-shard namespaces."""
    return _report_from_plan(ib.plan_split_multi(shards, spec), spec)


def build_report(rows, spec: cspec.CorpusBuildSpec, cfg: str = "hi") -> dict:
    """Single-shard legacy wrapper (tests + single-language runs)."""
    return build_report_multi([(cfg, rows)], spec)


def _load_shards(args) -> tuple:
    """-> (shards, counts) where shards = [(cfg, row_generator)] and `counts` fills with the
    realized rows-per-shard AFTER the generators are consumed by planning."""
    counts: dict = {}
    if args.synthetic:
        rows = ib.synthetic_msmarco_rows(n_rows=args.synthetic, seed=args.seed,
                                         multi_positive_rate=0.2)
        counts["hi"] = len(rows)
        return [("hi", rows)], counts

    cfgs = [c.strip() for c in args.languages.split(",") if c.strip()]
    bad = [c for c in cfgs if c not in ib.MSMARCO_CONFIGS]
    if bad:
        raise SystemExit(f"unknown MSMARCO-XI configs {bad}; valid: {list(ib.MSMARCO_CONFIGS)}")
    if len(set(cfgs)) != len(cfgs):
        raise SystemExit(f"duplicate configs in --languages: {cfgs}")
    revision = None if args.dataset_revision in ("", "UNPINNED") else args.dataset_revision

    def shard_gen(cfg):
        n = 0
        for i, row in enumerate(ib._iter_msmarco_rows(cfg, split=args.split, revision=revision)):
            if args.max_rows and i >= args.max_rows:
                break
            n = i + 1
            yield row
        counts[cfg] = n
        print(f"  loaded shard {cfg}: {n} rows", file=sys.stderr)

    return [(cfg, shard_gen(cfg)) for cfg in cfgs], counts


def main() -> int:
    try:                                    # Windows console is cp1252 by default
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Dry-run the leakage-safe split (no build)")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="use N synthetic rows instead of the real dataset (local testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--languages", type=str, default="hi",
                    help="comma-separated MSMARCO-XI config codes; ALL 14 = "
                         "as,bn,gu,hi,kn,ml,mr,ne,or,pa,sa,ta,te,ur")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap rows PER SHARD (0 = all rows in each shard)")
    ap.add_argument("--dataset-revision", type=str, default="UNPINNED",
                    help="hub commit SHA — pins BOTH the manifest and the actual download")
    ap.add_argument("--indexed", type=str, default="40000",
                    help="indexed docs target, or 'all' = whole-corpus-minus-holdout")
    ap.add_argument("--heldout", type=int, default=10_000)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--allocation-seed", type=int, default=20260817)
    ap.add_argument("--min-heldout-queries", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()

    indexed_target = parse_indexed(args.indexed)
    langs = tuple(x.strip() for x in args.languages.split(",") if x.strip())
    # the realized shard paths + per-shard row cap are corpus-shaping inputs -> HASHED into the
    # manifest (two caps must never share a manifest8)
    suffix = "train" if args.split == "train" else "val"
    shard_paths = (("validation/hinval.parquet",) if args.synthetic else
                   tuple(f"{args.split}/{ib._LANG_FILE[c]}{suffix}.parquet"
                         for c in langs if c in ib._LANG_FILE))
    spec = cspec.CorpusBuildSpec(
        dataset_revision=args.dataset_revision,
        shards=shard_paths, splits=(args.split,), languages=langs,
        max_rows_per_shard=args.max_rows,
        indexed_docs_target=indexed_target, heldout_docs_target=args.heldout,
        size_tolerance=args.tolerance, allocation_seed=args.allocation_seed,
        min_heldout_queries=args.min_heldout_queries)

    shards, counts = _load_shards(args)
    try:
        report = build_report_multi(shards, spec)
    except ib.BuildConstraintError as e:
        print(f"\n[UNREALIZABLE] {e}\n")
        return 2
    report["rows_per_shard"] = dict(sorted(counts.items()))

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    r = report["realization"]
    idx_label = ("ALL (whole-corpus minus holdout)" if indexed_target == 0
                 else str(indexed_target))
    print("=" * 72)
    print(f"DRY RUN -- manifest8 = {report['manifest8']}  (NO BUILD, NO PROMOTION)")
    print("=" * 72)
    print(f"dataset_revision : {spec.dataset_revision}")
    print(f"languages        : {list(spec.languages)}  rows/shard: {report['rows_per_shard']}")
    print(f"targets          : indexed={idx_label} heldout={spec.heldout_docs_target} "
          f"tol=+/-{spec.size_tolerance:.0%}"
          f"{' (holdout only)' if indexed_target == 0 else ''}")
    print("-" * 72)
    print("collections it WOULD build:")
    for c in report["collections_would_build"]:
        print(f"  {c}")
    print("-" * 72)
    print("realized split:")
    print(f"  source_rows       : {r['source_rows']}")
    print(f"  source_passages   : {r['source_passages']}")
    print(f"  unique_canonical  : {r['unique_canonical']}")
    print(f"  families          : {r['n_families']}")
    print(f"  indexed_docs      : {r['indexed_docs']}  ({r['indexed_families']} families)")
    print(f"  heldout_docs      : {r['heldout_docs']}  ({r['heldout_families']} families)")
    print(f"  indexed_queries   : {r['indexed_queries']}  "
          f"(groups: {report['query_groups'].get('indexed', '?')})")
    print(f"  heldout_queries   : {r['heldout_queries']}  "
          f"(groups: {report['query_groups'].get('heldout', '?')})")
    print(f"  excluded_queries  : {r['excluded_queries']}  {report['excluded_reasons']}")
    print(f"  corpus_hash       : {r['corpus_hash'][:16]}...")
    print("-" * 72)
    print(f"in-corpus   cal/dev/sealed : {report['in_corpus_partition']}")
    print(f"absent-evid cal/dev/sealed : {report['absent_evidence_partition']}")
    print("-" * 72)
    pj = report["projection"]
    print(f"BUILD PROJECTION (1 strategy + RAPTOR): ~{pj['leaf_chunks']} leaf chunks + "
          f"{pj['raptor_summaries']} summaries = {pj['total_points']} points -> "
          f"~{pj['projected_build_hours_1strategy_plus_raptor']} h wall "
          f"(embed {pj['embed_hours']}h + summaries {pj['summary_gen_hours']}h + overhead)")
    print("-" * 72)
    print(f"LEAKAGE ASSERTIONS (all must be 0): family_cap="
          f"{report['integrity']['family_intersection']}  "
          f"exact_hash_cap={report['integrity']['exact_hash_intersection']}  "
          f"stable_doc_id_cap={report['integrity']['stable_doc_id_intersection']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
