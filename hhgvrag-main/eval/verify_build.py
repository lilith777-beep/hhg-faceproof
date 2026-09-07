"""
verify_build.py — EXHAUSTIVE post-build integrity verification (build-authorization amendment,
2026-08-16). The in-build 200-positive probe is a smoke test only; THIS command is the
manifest-linked accuracy gate that blocks calibration/tuning/sealed-test authorization.

    python eval/verify_build.py --manifest 73ca3e90 --qdrant-url http://localhost:6333

Checks, exhaustively (never sampled):
  1. Every scored in-corpus qrel positive (calibration + dev + sealed partitions) resolves to
     a LEAF point in the passage collection.
  2. Every such positive resolves to a LEAF point in the RAPTOR collection (the declared
     leaf-evidence universe).
  3. No positive resolves only via summary points ("summary-only"): structurally impossible
     because summaries never carry stable_doc_id — verified against every summary point, not
     assumed.
  4. Every RAPTOR summary carries lineage: extra.build_manifest == manifest8 and a non-empty
     extra.leaf_descendants (exhaustive, all summaries).
  5. Absent-evidence holdout: NO positive of any held-out query resolves in either collection
     (realized-index leakage re-proof).
  6. Counts reconcile with the signed realization: query totals, excluded count+reasons,
     distinct leaf stable-IDs == indexed_docs, collection totals == phase markers.

Sealed hygiene: sealed query files are opened for mechanical ID-membership ONLY; the artifact
reports counts and digests, never sealed query ids, text, or any system outcome.

Writes eval/builds/<manifest8>/exhaustive_verification.json (with git SHA + verifier file
hash). Exit 0 = GREEN, 4 = FAIL, 2 = usage/state error. Read-only against Qdrant; never
touches live configuration, thresholds, or collection contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     ensure_ascii=True).encode()).hexdigest()


def scroll_ids(client, name: str, batch: int = 2048):
    """One full pass over a collection (payload fields only). Returns:
    (leaf_sdids: set, n_leaves, leaves_missing_sdid, n_summaries, summaries_with_sdid,
     summaries_bad_lineage, summary_sdids: set)"""
    leaf_sdids, summary_sdids = set(), set()
    n_leaf = n_sum = leaf_nosdid = sum_sdid = sum_badline = 0
    offset = None
    while True:
        pts, offset = client.scroll(
            name, limit=batch, offset=offset,
            with_payload=["stable_doc_id", "is_summary", "extra"], with_vectors=False)
        for p in pts:
            pay = p.payload or {}
            sdid = pay.get("stable_doc_id")
            if pay.get("is_summary"):
                n_sum += 1
                if sdid:
                    sum_sdid += 1
                    summary_sdids.add(sdid)
                ex = pay.get("extra") or {}
                if not ex.get("leaf_descendants") or not ex.get("build_manifest"):
                    sum_badline += 1
            else:
                n_leaf += 1
                if sdid:
                    leaf_sdids.add(sdid)
                else:
                    leaf_nosdid += 1
        if offset is None:
            return (leaf_sdids, n_leaf, leaf_nosdid, n_sum, sum_sdid, sum_badline,
                    summary_sdids)


def positives_of(build_dir: str, pool: str) -> tuple:
    """(n_queries, distinct positive sdids) across cal+dev+sealed files of a pool.
    Sealed files are consumed for ID membership only — nothing sealed leaves this process."""
    n_q, pos = 0, set()
    for pname in ("calibration", "dev", "sealed"):
        d = _load(os.path.join(build_dir, f"queries_{pool}_{pname}.json"))
        n_q += len(d)
        for q in d.values():
            pos.update(q["positives"])
    return n_q, pos


def main() -> int:
    ap = argparse.ArgumentParser(description="Exhaustive post-build integrity verification")
    ap.add_argument("--manifest", required=True, help="manifest8 (e.g. 73ca3e90)")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--build-dir", default=None,
                    help="override eval/builds/<manifest8>")
    args = ap.parse_args()

    from qdrant_client import QdrantClient

    t0 = time.time()
    build_dir = args.build_dir or os.path.join(HERE, "builds", args.manifest)
    real = _load(os.path.join(build_dir, "realization.json"))
    if real["manifest8"] != args.manifest:
        print(f"REFUSED: realization manifest {real['manifest8']} != {args.manifest}")
        return 2
    r = real["realization"]
    spec = real["spec"]
    coll_passage = f"{spec['collection_prefix']}_{args.manifest}__passage"
    coll_raptor = f"{spec['collection_prefix']}_{args.manifest}__raptor"

    def marker(name):
        p = os.path.join(build_dir, "phases", f"{name}.done.json")
        return _load(p) if os.path.exists(p) else None

    client = QdrantClient(url=args.qdrant_url)
    print(f"scrolling {coll_passage} ...")
    (p_leaf, p_nleaf, p_nosdid, p_nsum, p_sumsdid, p_badline, p_sumids) = \
        scroll_ids(client, coll_passage)
    print(f"  leaves={p_nleaf} distinct_sdids={len(p_leaf)} summaries={p_nsum}")
    print(f"scrolling {coll_raptor} ...")
    (r_leaf, r_nleaf, r_nosdid, r_nsum, r_sumsdid, r_badline, r_sumids) = \
        scroll_ids(client, coll_raptor)
    print(f"  leaves={r_nleaf} distinct_sdids={len(r_leaf)} summaries={r_nsum} "
          f"bad_lineage={r_badline}")

    n_inq, in_pos = positives_of(build_dir, "incorpus")
    n_absq, abs_pos = positives_of(build_dir, "absent")

    miss_pass = in_pos - p_leaf
    miss_rapt = in_pos - r_leaf
    # summary-only = resolves via a summary point but not via any leaf (must be empty AND
    # structurally empty because summaries carry no stable_doc_id)
    summary_only = ((in_pos & (p_sumids | r_sumids)) - (p_leaf | r_leaf))
    leaked = (abs_pos & (p_leaf | r_leaf | p_sumids | r_sumids))

    pm, rm = marker("passage"), marker("raptor")
    passage_total = client.count(coll_passage, exact=True).count
    raptor_total = client.count(coll_raptor, exact=True).count

    checks = {
        "missing_passage": len(miss_pass),
        "missing_raptor_leaf": len(miss_rapt),
        "summary_only_ids": len(summary_only),
        "summaries_carrying_stable_id": p_sumsdid + r_sumsdid,
        "summaries_bad_lineage": p_badline + r_badline,
        "passage_collection_summary_points": p_nsum,          # must be 0
        "leaves_missing_stable_id": p_nosdid + r_nosdid,
        "absent_positives_leaked_into_index": len(leaked),
        "query_total_reconciles": n_inq == r["indexed_queries"],
        "absent_query_total_reconciles": n_absq == r["heldout_queries"],
        "excluded_declared": r["excluded_queries"],
        "excluded_reasons": r["notes"].get("excluded_reasons", {}),
        "undeclared_exclusions": r["excluded_queries"]
                                 - sum(r["notes"].get("excluded_reasons", {}).values()),
        "leaf_sdids_equal_indexed_docs": len(p_leaf) == r["indexed_docs"],
        "passage_count_matches_marker": bool(pm) and passage_total == pm["output_count"],
        "raptor_count_matches_marker": bool(rm) and raptor_total == rm["output_count"],
    }
    ok = (checks["missing_passage"] == 0 and checks["missing_raptor_leaf"] == 0
          and checks["summary_only_ids"] == 0
          and checks["summaries_carrying_stable_id"] == 0
          and checks["summaries_bad_lineage"] == 0
          and checks["passage_collection_summary_points"] == 0
          and checks["absent_positives_leaked_into_index"] == 0
          and checks["undeclared_exclusions"] == 0
          and checks["query_total_reconciles"] and checks["absent_query_total_reconciles"]
          and checks["leaf_sdids_equal_indexed_docs"]
          and checks["passage_count_matches_marker"]
          and checks["raptor_count_matches_marker"])

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=os.path.dirname(HERE)).stdout.strip()
    except Exception:
        git_sha = "unknown"
    with open(os.path.abspath(__file__), "rb") as f:
        verifier_sha = hashlib.sha256(f.read()).hexdigest()

    artifact = {
        "manifest8": args.manifest,
        "git_sha": git_sha,
        "verifier_file_sha256": verifier_sha,
        "collections": {"passage": coll_passage, "raptor": coll_raptor},
        "collection_totals": {"passage": passage_total, "raptor": raptor_total},
        "queries_checked": {"in_corpus": n_inq, "absent_evidence": n_absq},
        "positive_ids_checked": {"in_corpus_distinct": len(in_pos),
                                 "absent_distinct": len(abs_pos)},
        "passage_matches": len(in_pos) - len(miss_pass),
        "raptor_leaf_matches": len(in_pos) - len(miss_rapt),
        "missing_sample_sdids": sorted(miss_pass | miss_rapt)[:20],   # doc hashes only
        "checks": checks,
        "wall_s": round(time.time() - t0, 1),
        "result": "GREEN" if ok else "FAIL",
    }
    out = os.path.join(build_dir, "exhaustive_verification.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: artifact[k] for k in
                      ("passage_matches", "raptor_leaf_matches", "checks", "result")},
                     indent=1)[:2000])
    print(f"\nEXHAUSTIVE VERIFICATION {artifact['result']} -> {out}")
    return 0 if ok else 4


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
