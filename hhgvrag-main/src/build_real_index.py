"""
build_real_index.py — build the real MSMARCO-XI index on ANY GPU box (Forge / on-prem).
Mirrors modal_app.build_index exactly, minus Modal. Run from hhgvrag/ with the GPU venv:

    python src/build_real_index.py --max-docs 20000 --languages hi
    python src/build_real_index.py --no-llm-raptor          # extractive summaries (faster)

Writes the Qdrant collections to --qdrant-path (default ./qdrant_data). The server
(src/server.py) reads the same path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


SIGNED_SPEC_ARGS = dict(
    # === THE SIGNED MANIFEST INPUTS (lead, 2026-08-17) — change = new manifest ===
    dataset_revision="bf5cdc1f26e581e519018e434db14edd1b77602b",
    splits=("validation",),
    languages=("as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or",
               "pa", "sa", "ta", "te", "ur"),
    # sizing gate (P90 <= 8h on A6000): full split = 97,941 rows/shard = ~16M chunks = 50h+,
    # INFEASIBLE. Max-fit = first 6000 rows/shard (query-aligned across shards -> max family
    # collapse). Documented as a sample of the validation split, never claimed as "full".
    max_rows_per_shard=6000,
    indexed_docs_target=0,            # whole-corpus-minus-holdout
    heldout_docs_target=10_000,
    size_tolerance=0.01,
    allocation_seed=20260817,
    min_heldout_queries=30,
    tokenizer_revision="5617a9f61b028005a4858fdac845db406aefb181",   # BGE-M3 repo SHA
    embed_revision="5617a9f61b028005a4858fdac845db406aefb181",
    raptor_cluster_size=15,           # summaries ~= leaves/13 -> P50 ~5.8h, P90 inside 8h gate
    collection_prefix="msmarco_xi_val14",
)

# _cluster's Lloyd step allocates an n x k sims matrix; beyond ~100k leaves per tree that
# matrix alone exceeds RAM (1M leaves -> ~400GB). Trees are therefore built per language
# group, sliced to this cap — semantically coherent (summaries stay single-language, matching
# the composer's same-language preference) and each tree runs at the scale the builder is
# proven at. Summaries are never cited, so slicing costs recall aid only at slice boundaries.
RAPTOR_GROUP_CAP = 100_000


def _raptor_groups(chunks, embeds, cap=RAPTOR_GROUP_CAP):
    by_lang: dict = {}
    for ch, e in zip(chunks, embeds):
        by_lang.setdefault(getattr(ch, "language", "und"), []).append((ch, e))
    for lang in sorted(by_lang):
        items = by_lang[lang]                      # corpus order -> deterministic slices
        for i in range(0, len(items), cap):
            yield lang, items[i:i + cap]


class PhaseLog:
    """Phase-level durability markers for the versioned build (authorization requirement).
    Each phase writes start/done/fail JSON under <build_dir>/phases/ carrying the manifest8,
    output counts, verification data, and wall time. A rerun consults done-markers + realized
    Qdrant counts to skip completed collections; a marker from a different manifest aborts."""

    def __init__(self, outdir: str, m8: str):
        import json as _json
        self._json = _json
        self.dir = os.path.join(outdir, "phases")
        os.makedirs(self.dir, exist_ok=True)
        self.m8 = m8
        self._t: dict = {}

    def _write(self, name: str, kind: str, obj: dict) -> None:
        obj = {"manifest8": self.m8, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **obj}
        with open(os.path.join(self.dir, f"{name}.{kind}.json"), "w", encoding="utf-8") as f:
            self._json.dump(obj, f, ensure_ascii=False, indent=1)

    def done_marker(self, name: str):
        p = os.path.join(self.dir, f"{name}.done.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            d = self._json.load(f)
        if d.get("manifest8") != self.m8:
            raise SystemExit(f"REFUSED: phase marker {name} carries manifest "
                             f"{d.get('manifest8')} != {self.m8}")
        return d

    def start(self, name: str, **info) -> None:
        self._t[name] = time.time()
        self._write(name, "start", info)
        print(f"[phase:{name}] start")

    def finish(self, name: str, output_count: int, **verify) -> None:
        wall = round(time.time() - self._t.get(name, time.time()), 1)
        self._write(name, "done", {"output_count": output_count, "wall_s": wall, **verify})
        print(f"[phase:{name}] done count={output_count} wall={wall}s")

    def fail(self, name: str, err: str) -> None:
        self._write(name, "fail", {"error": err[:2000]})
        print(f"[phase:{name}] FAIL {err[:200]}")


def _resources() -> dict:
    """Disk/RAM/VRAM snapshot for the guard + phase reports (Linux build box)."""
    import shutil as _sh
    import subprocess as _sp
    du = _sh.disk_usage("/")
    out = {"disk_free_gb": round(du.free / 1e9, 1), "disk_total_gb": round(du.total / 1e9, 1)}
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        out["ram_avail_gb"] = round(mi["MemAvailable"] / 1e6, 1)
        out["ram_total_gb"] = round(mi["MemTotal"] / 1e6, 1)
    except Exception:
        pass
    try:
        q = _sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        used, total = (int(x) for x in q.stdout.strip().split(",")[:2])
        out["vram_used_mb"], out["vram_total_mb"] = used, total
    except Exception:
        pass
    return out


def _guard(phases: PhaseLog, phase: str, start_free_gb: float) -> None:
    """Authorized stop conditions: free disk < 30% of start-free, RAM or VRAM headroom < 15%."""
    r = _resources()
    if r["disk_free_gb"] < 0.30 * start_free_gb:
        phases.fail(phase, f"STOP: disk free {r['disk_free_gb']}GB < 30% of start "
                           f"{start_free_gb}GB")
        raise SystemExit(3)
    if "ram_avail_gb" in r and r["ram_avail_gb"] < 0.15 * r["ram_total_gb"]:
        phases.fail(phase, f"STOP: RAM headroom {r['ram_avail_gb']}/{r['ram_total_gb']}GB < 15%")
        raise SystemExit(3)
    if "vram_total_mb" in r and (r["vram_total_mb"] - r["vram_used_mb"]) \
            < 0.15 * r["vram_total_mb"]:
        phases.fail(phase, f"STOP: VRAM headroom {r['vram_used_mb']}/{r['vram_total_mb']}MB < 15%")
        raise SystemExit(3)
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as resp:
            if resp.status != 200:
                raise OSError(f"health HTTP {resp.status}")
    except Exception as e:
        phases.fail(phase, f"STOP: live service unstable: {e}")
        raise SystemExit(3)
    print(f"[guard:{phase}] {r} live=ok")


def _collection_count(client, name: str):
    if not client.collection_exists(name):
        return None
    return client.count(name, exact=True).count


def _scroll_leaves(client, name: str, batch: int = 384):
    """Stream every point of a completed passage collection back (payload + named vectors) —
    the RAPTOR-resume checkpoint: leaves are re-used verbatim instead of re-embedding ~1M
    chunks. Yields raw scrolled points. Smaller batch + per-request retry: a heavy
    with_vectors scroll on a post-reboot-busy Qdrant timed out the earlier resume."""
    import time as _t
    offset = None
    while True:
        for attempt in range(5):
            try:
                pts, offset = client.scroll(name, limit=batch, offset=offset,
                                            with_payload=True, with_vectors=True)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"  scroll retry {attempt + 1}/4 ({type(e).__name__}) — Qdrant busy")
                _t.sleep(min(2 ** attempt, 15))
        for p in pts:
            yield p
        if offset is None:
            return


def build_versioned(args):
    """THE versioned production build (manifest-scoped, phase-durable):
      plan -> [passage: chunk+embed+upload] -> [raptor: build+upload] -> verify.
    Writes scope identity + partition manifests; sealed query ids are NEVER emitted in
    ordinary artifacts (counts + sha256 only; the sealed query files are chmod 600 and exist
    solely for eval/sealed_test.py). Never touches live config or non-manifest collections."""
    import hashlib
    import json
    import corpus_spec as cspec
    import index_build as ib
    from qdrant_client import QdrantClient
    from embeddings import BGEM3Embedder
    from raptor import ExtractSummarizer, RaptorTreeBuilder, load_hf_summarizer
    from retrieval import Retriever, point_id
    from topics import EmbeddingTopicClassifier, apply_topic_labels
    import chunking as ck

    t0 = time.time()
    spec = cspec.CorpusBuildSpec(
        shards=tuple(f"validation/{ib._LANG_FILE[c]}val.parquet"
                     for c in SIGNED_SPEC_ARGS["languages"]),
        **SIGNED_SPEC_ARGS)
    spec.validate()
    m8 = spec.manifest8()
    coll_passage = spec.collection_name("passage")
    coll_raptor = spec.collection_name("raptor")
    print(f"MANIFEST {m8} — collections {coll_passage}, {coll_raptor}")

    outdir = os.path.abspath(args.plan_dir or os.path.join(
        os.path.dirname(__file__), "..", "eval", "builds", m8))
    os.makedirs(outdir, exist_ok=True)
    phases = PhaseLog(outdir, m8)
    start_free_gb = _resources()["disk_free_gb"]

    def _sha(obj) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                         ensure_ascii=True).encode()).hexdigest()

    def _dump(name, obj, mode=0o644):
        p = os.path.join(outdir, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.chmod(p, mode)
        return p

    # ---- phase: plan --------------------------------------------------------------------
    phases.start("plan")

    def shard_gen(cfg):
        for i, row in enumerate(ib._iter_msmarco_rows(
                cfg, split="validation", revision=spec.dataset_revision)):
            if spec.max_rows_per_shard and i >= spec.max_rows_per_shard:
                break
            yield row

    plan = ib.plan_split_multi([(c, shard_gen(c)) for c in spec.languages], spec)
    r = plan.realization
    prior_plan = phases.done_marker("plan")
    if prior_plan and prior_plan.get("corpus_hash") != r.corpus_hash:
        phases.fail("plan", f"STOP: replanned corpus_hash {r.corpus_hash[:16]} != prior "
                            f"{prior_plan.get('corpus_hash', '')[:16]} (dataset/spec drift)")
        raise SystemExit(3)

    qr = plan.qrels
    part_summary, qfile_hashes = {}, {}
    for pool, part in (("incorpus", plan.incorpus_partition),
                       ("absent", plan.absent_partition)):
        part_summary[pool] = {}
        for pname in ("calibration", "dev", "sealed"):
            qids = getattr(part, pname)
            payload = {q: {"text": qr.query_text.get(q, ""),
                           "positives": sorted(qr.positives.get(q, ())),
                           "group": qr.group_of.get(q, q)} for q in qids}
            sealed = pname == "sealed"
            fname = f"queries_{pool}_{pname}.json"
            _dump(fname, payload, mode=0o600 if sealed else 0o644)
            qfile_hashes[fname] = _sha(payload)
            # sealed ids never appear in ordinary artifacts: counts + digests only
            part_summary[pool][pname] = ({"count": len(qids), "sha256_qids": _sha(sorted(qids))}
                                         if sealed else sorted(qids))
    scope = {
        "manifest": m8,
        "scope_name": (f"msmarco_xi_validation_sample_{len(spec.languages)}lang_"
                       f"{spec.max_rows_per_shard}_rows_per_shard"),
        "scope_type": "deterministic_validation_split_sample",
        "source_languages": len(spec.languages),
        "rows_per_language_shard": spec.max_rows_per_shard,
        "source_rows": r.source_rows, "source_passages": r.source_passages,
        "unique_canonical_docs": r.unique_canonical,
        "indexed_docs": r.indexed_docs, "indexed_families": r.indexed_families,
        "heldout_docs": r.heldout_docs, "heldout_families": r.heldout_families,
        "excluded_queries": r.excluded_queries,
        "public_wording": ("Deterministic validation-split sample: 84k source rows across "
                           "all 14 languages, producing 876k indexed canonical documents and "
                           "a leakage-disjoint 10.1k-document absent-evidence holdout."),
    }
    _dump("realization.json", {
        "manifest8": m8, "scope": scope, "spec": spec.to_canonical_dict(),
        "realization": r.to_dict(), "integrity": plan.integrity,
        "incorpus_partition": part_summary["incorpus"],
        "absent_partition": part_summary["absent"],
        "query_file_sha256": qfile_hashes})
    phases.finish("plan", r.indexed_docs, corpus_hash=r.corpus_hash,
                  integrity=plan.integrity, heldout_docs=r.heldout_docs)
    if args.plan_only:
        print("PLAN-ONLY: no build performed.")
        return

    # ---- phase: passage (chunk + embed + upload as one durable unit) --------------------
    # timeout=120: a background optimizer stall must slow us down, never kill us
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    emb = BGEM3Embedder(batch_size=args.batch)
    retr = Retriever(client, emb, use_sparse=True)

    def _set_indexing(name, threshold):
        """Bulk-ingest practice: indexing_threshold=0 defers HNSW while points pour in
        (the optimizer competing with ingest is what stalled upserts); restored after the
        last bulk phase so the index builds once, in the background."""
        from qdrant_client import models as qm
        try:
            client.update_collection(collection_name=name,
                                     optimizers_config=qm.OptimizersConfigDiff(
                                         indexing_threshold=threshold))
            print(f"[indexing] {name} indexing_threshold={threshold}")
        except Exception as e:
            print(f"[indexing] WARN could not set threshold on {name}: {e}")

    _guard(phases, "passage", start_free_gb)

    pass_done = phases.done_marker("passage")
    realized = _collection_count(client, coll_passage)
    chunks = dense_all = None
    if pass_done and realized == pass_done["output_count"]:
        print(f"[phase:passage] SKIP — complete at {realized} points (marker matches)")
    else:
        if realized is not None:
            # partial collection from an aborted run of THIS manifest: delete, restart phase
            assert m8 in coll_passage
            print(f"[phase:passage] partial collection ({realized} pts, marker="
                  f"{bool(pass_done)}) -> delete + rebuild")
            client.delete_collection(coll_passage)
        phases.start("passage")
        import numpy as np
        docs = ib.documents_from_allocation(plan.allocation, plan.family_index, spec)
        # the family graph + qrels are multi-GB of strings and no longer needed in RAM (all
        # partition/qrel files are on disk; `r` keeps the realization). The 2026-08-16 OOM at
        # 731k/959k chunks was accumulated Python-object overhead — everything below streams.
        del plan, qr
        chunks = ck.PassageAwareChunker().chunk_corpus(docs)
        del docs
        n = len(chunks)
        print(f"[phase:passage] {n} chunks; block-streaming embed+upload...")
        dense_all = np.empty((n, emb.dim), dtype=np.float16)   # 959k x 1024 fp16 ~= 2.0 GB
        retr.ensure_collection(coll_passage)
        _set_indexing(coll_passage, 0)          # defer HNSW until all bulk phases finish
        B = 8192
        up = 0
        for i in range(0, n, B):
            block = chunks[i:i + B]
            ers = emb.embed_docs([c.text for c in block])
            for j, er in enumerate(ers):
                dense_all[i + j] = er.dense
            up += retr.index(coll_passage, block, ers)   # upsert NOW; sparse never retained
            del ers
            if (i // B) % 10 == 0:
                rss = 0
                try:
                    with open("/proc/self/status") as f:
                        rss = next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) // 1e6
                except Exception:
                    pass
                print(f"  embedded+uploaded {min(i + B, n)}/{n} rss={rss:.1f}GB "
                      f"at {(time.time() - t0)/60:.1f}min")
                if rss > 45:
                    phases.fail("passage", f"STOP: build RSS {rss:.1f}GB — memory envelope "
                                           f"exceeded, aborting before system pressure")
                    raise SystemExit(3)
        # topic labels ride a payload-update pass (points were upserted topic-less).
        # 100k slices: classify_batch's list-comp would otherwise re-box all 959k rows
        # (~8GB transient); topic names come from the fixed keyword vocabulary, so
        # slice-wise fitting keeps a consistent label set.
        clf = EmbeddingTopicClassifier(n_topics=12)
        for s in range(0, n, 100_000):
            apply_topic_labels(chunks[s:s + 100_000], dense_all[s:s + 100_000], clf)
        by_topic: dict = {}
        for ch in chunks:
            t = (ch.extra or {}).get("topic")
            if t:
                by_topic.setdefault(t, []).append(point_id(ch.chunk_id))
        for t, ids in by_topic.items():
            for k in range(0, len(ids), 4096):
                client.set_payload(coll_passage, payload={"topic": t},
                                   points=ids[k:k + 4096])
        phases.finish("passage", up, collection=coll_passage,
                      qdrant_count=_collection_count(client, coll_passage))

    # ---- phase: raptor (summaries interleave gen+embed level-wise; then upload) ---------
    _guard(phases, "raptor", start_free_gb)
    rap_done = phases.done_marker("raptor")
    rap_realized = _collection_count(client, coll_raptor)
    if rap_done and rap_realized == rap_done["output_count"]:
        print(f"[phase:raptor] SKIP — complete at {rap_realized} points (marker matches)")
    else:
        if rap_realized is not None:
            assert m8 in coll_raptor
            print(f"[phase:raptor] partial collection ({rap_realized} pts) -> delete + rebuild")
            client.delete_collection(coll_raptor)
        phases.start("raptor")
        import numpy as np
        from qdrant_client import models as qm
        from types import SimpleNamespace
        # leaves ALWAYS copy verbatim from the completed passage collection (payload + dense
        # + sparse pass through untouched) — one scroll, no re-embedding, no sparse in RAM.
        # On a resume (chunks is None) the same scroll also rebuilds the lite chunk list and
        # the compact fp16 dense matrix the summary builder needs.
        from retrieval import upsert_with_retry
        retr.ensure_collection(coll_raptor)
        _set_indexing(coll_raptor, 0)           # defer HNSW during the bulk leaf copy
        resume = chunks is None
        lite, dvecs, buf, copied = [], [], [], 0
        for p in _scroll_leaves(client, coll_passage):
            buf.append(qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload))
            if len(buf) >= 256:
                upsert_with_retry(client, coll_raptor, buf)
                copied += len(buf)
                buf = []
            if resume:
                pay = p.payload or {}
                lite.append(SimpleNamespace(text=pay.get("text", ""),
                                            chunk_id=pay.get("chunk_id", ""),
                                            language=pay.get("language", "und")))
                v = p.vector["dense"] if isinstance(p.vector, dict) else p.vector
                dvecs.append(np.asarray(v, dtype=np.float16))   # compact immediately
        if buf:
            upsert_with_retry(client, coll_raptor, buf)
            copied += len(buf)
        print(f"[phase:raptor] {copied} leaves copied into {coll_raptor} "
              f"at {(time.time() - t0)/60:.1f}min")
        if resume:
            leaf_chunks, leaf_vecs = lite, np.stack(dvecs) if dvecs else np.empty((0, emb.dim))
            del dvecs
        else:
            leaf_chunks, leaf_vecs = chunks, dense_all

        if args.llm_raptor:
            # batch 32: c=15 prompts are ~3x the c=5 baseline (2.6/s measured at batch 16);
            # larger batches amortize prefill -> ~1.5x, fitting gen inside the window
            summarizer, cleanup = load_hf_summarizer(spec.raptor_summarizer, batch=32)
        else:
            summarizer, cleanup = ExtractSummarizer(), None
        builder = RaptorTreeBuilder(emb, summarizer, cluster_size=spec.raptor_cluster_size,
                                    max_levels=spec.raptor_max_levels, build_manifest=m8)
        summaries, s_embeds = [], []
        for lang, group in _raptor_groups(leaf_chunks, leaf_vecs):
            if time.time() - t0 > args.window_hours * 3600:
                phases.fail("raptor", f"STOP: elapsed beyond the authorized window "
                                      f"({args.window_hours}h)")
                raise SystemExit(3)
            g_sum, g_emb = builder.build([c for c, _ in group], [e for _, e in group])
            summaries.extend(g_sum)
            s_embeds.extend(g_emb)
            print(f"  raptor[{lang}] {len(group)} leaves -> {len(g_sum)} summaries "
                  f"at {(time.time() - t0)/60:.1f}min")
        if cleanup:
            cleanup()
        n_sum = retr.index(coll_raptor, summaries, s_embeds)
        # all bulk ingest done — let HNSW build once, in the background (verify/exhaustive
        # verification use scroll+count only; calibration later waits for index green)
        _set_indexing(coll_passage, 20_000)
        _set_indexing(coll_raptor, 20_000)
        phases.finish("raptor", _collection_count(client, coll_raptor),
                      collection=coll_raptor, summaries=len(summaries),
                      cluster_size=spec.raptor_cluster_size)

    # ---- phase: verify (counts reconcile, schema sample, qrel spot-resolve) -------------
    phases.start("verify")
    from qdrant_client import models as qm
    v: dict = {"passage_count": _collection_count(client, coll_passage),
               "raptor_count": _collection_count(client, coll_raptor)}
    pass_marker = phases.done_marker("passage")
    rap_marker = phases.done_marker("raptor")
    ok = (pass_marker and v["passage_count"] == pass_marker["output_count"]
          and rap_marker and v["raptor_count"] == rap_marker["output_count"]
          and v["raptor_count"] > v["passage_count"])
    # sent == realized: upsert-id collisions are SILENT (last writer wins), and the marker
    # count is read back from the collection so it can't see them. This one-line identity
    # is what would have caught the 73ca3e90 summary collision (60k sent -> 6,833 realized).
    sent = (rap_marker or {}).get("summaries")
    v["summaries_sent"] = sent
    v["summaries_realized"] = v["raptor_count"] - v["passage_count"]
    if sent is not None and sent != v["summaries_realized"]:
        ok = False
        v["summary_collision"] = f"sent {sent} != realized {v['summaries_realized']}"
    sample, _ = client.scroll(coll_raptor, limit=200, with_payload=True)
    leaves = [p for p in sample if not p.payload.get("is_summary")]
    sums = [p for p in sample if p.payload.get("is_summary")]
    v["schema_leaf_sample"] = len(leaves)
    v["schema_summary_ok"] = all(
        p.payload.get("extra", {}).get("leaf_descendants") and
        p.payload.get("extra", {}).get("build_manifest") == m8 for p in sums) if sums else None
    cal = json.load(open(os.path.join(outdir, "queries_incorpus_calibration.json"),
                         encoding="utf-8"))
    import random as _rnd
    rng = _rnd.Random(20260817)
    probe = rng.sample(sorted(cal), min(200, len(cal)))
    unresolved = 0
    for qid in probe:
        sdid = cal[qid]["positives"][0]
        hits, _ = client.scroll(coll_passage, limit=1, scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="stable_doc_id", match=qm.MatchValue(value=sdid))]))
        if not hits:
            unresolved += 1
    v["qrel_probe"] = {"n": len(probe), "unresolved": unresolved}
    ok = ok and unresolved == 0
    v["result"] = "PASS" if ok else "FAIL"
    phases.finish("verify", v["raptor_count"] or 0, **v)
    _dump("build_verification.json", v)
    print(f"\nVERSIONED BUILD {m8} {'VERIFIED' if ok else 'VERIFY-FAIL'} "
          f"in {(time.time() - t0)/3600:.2f}h — live config untouched; promotion is manual.")


def build_tier2(args):
    """TIER-2 SCALE INDEX (user-approved 2026-08-16): the FULL validation split as a
    passage-only, dense-only, int8-quantized collection — a latency-verified scale
    demonstration. NO sealed claims, NO qrels, NO RAPTOR (106h at measured rates).
    Exact-SHA dedup only (SimHash families are a tier-1 property, documented). Excludes
    tier-1's heldout stable_doc_ids (re-derived deterministically) so absent-evidence
    behavior stays consistent when the UI switches tiers. Single streaming pass:
    shard -> occurrences -> dedup/exclude -> chunk -> embed -> upsert (topic labels
    skipped at this tier, documented)."""
    import json
    import corpus_spec as cspec
    import index_build as ib
    from qdrant_client import QdrantClient, models as qm
    from embeddings import BGEM3Embedder
    from retrieval import point_id, upsert_with_retry
    import chunking as ck

    t0 = time.time()
    name = "msmarco_xi_valfull__passage"
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval",
                                          "builds", "tier2_valfull"))
    os.makedirs(outdir, exist_ok=True)
    phases = PhaseLog(outdir, "tier2")
    start_free_gb = _resources()["disk_free_gb"]

    # tier-1 heldout exclusion set, re-derived deterministically (~3.3 min)
    phases.start("exclusions")
    spec = cspec.CorpusBuildSpec(
        shards=tuple(f"validation/{ib._LANG_FILE[c]}val.parquet"
                     for c in SIGNED_SPEC_ARGS["languages"]),
        **SIGNED_SPEC_ARGS)

    def capped_gen(cfg):
        for i, row in enumerate(ib._iter_msmarco_rows(
                cfg, split="validation", revision=spec.dataset_revision)):
            if i >= spec.max_rows_per_shard:
                break
            yield row

    plan = ib.plan_split_multi([(c, capped_gen(c)) for c in spec.languages], spec)
    heldout = {int(s[:16], 16) for s in
               plan.allocation.heldout_stable_doc_ids(plan.family_index)}
    phases.finish("exclusions", len(heldout))
    del plan

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    if client.collection_exists(name):
        n0 = client.count(name, exact=True).count
        print(f"[tier2] collection exists with {n0} points -> delete + rebuild")
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE)},
        quantization_config=qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
            type=qm.ScalarType.INT8, quantile=0.99, always_ram=False)),
        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=0))
    emb = BGEM3Embedder(batch_size=args.batch)
    chunker = ck.PassageAwareChunker()

    phases.start("stream")
    seen: set = set()
    stats = {"rows": 0, "occurrences": 0, "unique_docs": 0, "excluded_heldout": 0,
             "chunks": 0}
    docbuf: list = []

    def flush():
        if not docbuf:
            return
        chunks = chunker.chunk_corpus(docbuf)
        ers = emb.embed_docs([c.text for c in chunks])
        buf = []
        for ch, er in zip(chunks, ers):
            buf.append(qm.PointStruct(id=point_id(ch.chunk_id),
                                      vector={"dense": er.dense},
                                      payload=ch.to_payload()))
            if len(buf) >= 256:
                upsert_with_retry(client, name, buf)
                buf = []
        if buf:
            upsert_with_retry(client, name, buf)
        stats["chunks"] += len(chunks)
        docbuf.clear()

    for cfg in spec.languages:
        n_rows = 0
        for idx, row in enumerate(ib._iter_msmarco_rows(
                cfg, split="validation", revision=spec.dataset_revision)):
            n_rows = idx + 1
            for occ in ib.iter_passage_occurrences(
                    [row], spec.include_english, spec.include_translated,
                    cfg=cfg, row_prefix=f"{cfg}:{idx}:"):
                stats["occurrences"] += 1
                key = int(occ.stable_doc_id[:16], 16)
                if key in seen:
                    continue
                if key in heldout:
                    stats["excluded_heldout"] += 1
                    continue
                seen.add(key)
                stats["unique_docs"] += 1
                docbuf.append(ck.Document(doc_id=f"{occ.lang}-{occ.stable_doc_id}",
                                          text=occ.text, language=occ.lang,
                                          stable_doc_id=occ.stable_doc_id))
                if len(docbuf) >= 8192:
                    flush()
            if idx % 10_000 == 0:
                r = _resources()
                print(f"[tier2:{cfg}] row {idx} uniq={stats['unique_docs']} "
                      f"chunks={stats['chunks']} disk={r['disk_free_gb']}GB "
                      f"at {(time.time() - t0)/60:.1f}min")
                if r["disk_free_gb"] < max(20.0, 0.25 * start_free_gb):
                    phases.fail("stream", f"STOP: disk {r['disk_free_gb']}GB")
                    raise SystemExit(3)
                if r.get("ram_avail_gb", 99) < 0.15 * r.get("ram_total_gb", 64):
                    phases.fail("stream", "STOP: RAM headroom < 15%")
                    raise SystemExit(3)
        stats["rows"] += n_rows
        flush()
        print(f"[tier2] shard {cfg} done: rows={n_rows} uniq={stats['unique_docs']} "
              f"chunks={stats['chunks']} at {(time.time() - t0)/60:.1f}min")
    flush()
    phases.finish("stream", stats["chunks"], **{k: v for k, v in stats.items()
                                                if k != "chunks"})

    client.update_collection(collection_name=name,
                             optimizers_config=qm.OptimizersConfigDiff(
                                 indexing_threshold=20_000))
    total = client.count(name, exact=True).count
    real = {"tier": 2, "collection": name, "config": {
                "dense_only": True, "quantization": "int8-scalar q0.99",
                "dedup": "exact stable_doc_id only (SimHash families = tier 1 only)",
                "topics": "skipped at scale tier",
                "excluded_tier1_heldout": stats["excluded_heldout"]},
            "stats": stats, "qdrant_points": total,
            "wall_hours": round((time.time() - t0) / 3600, 2),
            "label": ("scale tier: latency-verified; accuracy/abstention evidence "
                      "lives on the sealed tier (manifest 73ca3e90)")}
    with open(os.path.join(outdir, "tier2_realization.json"), "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=1)
    print(f"\nTIER-2 SCALE INDEX DONE: {total} points in {real['wall_hours']}h "
          f"-> {outdir}/tier2_realization.json — never promoted without explicit config.")


def main():
    ap = argparse.ArgumentParser(description="Build MSMARCO-XI index on a local GPU")
    ap.add_argument("--max-docs", type=int, default=20_000)
    ap.add_argument("--languages", type=str, default="hi",
                    help="comma-separated MSMARCO-XI config codes (hi,ta,bn,...)")
    ap.add_argument("--qdrant-path", type=str, default="./qdrant_data")
    ap.add_argument("--qdrant-url", type=str, default=None,
                    help="Qdrant SERVER url (e.g. http://localhost:6333) — HNSW, "
                         "concurrent access; overrides --qdrant-path")
    ap.add_argument("--llm-raptor", dest="llm_raptor", action="store_true", default=True)
    ap.add_argument("--no-llm-raptor", dest="llm_raptor", action="store_false",
                    help="use extractive cluster summaries (no 3B load, much faster)")
    ap.add_argument("--include-semantic", action="store_true", default=False,
                    help="also build the semantic-chunker collection (slow: per-doc embeds)")
    ap.add_argument("--batch", type=int, default=64, help="embedding batch size")
    ap.add_argument("--versioned", action="store_true", default=False,
                    help="THE production build: SIGNED_SPEC_ARGS -> leakage-safe plan -> "
                         "versioned msmarco_xi_val14_<manifest8>__* collections + manifests")
    ap.add_argument("--plan-only", action="store_true", default=False,
                    help="with --versioned: plan + write manifests, build nothing")
    ap.add_argument("--plan-dir", type=str, default=None,
                    help="override the manifest output dir (two-run reproducibility checks)")
    ap.add_argument("--window-hours", type=float, default=8.5,
                    help="authorized wall-clock window; the raptor loop stops beyond it "
                         "(set per the human authorization for this launch)")
    ap.add_argument("--tier2", action="store_true", default=False,
                    help="TIER-2 scale index: full validation split, passage-only, "
                         "dense-only int8-quantized (user-approved 2026-08-16)")
    args = ap.parse_args()

    if args.tier2:
        build_tier2(args)
        return
    if args.versioned:
        build_versioned(args)
        return

    from qdrant_client import QdrantClient
    from embeddings import BGEM3Embedder
    from index_build import build_all_strategies, build_raptor_index, load_msmarco_slice
    from raptor import ExtractSummarizer, load_hf_summarizer
    from topics import EmbeddingTopicClassifier

    t0 = time.time()
    langs = tuple(x.strip() for x in args.languages.split(",") if x.strip())
    print(f"loading MSMARCO-XI slice: {langs}, max_docs={args.max_docs}...")
    docs = load_msmarco_slice(languages=langs, max_docs=args.max_docs)
    print(f"  {len(docs)} docs loaded in {time.time() - t0:.0f}s")

    print("loading BGE-M3...")
    emb = BGEM3Embedder(batch_size=args.batch)
    client = (QdrantClient(url=args.qdrant_url) if args.qdrant_url
              else QdrantClient(path=args.qdrant_path))

    topic_clf = EmbeddingTopicClassifier(n_topics=12)
    report = build_all_strategies(docs, emb, client, prefix="msmarco_xi",
                                  include_semantic=args.include_semantic,
                                  topic_classifier=topic_clf)
    print(f"strategies built at {time.time() - t0:.0f}s: "
          f"{ {k: v['n_chunks'] for k, v in report.items()} }")

    cleanup = None
    if args.llm_raptor:
        print("loading 3B for RAPTOR abstractive summaries...")
        summarizer, cleanup = load_hf_summarizer()
    else:
        summarizer = ExtractSummarizer()

    raptor_report = build_raptor_index(
        docs, emb, summarizer, client, prefix="msmarco_xi",
        strategy_name="passage", cluster_size=5, max_levels=3)
    report["raptor"] = raptor_report
    print("raptor:", raptor_report)
    if cleanup is not None:
        cleanup()

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. "
          f"Collections at {args.qdrant_url or args.qdrant_path}:")
    for k, v in report.items():
        print(f"  {v['collection']}: {v['n_chunks']} chunks")


if __name__ == "__main__":
    main()
