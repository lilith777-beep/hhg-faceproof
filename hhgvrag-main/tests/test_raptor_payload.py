"""
test_raptor_payload.py — P0.3 RAPTOR payload contract (Opus Data side).

The approved contract: every point carries top-level `is_summary`; summaries additionally carry
extra.{raptor_level, children, leaf_descendants (FULL transitive closure of leaf chunk_ids),
build_manifest}; leaves carry is_summary=False + their stable_doc_id. The lead's
retrieval.expand_to_leaves consumes leaf_descendants first — these tests prove the closure is
correct, complete, and leaf-only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck                                    # noqa: E402
from embeddings import HashEmbedder                      # noqa: E402
from index_build import build_raptor_index, synthetic_docs  # noqa: E402
from raptor import ExtractSummarizer, RaptorTreeBuilder, _leaf_closure  # noqa: E402
from retrieval import (Retriever, memory_client, expand_to_leaves,  # noqa: E402
                       is_summary_payload)


def _tree(n=40, manifest="deadbeef", levels=3):
    emb = HashEmbedder(dim=256)
    chunks = ck.PassageAwareChunker(min_tokens=1, max_tokens=200).chunk_corpus(synthetic_docs(n))
    embeds = emb.embed_docs([c.text for c in chunks])
    summaries, s_embeds = RaptorTreeBuilder(emb, ExtractSummarizer(), cluster_size=5,
                                            max_levels=levels, build_manifest=manifest).build(
        chunks, embeds)
    return emb, chunks, embeds, summaries, s_embeds


# ---- leaf-closure unit ------------------------------------------------------------------
def test_leaf_closure_dedup_and_order():
    leaf_desc = {"a": ["l1", "l2"], "b": ["l2", "l3"], "l4": ["l4"]}
    out = _leaf_closure(["a", "b", "l4"], leaf_desc)
    assert out == ["l1", "l2", "l3", "l4"], "closure is first-seen, deduped, ordered"


def test_leaf_closure_unknown_child_is_own_leaf():
    assert _leaf_closure(["ghost"], {}) == ["ghost"]   # defensive fallback


# ---- payload contract -------------------------------------------------------------------
def test_summaries_carry_full_contract():
    _, chunks, _, summaries, _ = _tree()
    leaf_ids = {c.chunk_id for c in chunks}
    for s in summaries:
        assert s.is_summary is True
        pl = s.to_payload()
        assert pl["is_summary"] is True, "is_summary lifted to top level"
        assert "raptor_level" in s.extra and s.extra["raptor_level"] >= 1
        assert s.extra["children"], "summary must record immediate children"
        ld = s.extra["leaf_descendants"]
        assert ld, "leaf_descendants must be non-empty"
        assert all(l in leaf_ids for l in ld), "leaf_descendants must be REAL leaf chunk_ids"
        assert len(ld) == len(set(ld)), "leaf_descendants deduped"
        assert s.extra["build_manifest"] == "deadbeef", "lineage manifest stamped"


def test_leaves_carry_is_summary_false_and_stable_id():
    docs = synthetic_docs(6)
    docs[0].stable_doc_id = "sid-xyz"
    chunks = ck.PassageAwareChunker(min_tokens=1, max_tokens=200).chunk(docs[0])
    for c in chunks:
        pl = c.to_payload()
        assert pl["is_summary"] is False
        assert pl["stable_doc_id"] == "sid-xyz"


def test_closure_equals_recursive_union():
    """leaf_descendants (fast path) must equal what recursive children-resolution would produce
    (the legacy fallback) — so both harness paths agree."""
    _, chunks, _, summaries, _ = _tree()
    leaf_ids = {c.chunk_id for c in chunks}
    by_id = {s.chunk_id: s for s in summaries}

    def recursive(node_id, depth=8):
        if node_id in leaf_ids:
            return {node_id}
        s = by_id.get(node_id)
        if s is None or depth <= 0:
            return set()
        out = set()
        for ch in s.extra["children"]:
            out |= recursive(ch, depth - 1)
        return out

    for s in summaries:
        assert set(s.extra["leaf_descendants"]) == recursive(s.chunk_id), \
            "fast-path closure must equal recursive union"


def test_closure_covers_every_reachable_leaf():
    """No leaf reachable through a summary's subtree is missing from its leaf_descendants."""
    _, chunks, _, summaries, _ = _tree(n=50)
    by_id = {s.chunk_id: s for s in summaries}
    for s in summaries:
        if s.extra["raptor_level"] >= 2:
            union = set()
            for ch in s.extra["children"]:
                union |= set(by_id[ch].extra["leaf_descendants"]) if ch in by_id else {ch}
            assert set(s.extra["leaf_descendants"]) == union


# ---- integration with the lead's expansion ---------------------------------------------
def test_expand_to_leaves_uses_leaf_descendants_primary_path():
    emb, chunks, embeds, summaries, s_embeds = _tree()
    client = memory_client()
    retr = Retriever(client, emb, use_sparse=True)
    retr.index("rp", list(chunks) + summaries, list(embeds) + s_embeds)
    hits = retr.search("rp", "qdrant vector database", top_k=30)
    only_summaries = [h for h in hits if is_summary_payload(h.payload)]
    assert only_summaries, "need summary hits to test expansion"
    expanded = expand_to_leaves(retr, "rp", only_summaries, top_k=12)
    assert expanded, "summary hits must resolve to leaves via leaf_descendants"
    assert all(not is_summary_payload(r.payload) for r in expanded)
    leaf_ids = {c.chunk_id for c in chunks}
    assert all(r.chunk_id in leaf_ids for r in expanded), "expansion yields only indexed leaves"


def test_build_raptor_index_threads_manifest():
    emb = HashEmbedder(dim=256)
    client = memory_client()
    report = build_raptor_index(synthetic_docs(30), emb, ExtractSummarizer(), client,
                                prefix="rp2", strategy_name="passage",
                                cluster_size=5, max_levels=3, build_manifest="cafebabe")
    pts, _ = client.scroll(report["collection"], limit=1000, with_payload=True)
    summ = [p for p in pts if p.payload.get("is_summary")]
    assert summ, "indexed collection must contain summaries"
    assert all(p.payload["extra"]["build_manifest"] == "cafebabe" for p in summ)
    assert all(p.payload["extra"].get("leaf_descendants") for p in summ)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} RAPTOR payload-contract tests passed")
