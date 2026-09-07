"""P0.3 leaf-evidence semantics — summaries NAVIGATE, leaves are EVIDENCE.
Exit gates: summaries never generate/cite; summary-only retrieval cannot admit an answer;
expansion yields stable leaf evidence; empty expansion abstains; PageIndex returns leaves."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck                                    # noqa: E402
from api import build_local_harness                      # noqa: E402
from embeddings import HashEmbedder                      # noqa: E402
from index_build import build_raptor_index, synthetic_docs  # noqa: E402
from raptor import ExtractSummarizer, RaptorTreeBuilder  # noqa: E402
from retrieval import (Retriever, expand_to_leaves, is_summary_payload,  # noqa: E402
                       memory_client)
from schemas import Decision, Query, RetrievedChunk      # noqa: E402


def _raptor_setup(n=30):
    emb = HashEmbedder(dim=256)
    client = memory_client()
    report = build_raptor_index(synthetic_docs(n), emb, ExtractSummarizer(), client,
                                prefix="lev", strategy_name="passage",
                                cluster_size=5, max_levels=3)
    retr = Retriever(client, emb, use_sparse=True)
    return retr, report["collection"]


def test_is_summary_detector_legacy_and_contract():
    assert is_summary_payload({"strategy": "raptor"})            # legacy marker
    assert is_summary_payload({"is_summary": True})              # new contract
    assert not is_summary_payload({"is_summary": False, "strategy": "passage"})
    assert not is_summary_payload({"strategy": "passage"})


def test_expansion_replaces_summaries_with_leaves():
    retr, coll = _raptor_setup()
    hits = retr.search(coll, "qdrant vector database", top_k=20)
    expanded = expand_to_leaves(retr, coll, hits, top_k=20)
    assert expanded, "expansion must yield leaf evidence on a healthy tree"
    assert all(not is_summary_payload(r.payload) for r in expanded), \
        "no summary may survive expansion"


def test_summary_only_hits_expand_to_leaves():
    retr, coll = _raptor_setup()
    all_hits = retr.search(coll, "qdrant vector database", top_k=40)
    only_summaries = [h for h in all_hits if is_summary_payload(h.payload)]
    assert only_summaries, "test needs at least one summary hit"
    expanded = expand_to_leaves(retr, coll, only_summaries, top_k=10)
    assert expanded, "summary-only retrieval must resolve to leaf descendants"
    assert all(not is_summary_payload(r.payload) for r in expanded)


def test_unresolvable_summary_yields_empty():
    retr, coll = _raptor_setup()
    orphan = RetrievedChunk(chunk_id="ghost::raptor::999", text="fake summary",
                            score=1.0, payload={"strategy": "raptor",
                                                "extra": {"children": ["nope::x::1"]}})
    expanded = expand_to_leaves(retr, coll, [orphan], top_k=8)
    assert expanded == [], "unresolvable summaries must yield EMPTY, never themselves"


def test_harness_never_cites_summaries():
    h = build_local_harness()   # local harness uses the raptor-backed pageindex + passage coll
    for q in ("what is qdrant", "python programming language", "goa beaches"):
        r = h.answer(Query(text=q))
        if r.decision == Decision.ANSWER:
            for c in r.citations:
                assert "raptor" not in (c.doc_id or ""), f"summary cited: {c.chunk_id}"
                assert "::raptor::" not in c.chunk_id, f"summary cited: {c.chunk_id}"


def test_pageindex_returns_leaf_evidence():
    from pageindex import PageIndexTree
    emb = HashEmbedder(dim=256)
    chunks = ck.PassageAwareChunker(min_tokens=1, max_tokens=200).chunk_corpus(
        synthetic_docs(30))
    embeds = emb.embed_docs([c.text for c in chunks])
    summaries, _ = RaptorTreeBuilder(emb, ExtractSummarizer(),
                                     cluster_size=5, max_levels=3).build(chunks, embeds)
    tree = PageIndexTree.from_nodes(chunks, summaries)
    results = tree.search("qdrant vector database", top_k=8)
    assert results
    assert all("::raptor::" not in r.chunk_id for r in results), \
        "PageIndex must return leaf evidence only"


def test_broad_summary_heavy_query_never_errors():
    """LIVE BUG 2026-08-15: 'summarise the whole dataset…' matched many summaries; legacy
    per-summary recursive resolution blew the deadline; unwrapped leaf_expand raised ⇒ ERROR."""
    retr, coll = _raptor_setup()
    from harness import RAGHarness
    from generation import ExtractiveGenerator
    from guardrails import KeywordSafety
    h = RAGHarness(embedder=HashEmbedder(dim=256), retriever=retr,
                   generator=ExtractiveGenerator(), safety=KeywordSafety(),
                   collection=coll, ood_threshold=0.15, budget_ms=60,
                   stage_timeouts={"leaf_expand": 1})   # 1ms: force expansion timeout
    r = h.answer(Query(text="summarise the whole dataset and retrieve the executive summary"))
    assert r.decision != Decision.ERROR, f"expansion failure must degrade, got ERROR: {r.reason}"
    if r.decision == Decision.ANSWER:
        assert all("::raptor::" not in c.chunk_id for c in r.citations)


def test_batched_resolution_is_bounded():
    """Legacy resolution must be level-batched: ≤ max_depth fetch calls regardless of
    how many summaries are in the candidate set."""
    retr, coll = _raptor_setup()
    hits = retr.search(coll, "document number qdrant goa python sarvam rag", top_k=40)
    calls = {"n": 0}
    orig = retr.fetch_by_chunk_ids

    def counting(name, ids):
        calls["n"] += 1
        return orig(name, ids)

    retr.fetch_by_chunk_ids = counting
    try:
        out = expand_to_leaves(retr, coll, hits, top_k=40)
    finally:
        retr.fetch_by_chunk_ids = orig
    assert calls["n"] <= 4, f"resolution must batch per level, made {calls['n']} fetches"
    assert all(not is_summary_payload(r.payload) for r in out)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} leaf-evidence tests passed")
