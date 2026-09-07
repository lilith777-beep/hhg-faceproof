"""Vectorless PageIndex retrieval — tree build, beam descent, harness integration."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck                              # noqa: E402
from api import build_local_harness                # noqa: E402
from embeddings import HashEmbedder                # noqa: E402
from index_build import synthetic_docs             # noqa: E402
from pageindex import PageIndexTree                # noqa: E402
from raptor import ExtractSummarizer, RaptorTreeBuilder  # noqa: E402
from retrieval import Retriever, memory_client     # noqa: E402
from schemas import Decision, Query                # noqa: E402


def _tree(n_docs=40):
    emb = HashEmbedder(dim=256)
    chunks = ck.PassageAwareChunker(min_tokens=1, max_tokens=200).chunk_corpus(
        synthetic_docs(n_docs))
    embeds = emb.embed_docs([c.text for c in chunks])
    summaries, _ = RaptorTreeBuilder(emb, ExtractSummarizer(),
                                     cluster_size=5, max_levels=3).build(chunks, embeds)
    return PageIndexTree.from_nodes(chunks, summaries), chunks, summaries


def test_tree_builds_with_roots_and_leaves():
    tree, chunks, summaries = _tree()
    assert tree.size == len(chunks) + len(summaries)
    assert len(tree._roots) >= 1
    assert all(not tree._nodes[c.chunk_id].children for c in chunks), "leaves have no children"


def test_descent_finds_relevant_leaf():
    tree, _, _ = _tree()
    results = tree.search("qdrant vector database hybrid search", top_k=5)
    assert results, "descent returned nothing"
    assert any("qdrant" in r.text.lower() for r in results), \
        f"no qdrant leaf in top-5: {[r.text[:40] for r in results]}"


def test_descent_no_vectors_touched():
    tree, _, _ = _tree()
    results = tree.search("python programming", top_k=5)
    assert all(r.dense_score is None and r.sparse_score is None for r in results), \
        "pageindex results must carry NO vector scores — that's the point"


def test_empty_query_returns_empty():
    tree, _, _ = _tree()
    assert tree.search("", top_k=5) == []
    assert tree.search("the of a an", top_k=5) == []   # stopwords only


def test_from_qdrant_roundtrip():
    emb = HashEmbedder(dim=256)
    client = memory_client()
    from index_build import build_raptor_index
    report = build_raptor_index(synthetic_docs(30), emb, ExtractSummarizer(), client,
                                prefix="pitest", strategy_name="passage",
                                cluster_size=5, max_levels=2)
    tree = PageIndexTree.from_qdrant(client, report["collection"])
    assert tree.size == report["n_chunks"], \
        f"tree {tree.size} != indexed {report['n_chunks']}"
    results = tree.search("qdrant vector database", top_k=5)
    assert results


def test_harness_pageindex_mode_answers():
    h = build_local_harness()
    r = h.answer(Query(text="what is qdrant", retrieval_mode="pageindex"))
    assert r.decision == Decision.ANSWER, f"pageindex mode got {r.decision}"
    assert r.trace.retrieval_mode == "pageindex"
    assert r.citations, "pageindex answers must still cite"


def test_harness_pageindex_ood_still_abstains():
    h = build_local_harness()
    r = h.answer(Query(text="quantum chromodynamics quark gluon plasma",
                       retrieval_mode="pageindex"))
    assert r.decision in (Decision.ABSTAIN_OOD, Decision.ABSTAIN_UNGROUNDED), \
        f"OOD must abstain in pageindex mode too, got {r.decision}"


def test_harness_default_stays_hybrid():
    h = build_local_harness()
    r = h.answer(Query(text="what is qdrant"))
    assert r.trace.retrieval_mode == "hybrid"


def test_cache_isolated_per_mode():
    h = build_local_harness()
    r1 = h.answer(Query(text="qdrant hybrid search database"))
    assert r1.decision == Decision.ANSWER
    r2 = h.answer(Query(text="qdrant hybrid search database", retrieval_mode="pageindex"))
    assert not r2.trace.cache_hit, "hybrid-cached answer must not serve pageindex mode"


def test_llm_navigate_variant():
    tree, _, _ = _tree()
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "1"          # always pick the first branch

    results = tree.llm_navigate("qdrant vector database", fake_llm, top_k=5)
    assert isinstance(results, list)
    assert calls, "llm_navigate should consult the LLM at least once"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} pageindex tests passed")
