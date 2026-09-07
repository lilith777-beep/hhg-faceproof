"""R4 RAPTOR + R5 CAG (semantic cache + session context) — unit + integrated tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck                          # noqa: E402
from cache import SemanticResponseCache, SessionContext  # noqa: E402
from embeddings import HashEmbedder             # noqa: E402
from generation import ExtractiveGenerator      # noqa: E402
from guardrails import KeywordSafety            # noqa: E402
from harness import RAGHarness                  # noqa: E402
from index_build import build_raptor_index, synthetic_docs  # noqa: E402
from normalize import QueryNormalizer           # noqa: E402
from raptor import ExtractSummarizer, RaptorTreeBuilder  # noqa: E402
from reranker import LexicalReranker            # noqa: E402
from retrieval import Retriever, memory_client  # noqa: E402
from router import HeuristicRouter              # noqa: E402
from schemas import Decision, Query, RAGResponse  # noqa: E402
from stt import MockSTT                         # noqa: E402


# ---- helpers ---------------------------------------------------------------------------
def _corpus_chunks(emb, n=30):
    docs = synthetic_docs(n)
    chunker = ck.PassageAwareChunker(min_tokens=1, max_tokens=200)
    chunks = chunker.chunk_corpus(docs)
    embeds = emb.embed_docs([c.text for c in chunks])
    return chunks, embeds


def build_cag_harness(with_cache=True, with_session=True):
    """Harness with RAPTOR collection + cache + session for integration tests."""
    emb = HashEmbedder(dim=512)
    client = memory_client()
    retr = Retriever(client, emb, use_sparse=True)

    # build RAPTOR index (leaves + summaries in one collection)
    docs = synthetic_docs(30)
    report = build_raptor_index(
        docs, emb, ExtractSummarizer(), client, prefix="test",
        strategy_name="passage", cluster_size=5, max_levels=3)

    cache = SemanticResponseCache(threshold=0.85, max_size=64) if with_cache else None
    session = SessionContext(max_turns=5) if with_session else None

    return RAGHarness(
        embedder=emb, retriever=retr, generator=ExtractiveGenerator(),
        safety=KeywordSafety(), collection=report["collection"],
        ood_threshold=0.15, grounding_min_support=0.25,
        stt=MockSTT("what is qdrant", "en"),
        normalizer=QueryNormalizer(), router=HeuristicRouter(),
        reranker=LexicalReranker(), rerank_min=0.2, rerank_candidates=20,
        response_cache=cache, session_ctx=session,
        stage_timeouts={"retrieve": 60, "rerank": 40, "generate": 150, "ground_check": 40})


# ---- R4 RAPTOR -------------------------------------------------------------------------
def test_raptor_tree_builds_multiple_levels():
    emb = HashEmbedder(dim=256)
    chunks, embeds = _corpus_chunks(emb, n=30)
    builder = RaptorTreeBuilder(emb, ExtractSummarizer(), cluster_size=5, max_levels=3)
    summaries, summary_embeds = builder.build(chunks, embeds)
    assert len(summaries) >= 2, f"expected >=2 summaries, got {len(summaries)}"
    assert len(summary_embeds) == len(summaries)
    levels = {c.extra.get("raptor_level", 0) for c in summaries}
    assert max(levels) >= 1, f"expected at least level 1, got {levels}"
    assert all(c.strategy == "raptor" for c in summaries)


def test_raptor_summaries_have_children():
    emb = HashEmbedder(dim=256)
    chunks, embeds = _corpus_chunks(emb, n=20)
    builder = RaptorTreeBuilder(emb, ExtractSummarizer(), cluster_size=5, max_levels=2)
    summaries, _ = builder.build(chunks, embeds)
    for s in summaries:
        assert "children" in s.extra, f"summary missing children: {s.chunk_id}"
        assert len(s.extra["children"]) >= 2, f"summary should have >=2 children"


def test_raptor_index_searchable():
    emb = HashEmbedder(dim=256)
    client = memory_client()
    docs = synthetic_docs(30)
    report = build_raptor_index(docs, emb, ExtractSummarizer(), client, prefix="rtest",
                                 strategy_name="passage", cluster_size=5, max_levels=3)
    assert report["n_summaries"] >= 1
    assert report["n_chunks"] == report["n_leaves"] + report["n_summaries"]

    retr = Retriever(client, emb, use_sparse=True)
    results = retr.search(report["collection"], "qdrant vector database", top_k=8)
    assert len(results) > 0
    has_raptor = any("raptor" in r.payload.get("strategy", "") for r in results)
    has_leaf = any("raptor" not in r.payload.get("strategy", "") for r in results)
    assert has_raptor or has_leaf, "RAPTOR collection should return results"


def test_raptor_uses_batched_summarizer():
    """Feasibility: the tree builder must call summarize_batch (one call per level), not
    one summarize per cluster — at 50k docs that's the difference between minutes and hours."""
    calls = {"batch": 0, "single": 0}

    class BatchSpy:
        def summarize(self, texts):
            calls["single"] += 1
            return "single summary of " + texts[0][:20]

        def summarize_batch(self, groups):
            calls["batch"] += 1
            return ["batch summary: " + g[0][:30] for g in groups]

    emb = HashEmbedder(dim=256)
    chunks, embeds = _corpus_chunks(emb, n=30)
    builder = RaptorTreeBuilder(emb, BatchSpy(), cluster_size=5, max_levels=3)
    summaries, s_embeds = builder.build(chunks, embeds)
    assert calls["batch"] >= 1, "summarize_batch was never used"
    assert calls["single"] == 0, "per-cluster summarize defeats batching"
    assert len(summaries) >= 2 and len(s_embeds) == len(summaries)


def test_llm_summarizer_batch_path():
    from raptor import LLMSummarizer
    seen = []

    def fake_batch(prompts):
        seen.append(len(prompts))
        return [f"summary {i}" for i in range(len(prompts))]

    s = LLMSummarizer(generate_fn=lambda p: "single", batch_fn=fake_batch)
    out = s.summarize_batch([["passage a", "passage b"], ["passage c"]])
    assert out == ["summary 0", "summary 1"]
    assert seen == [2], "batch_fn should receive all prompts in one call"


# ---- R5a semantic response cache -------------------------------------------------------
def test_semantic_cache_hit_and_miss():
    cache = SemanticResponseCache(threshold=0.80, max_size=10)
    emb = HashEmbedder(dim=128)
    vec = emb.embed_query("what is qdrant").dense
    resp = RAGResponse(answer="A vector DB", decision=Decision.ANSWER)
    cache.put(vec, resp)

    assert cache.get(vec) is not None, "exact vector should hit"
    assert cache.get(vec).answer == "A vector DB"

    vec_diff = emb.embed_query("quantum chromodynamics quarkgluon lagrangian").dense
    assert cache.get(vec_diff) is None, "unrelated query should miss"


def test_semantic_cache_evicts_oldest():
    cache = SemanticResponseCache(threshold=0.9, max_size=3)
    emb = HashEmbedder(dim=64)
    for i in range(5):
        v = emb.embed_query(f"query number {i} unique topic {i * 7}").dense
        cache.put(v, RAGResponse(answer=f"a{i}", decision=Decision.ANSWER))
    assert cache.size == 3, f"expected max 3 entries, got {cache.size}"


# ---- R5c session context ---------------------------------------------------------------
def test_session_context_expansion():
    ctx = SessionContext(max_turns=3)
    ctx.add_turn("s1", "what is qdrant", "Qdrant is a vector database")
    rewritten = ctx.rewrite("s1", "what are its features")
    assert "qdrant" in rewritten.lower(), "expansion should include prior topic"
    assert "features" in rewritten.lower()


def test_session_no_history_unchanged():
    ctx = SessionContext()
    assert ctx.rewrite("new_session", "hello world") == "hello world"
    assert not ctx.has_history("new_session")


# ---- integrated harness ----------------------------------------------------------------
def test_harness_cache_hit_skips_retrieval():
    h = build_cag_harness(with_cache=True, with_session=False)
    r1 = h.answer(Query(text="what is qdrant"))
    assert r1.decision == Decision.ANSWER
    assert not r1.trace.cache_hit

    r2 = h.answer(Query(text="what is qdrant"))
    assert r2.decision == Decision.ANSWER
    assert r2.trace.cache_hit
    assert not any(s.stage == "retrieve" for s in r2.trace.stages)


def test_harness_cache_only_stores_answers():
    h = build_cag_harness(with_cache=True, with_session=False)
    # off-topic → abstain (should NOT be cached)
    r1 = h.answer(Query(text="explain quantum chromodynamics quarkgluon lagrangian"))
    assert r1.decision == Decision.ABSTAIN_OOD
    assert h.response_cache.size == 0, "abstain responses should not be cached"

    # grounded answer → cached
    r2 = h.answer(Query(text="what is qdrant"))
    assert r2.decision == Decision.ANSWER
    assert h.response_cache.size == 1


def test_harness_session_records_turns():
    h = build_cag_harness(with_cache=False, with_session=True)
    h.answer(Query(text="what is qdrant", session_id="sess1"))
    assert h.session_ctx.has_history("sess1")


# ---- H8 quality mode ------------------------------------------------------------------
def test_harness_quality_mode():
    from generation import GenOutput
    h = build_cag_harness(with_cache=False)

    class MarkerGen:
        def generate(self, q, ctxs):
            return GenOutput(text=f"QUALITY: {ctxs[0].text} [1]",
                             cited_chunk_ids=[ctxs[0].chunk_id])
    h.quality_generator = MarkerGen()

    r1 = h.answer(Query(text="what is qdrant"))
    assert r1.decision == Decision.ANSWER
    assert "QUALITY" not in r1.answer
    assert not r1.trace.quality_mode

    r2 = h.answer(Query(text="what is qdrant", quality_mode=True))
    assert r2.decision == Decision.ANSWER
    assert "QUALITY" in r2.answer
    assert r2.trace.quality_mode


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} RAPTOR + CAG tests passed")
