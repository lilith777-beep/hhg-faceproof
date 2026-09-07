"""
test_ingest.py — data pipeline quality: batching, language detection, quality filtering,
topic classification.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck  # noqa: E402
from embeddings import HashEmbedder  # noqa: E402
from index_build import (  # noqa: E402
    _detect_lang, _passage_quality, synthetic_docs, build_all_strategies,
)
from retrieval import memory_client  # noqa: E402
from topics import (  # noqa: E402
    KeywordTopicClassifier, EmbeddingTopicClassifier, apply_topic_labels,
)


# ---- language detection ------------------------------------------------------------------
def test_detect_lang_devanagari():
    assert _detect_lang("यह एक हिंदी वाक्य है") == "hi"

def test_detect_lang_tamil():
    assert _detect_lang("இது ஒரு தமிழ் வாக்கியம்") == "ta"

def test_detect_lang_bengali():
    assert _detect_lang("এটি একটি বাংলা বাক্য") == "bn"

def test_detect_lang_english_fallback():
    assert _detect_lang("This is a normal English sentence") == "en"

def test_detect_lang_mixed_indic_wins():
    assert _detect_lang("Hello world यह mixed है") == "hi"


# ---- quality filtering -------------------------------------------------------------------
def test_quality_accepts_good_passage():
    assert _passage_quality("Qdrant is a vector database that supports dense and sparse hybrid search for efficient retrieval.")

def test_quality_rejects_short():
    assert not _passage_quality("too short")

def test_quality_rejects_repetitive():
    assert not _passage_quality("the the the the the the the the the the the")

def test_quality_rejects_boilerplate():
    assert not _passage_quality("Click here to subscribe to our newsletter and accept our cookie policy terms of service.")

def test_quality_keeps_passages_with_urls():
    # MS MARCO is a web corpus — a passage CITING a link is legitimate content
    assert _passage_quality("The official census data is published at https://census.gov and updated every decade by the bureau.")

def test_quality_rejects_url_only_fragment():
    assert not _passage_quality("Visit https://example.com now")


# ---- keyword topic classification --------------------------------------------------------
def test_keyword_topic_technology():
    clf = KeywordTopicClassifier()
    assert clf.classify_one("Python is a programming language for software development and machine learning") == "technology"

def test_keyword_topic_geography():
    clf = KeywordTopicClassifier()
    assert clf.classify_one("Goa is a state on the western coast of India with beautiful beaches") == "geography"

def test_keyword_topic_general_fallback():
    clf = KeywordTopicClassifier()
    assert clf.classify_one("something something nothing special here") == "general"


# ---- embedding topic classification ------------------------------------------------------
def test_embedding_topic_classifier():
    emb = HashEmbedder(dim=128)
    docs = synthetic_docs(30)
    chunker = ck.PassageAwareChunker(min_tokens=1, max_tokens=200)
    chunks = chunker.chunk_corpus(docs)
    embeds = emb.embed_docs([c.text for c in chunks])

    clf = EmbeddingTopicClassifier(n_topics=4)
    topics = clf.classify_batch(chunks, embeds)
    assert len(topics) == len(chunks)
    assert all(isinstance(t, str) for t in topics)
    assert len(set(topics)) >= 2, "should produce at least 2 distinct topics"


# ---- apply_topic_labels integration -----------------------------------------------------
def test_apply_topic_labels_mutates_chunks():
    emb = HashEmbedder(dim=128)
    docs = synthetic_docs(20)
    chunker = ck.PassageAwareChunker(min_tokens=1, max_tokens=200)
    chunks = chunker.chunk_corpus(docs)
    embeds = emb.embed_docs([c.text for c in chunks])
    apply_topic_labels(chunks, embeds)
    assert all("topic" in c.extra for c in chunks), "every chunk should have a topic"


# ---- topic labels survive indexing -------------------------------------------------------
def test_topic_in_indexed_payload():
    emb = HashEmbedder(dim=128)
    client = memory_client()
    docs = synthetic_docs(20)
    report = build_all_strategies(docs, emb, client, prefix="ttest", include_semantic=False)
    from retrieval import Retriever
    retr = Retriever(client, emb, use_sparse=True)
    coll = report["passage"]["collection"]
    results = retr.search(coll, "qdrant vector database", top_k=5)
    assert len(results) > 0
    has_topic = any(r.payload.get("extra", {}).get("topic") for r in results)
    assert has_topic, "indexed chunks should carry topic metadata in extra.topic"


# ---- batching (BGEM3Embedder is Modal-only, test HashEmbedder batch parity) --------------
def test_hash_embedder_batch_consistency():
    emb = HashEmbedder(dim=256)
    texts = [f"document about topic number {i} with content" for i in range(100)]
    batch_all = emb.embed_docs(texts)
    batch_a = emb.embed_docs(texts[:50])
    batch_b = emb.embed_docs(texts[50:])
    combined = batch_a + batch_b
    for i in range(100):
        assert batch_all[i].dense == combined[i].dense, f"mismatch at {i}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} ingest pipeline tests passed")
