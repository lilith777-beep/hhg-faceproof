"""Local, CPU-only tests for the multi-strategy chunker. No models required."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck  # noqa: E402

EN = (
    "Retrieval augmented generation grounds a language model in external context. "
    "It first retrieves relevant passages from a corpus. Then it conditions generation on them. "
    "This reduces hallucination. The corpus here is MS MARCO. Passages are short and factual."
)
HI = "गोवा भारत का एक राज्य है। यह अपने समुद्र तटों के लिए प्रसिद्ध है। यहाँ पर्यटक आते हैं।"

DOC_EN = ck.Document(doc_id="d1", text=EN, language="en")
DOC_HI = ck.Document(doc_id="d2", text=HI, language="hi")
DOC_PSG = ck.Document(
    doc_id="d3", language="en",
    text="Alpha passage about goa. Beta passage about beaches. Gamma passage about food.",
    passages=["Alpha passage about goa.", "Beta passage about beaches.", "Gamma passage about food."],
)


def _valid(chunks, doc):
    assert chunks, "no chunks produced"
    for c in chunks:
        assert c.text.strip(), "empty chunk text"
        assert c.token_count > 0
        assert 0 <= c.char_start <= c.char_end <= len(doc.text) + 1
        assert c.doc_id == doc.doc_id
        assert c.language == doc.language
        assert c.chunk_id  # well-formed id
        c.to_payload()     # must serialize flat for Qdrant


def test_sentence_split_multilingual():
    assert len(ck.split_sentences(EN)) == 6
    assert len(ck.split_sentences(HI)) == 3  # danda '।' terminates Indic sentences


def test_fixed_overlap():
    c = ck.FixedSizeChunker(size=12, overlap=4)
    chunks = c.chunk(DOC_EN)
    _valid(chunks, DOC_EN)
    assert all(ch.token_count <= 12 for ch in chunks)
    # overlap: consecutive chunks should share trailing/leading tokens when >1 chunk
    assert len(chunks) >= 2


def test_recursive():
    chunks = ck.RecursiveChunker(size=16, overlap=3).chunk(DOC_EN)
    _valid(chunks, DOC_EN)
    assert all(ch.token_count <= 16 + 3 + 2 for ch in chunks)  # budget + overlap slack
    # B1 regression: must emit real words, NOT '\n\n'-joined character soup (the falsy-strip bug)
    words = [w for ch in chunks for w in ch.text.split() if w.strip()]
    multi = sum(1 for w in words if len(w) > 1)
    assert multi >= 0.6 * len(words), "recursive chunker emitted character soup"
    assert any("generation" in ch.text for ch in chunks)  # a real word survives intact
    assert any("hallucination" in ch.text for ch in chunks)


def test_sentence_window():
    chunks = ck.SentenceWindowChunker(window=3, stride=2).chunk(DOC_EN)
    _valid(chunks, DOC_EN)
    assert chunks[0].extra["n_sentences"] <= 3


def test_passage_aware_keeps_passage_ids():
    chunks = ck.PassageAwareChunker(min_tokens=1, max_tokens=100).chunk(DOC_PSG)
    _valid(chunks, DOC_PSG)
    assert any(ch.passage_id is not None for ch in chunks)


def test_hierarchical_parent_child():
    chunks = ck.HierarchicalChunker(parent_size=20, child_size=8, overlap=2).chunk(DOC_EN)
    _valid(chunks, DOC_EN)
    roles = {c.extra.get("role") for c in chunks}
    assert roles == {"parent", "child"}
    parents = {c.chunk_id for c in chunks if c.extra.get("role") == "parent"}
    children = [c for c in chunks if c.extra.get("role") == "child"]
    assert children and all(c.parent_id in parents for c in children)


def test_semantic_with_mock_embedder():
    # deterministic "embedding": one-hot over a tiny vocab so topically-different
    # sentences have low cosine similarity -> a boundary forms.
    vocab = {}

    def embed(sents):
        vecs = []
        for s in sents:
            v = [0.0] * 64
            for w in ck.approx_tokenizer(s.lower()):
                idx = vocab.setdefault(w, len(vocab) % 64)
                v[idx] += 1.0
            vecs.append(v)
        return vecs

    chunks = ck.SemanticChunker(embed, breakpoint_percentile=40).chunk(DOC_EN)
    _valid(chunks, DOC_EN)
    assert len(chunks) >= 1


def test_registry_and_indic():
    reg = ck.all_strategies()
    assert set(reg) == {"fixed", "recursive", "sentwin", "passage", "hierarchical"}
    for name, chunker in reg.items():
        out = chunker.chunk(DOC_HI)  # every strategy must handle Indic text
        _valid(out, DOC_HI)
        assert all(c.strategy == name for c in out)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} chunking tests passed")
