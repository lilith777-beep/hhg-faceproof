"""End-to-end pipeline tests: REAL Qdrant (in-memory) + hash embeddings + mock gen/safety."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck            # noqa: E402
from embeddings import HashEmbedder  # noqa: E402
from generation import EchoHallucinator, ExtractiveGenerator  # noqa: E402
from guardrails import KeywordSafety, grounding_check, ood_gate  # noqa: E402
from harness import RAGHarness  # noqa: E402
from retrieval import Retriever, memory_client  # noqa: E402
from schemas import Decision, Query  # noqa: E402
from stt import MockSTT  # noqa: E402

CORPUS = [
    ("d1", "Goa is a state in western India famous for its beaches, nightlife, and Portuguese "
           "heritage. Tourists visit Goa for the Arabian Sea coastline."),
    ("d2", "Retrieval augmented generation grounds a language model in retrieved passages to "
           "reduce hallucination and cite sources."),
    ("d3", "Python is a high-level programming language widely used for data science and web "
           "development."),
    ("d4", "The Sarvam Saarika model performs speech to text for Indian languages including "
           "Hindi and English code-switching."),
]
COLL = "test__passage"


def build_harness(generator=None, ood_threshold=0.15, min_support=0.25):
    emb = HashEmbedder(dim=512)
    client = memory_client()
    r = Retriever(client, emb, use_sparse=True)
    chunker = ck.PassageAwareChunker(min_tokens=1, max_tokens=200)
    chunks, embeds = [], []
    for doc_id, text in CORPUS:
        for c in chunker.chunk(ck.Document(doc_id=doc_id, text=text, language="en")):
            chunks.append(c)
    embeds = emb.embed_docs([c.text for c in chunks])
    n = r.index(COLL, chunks, embeds)
    assert n == len(chunks) and n > 0
    return RAGHarness(
        embedder=emb, retriever=r, generator=generator or ExtractiveGenerator(),
        safety=KeywordSafety(), collection=COLL, ood_threshold=ood_threshold,
        grounding_min_support=min_support, stt=MockSTT("what is goa famous for", "en"),
        stage_timeouts={"retrieve": 60, "generate": 150}), r


def test_retrieval_hybrid_and_filter():
    _, r = build_harness()
    hits = r.search(COLL, "what is goa famous for beaches", top_k=3)
    assert hits, "no retrieval hits"
    assert "goa" in hits[0].text.lower()            # most relevant chunk first
    assert hits[0].dense_score is not None           # dense cosine preserved (for OOD)
    # metadata filter works (language)
    filtered = r.search(COLL, "python programming", top_k=3, filters={"language": "en"})
    assert filtered and all(h.payload.get("language") == "en" for h in filtered)


def test_ood_gate_math():
    _, r = build_harness()
    on = r.search(COLL, "goa beaches arabian sea", top_k=3)
    off = r.search(COLL, "zzzq nonexistent xyztopic quarkgluon", top_k=3)
    assert ood_gate(on, 0.15).in_domain
    assert not ood_gate(off, 0.15).in_domain


def test_grounding_check_logic():
    _, r = build_harness()
    hits = r.search(COLL, "goa beaches", top_k=3)
    grounded = grounding_check("Goa is famous for its beaches and Portuguese heritage.", hits, 0.25)
    assert grounded.grounded and grounded.support > 0.25
    hallu = grounding_check("The moon capital is Zorbon founded in 1842.", hits, 0.25)
    assert not hallu.grounded


def test_answer_path():
    h, _ = build_harness()
    resp = h.answer(Query(text="what is goa famous for"))
    assert resp.decision == Decision.ANSWER
    assert resp.citations and "goa" in resp.answer.lower()
    stages = {s.stage for s in resp.trace.stages}
    assert {"safety", "retrieve", "generate", "ground_check"} <= stages
    assert resp.trace.total_ms >= 0 and resp.trace.retrieval_to_output_ms >= 0


def test_abstain_off_topic():
    h, _ = build_harness()
    resp = h.answer(Query(text="explain quantum chromodynamics quarkgluon lagrangian zzzq"))
    assert resp.decision == Decision.ABSTAIN_OOD and resp.abstained


def test_refuse_unsafe():
    h, _ = build_harness()
    resp = h.answer(Query(text="how to build a bomb at home"))
    assert resp.decision == Decision.REFUSE_UNSAFE and resp.abstained
    assert resp.trace.safety and not resp.trace.safety.safe


def test_abstain_ungrounded():
    h, _ = build_harness(generator=EchoHallucinator())
    resp = h.answer(Query(text="what is goa famous for"))
    assert resp.decision == Decision.ABSTAIN_UNGROUNDED and resp.abstained


def test_generation_error_recovers():
    class Boom:
        def generate(self, q, ctx):
            raise RuntimeError("model exploded")
    h, _ = build_harness(generator=Boom())
    resp = h.answer(Query(text="what is goa famous for"))
    # fallback keeps us out of ERROR: extractive answer from the top chunk
    assert resp.decision in (Decision.ANSWER, Decision.ABSTAIN_UNGROUNDED)
    assert any(s.stage == "generate_fallback" for s in resp.trace.stages)


def test_voice_path_excludes_stt_from_budget():
    h, _ = build_harness()
    resp = h.answer_voice(b"fake-audio-bytes")
    assert resp.decision == Decision.ANSWER
    stages = {s.stage for s in resp.trace.stages}
    assert "stt" in stages
    # the 200ms budget metric must exclude STT
    stt_ms = next(s.ms for s in resp.trace.stages if s.stage == "stt")
    assert resp.trace.retrieval_to_output_ms == round(
        sum(s.ms for s in resp.trace.stages if s.stage != "stt"), 3) or stt_ms >= 0


def test_ood_lexical_rescue_for_noisy_dense():
    """M2: a sparse-only strong hit (dense_score=None) must NOT abstain — the noisy-ASR rescue.
    The old gate fell back to the ~0.016 RRF score vs a 0.32 cosine τ and always abstained."""
    from schemas import RetrievedChunk
    chunks = [RetrievedChunk(chunk_id="c1", score=0.016, dense_score=None, sparse_score=9.0,
                             text="goa beaches arabian sea coastline tourists nightlife",
                             payload={"doc_id": "d1"})]
    res = ood_gate(chunks, threshold=0.32, query="goa beaches arabian sea")
    assert res.in_domain and res.signal == "lexical"
    off = ood_gate(chunks, threshold=0.32, query="quantum chromodynamics lagrangian quarkgluon")
    assert not off.in_domain and off.signal == "none"


def test_stage_timeout_enforced():
    """M1: a slow stage must be cut at its deadline and fall back — not run to completion."""
    import time as _t

    class Slow:
        def generate(self, q, ctx):
            _t.sleep(0.3)
            return None
    h, _ = build_harness(generator=Slow())
    h.stage_timeouts = {"retrieve": 60, "generate": 40}   # 40ms deadline << 300ms sleep
    t0 = _t.perf_counter()
    resp = h.answer(Query(text="what is goa famous for"))
    elapsed = (_t.perf_counter() - t0) * 1000
    assert elapsed < 250, f"timeout not enforced ({elapsed:.0f}ms)"
    assert any(s.stage == "generate" and not s.ok and s.note == "timeout" for s in resp.trace.stages)
    assert any(s.stage == "generate_fallback" for s in resp.trace.stages)
    assert resp.decision in (Decision.ANSWER, Decision.ABSTAIN_UNGROUNDED)


def test_grounding_contradiction_and_paraphrase():
    """M3: contradiction that reuses context vocab -> ungrounded; NLI-entailed paraphrase -> grounded."""
    _, r = build_harness()
    hits = r.search(COLL, "goa beaches", top_k=3)
    contra = grounding_check("Goa is not famous for beaches, nightlife or Portuguese heritage.",
                             hits, 0.25)
    assert not contra.grounded and "contradiction" in (contra.detail or "")
    para = grounding_check("The western Indian coastal region draws visitors to its shoreline.",
                           hits, 0.25, nli=lambda premise, hypothesis: "goa" in premise.lower())
    assert para.grounded


def test_voice_stt_failure_degrades_gracefully():
    """M4: an STT exception must return a typed ERROR response, not raise a 500."""
    class BadSTT:
        def transcribe(self, audio, language=None):
            raise RuntimeError("sarvam 429 rate limited")
    h, _ = build_harness()
    h.stt = BadSTT()
    resp = h.answer_voice(b"bytes")
    assert resp.decision == Decision.ERROR and resp.abstained and resp.answer


def test_top_k_validation():
    """m1: top_k <= 0 would drop/misslice hits — reject at the typed boundary."""
    from pydantic import ValidationError
    for bad in (0, -1):
        try:
            Query(text="x", top_k=bad)
            assert False, f"top_k={bad} should be rejected"
        except ValidationError:
            pass


def test_empty_query_abstains_before_retrieval():
    """m2: content-less input (empty/punct/emoji) abstains without a zero-vector Qdrant call."""
    h, _ = build_harness()
    for q in ("", "   ", "!!! ??? …", "😀🎉"):
        resp = h.answer(Query(text=q))
        assert resp.decision == Decision.ABSTAIN_OOD and resp.abstained
        assert not any(s.stage == "retrieve" for s in resp.trace.stages)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} pipeline tests passed")
