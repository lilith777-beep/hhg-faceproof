"""R1 reranker + R2 router + R3 noisy-ASR normalizer — unit + integrated harness tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck                       # noqa: E402
from embeddings import HashEmbedder          # noqa: E402
from generation import ExtractiveGenerator   # noqa: E402
from guardrails import KeywordSafety         # noqa: E402
from harness import RAGHarness               # noqa: E402
from normalize import QueryNormalizer        # noqa: E402
from reranker import LexicalReranker         # noqa: E402
from retrieval import Retriever, memory_client  # noqa: E402
from router import HeuristicRouter, detect_language  # noqa: E402
from schemas import Decision, Query, RetrievedChunk  # noqa: E402
from stt import MockSTT                       # noqa: E402

CORPUS = [
    ("d1", "Goa is a state in western India famous for its beaches and Portuguese heritage."),
    ("d2", "Qdrant is a vector database supporting dense and sparse hybrid search with HNSW."),
    ("d3", "Sarvam Saarika transcribes Indian language speech including Hindi and English."),
    ("d4", "Python is a high level programming language used for data science and web development."),
]
COLL = "feat__passage"


def build_feature_harness():
    emb = HashEmbedder(dim=512)
    client = memory_client()
    r = Retriever(client, emb, use_sparse=True)
    chunker = ck.PassageAwareChunker(min_tokens=1, max_tokens=200)
    chunks = [c for did, txt in CORPUS
              for c in chunker.chunk(ck.Document(doc_id=did, text=txt, language="en"))]
    r.index(COLL, chunks, emb.embed_docs([c.text for c in chunks]))
    return RAGHarness(
        embedder=emb, retriever=r, generator=ExtractiveGenerator(), safety=KeywordSafety(),
        collection=COLL, ood_threshold=0.15, grounding_min_support=0.25,
        stt=MockSTT("what is qdrant", "en"),
        normalizer=QueryNormalizer(), router=HeuristicRouter(), reranker=LexicalReranker(),
        rerank_min=0.2, rerank_candidates=10,
        stage_timeouts={"retrieve": 60, "rerank": 40, "generate": 150, "ground_check": 40})


# ---- R3 normalize ----------------------------------------------------------------------
def test_normalize_strips_filler_and_leadin():
    nq = QueryNormalizer().normalize("um, so like, tell me what is Qdrant?")
    assert "qdrant" in nq.text.lower()
    assert "um" not in nq.text.lower().split() and "tell me" not in nq.text.lower()


def test_normalize_folds_homophone():
    nq = QueryNormalizer().normalize("what is quadrant")     # ASR homophone of 'qdrant'
    assert "qdrant" in nq.text.lower()


def test_normalize_confidence_widens():
    lo = QueryNormalizer().normalize("what is qdrant", confidence=0.3, base_top_k=8)
    hi = QueryNormalizer().normalize("what is qdrant", confidence=0.95, base_top_k=8)
    assert lo.expand and lo.top_k > hi.top_k and lo.ood_relax > 0
    assert not hi.expand and hi.ood_relax == 0.0
    assert nq_phon(lo)                                        # phonetic keys produced


def nq_phon(nq):
    return bool(nq.phonetic)


# ---- R2 router -------------------------------------------------------------------------
def test_router_language_detection():
    assert detect_language("गोवा कहाँ है") == "hi"
    assert detect_language("qdrant vector database") == "en"
    assert detect_language("கோவா எங்கே") == "ta"


def test_router_intents():
    r = HeuristicRouter()
    assert r.route("hello").intent == "chitchat"        # whole-string greeting (won't eat real Qs)
    assert r.route("who are you").intent == "meta"
    assert r.route("what is a vector database").intent == "qa"
    # default keeps cross-lingual retrieval open (no hard language filter)
    assert r.route("गोवा कहाँ है").filters == {}


# ---- R1 reranker -----------------------------------------------------------------------
def test_reranker_reorders_by_relevance():
    chunks = [
        RetrievedChunk(chunk_id="a", text="python programming language data science", score=0.9),
        RetrievedChunk(chunk_id="b", text="qdrant vector database hybrid search hnsw", score=0.5),
        RetrievedChunk(chunk_id="c", text="goa beaches portuguese heritage", score=0.4),
    ]
    out = LexicalReranker().rerank("qdrant vector database", chunks, top_n=3)
    assert out[0].chunk_id == "b"                            # most relevant rises despite lower score
    assert all(c.rerank_score is not None for c in out)
    assert out[0].rerank_score >= out[1].rerank_score


# ---- integrated harness ----------------------------------------------------------------
def test_harness_noisy_query_is_repaired_and_answered():
    h = build_feature_harness()
    resp = h.answer(Query(text="um, tell me what is quadrant"))   # filler + homophone
    assert resp.decision == Decision.ANSWER
    assert "qdrant" in resp.answer.lower()
    assert resp.trace.normalized and "qdrant" in resp.trace.normalized.lower()
    stages = {s.stage for s in resp.trace.stages}
    assert {"normalize", "route", "retrieve", "rerank", "generate"} <= stages
    assert resp.trace.rerank_top is not None


def test_harness_greeting_and_meta_small_talk():
    h = build_feature_harness()
    g = h.answer(Query(text="hello"))
    assert g.decision == Decision.SMALL_TALK and not g.abstained and g.answer
    assert not any(s.stage == "retrieve" for s in g.trace.stages)   # no wasted retrieval
    m = h.answer(Query(text="who are you"))
    assert m.decision == Decision.SMALL_TALK and m.trace.route.intent == "meta"


def test_harness_offtopic_still_abstains_with_reranker():
    h = build_feature_harness()
    resp = h.answer(Query(text="explain quantum chromodynamics quarkgluon lagrangian"))
    assert resp.decision == Decision.ABSTAIN_OOD and resp.abstained


def test_harness_low_confidence_voice_relaxes_gate():
    h = build_feature_harness()
    # a low-confidence transcript still routes through normalize (expand) without crashing
    resp = h.answer(Query(text="what is qdrant", asr_confidence=0.3))
    assert resp.decision in (Decision.ANSWER, Decision.ABSTAIN_OOD)


class _CertainReranker:
    """Stub cross-encoder with a fixed verdict — lets the test decouple the rerank signal
    from lexical overlap (LexicalReranker and the gate's lexical arm measure the same thing)."""

    def __init__(self, score: float):
        self.score = score

    def rerank(self, query, chunks, top_n=8):
        for c in chunks:
            c.rerank_score = self.score
        return list(chunks)[:top_n]


def test_lexical_admission_requires_reranker_confirmation():
    """876k-scale regression (the westeros class): dense fails τ, common-token lexical
    overlap admits, but the cross-encoder is certain nothing retrieved answers (0.05) —
    the gate must abstain with signal lexical_unconfirmed, never serve stitched junk."""
    h = build_feature_harness()
    h.ood_threshold = 0.99          # force the dense arm to fail -> lexical override path
    h.rerank_veto = 0.0             # isolate the confirmation rule from the veto
    h.reranker = _CertainReranker(0.05)          # junk-certain, above any veto
    resp = h.answer(Query(text="what is qdrant"))
    assert resp.decision == Decision.ABSTAIN_OOD, resp.decision
    assert resp.trace.ood.signal == "lexical_unconfirmed", resp.trace.ood.signal


def test_lexical_rescue_survives_when_reranker_confirms():
    """The noisy-ASR rescue the lexical arm exists for: dense fails, lexical admits, and
    the cross-encoder CONFIRMS relevance -> the query still answers."""
    h = build_feature_harness()
    h.ood_threshold = 0.99
    h.rerank_veto = 0.0
    h.reranker = _CertainReranker(0.9)           # confident the evidence answers
    resp = h.answer(Query(text="what is qdrant"))
    assert resp.decision == Decision.ANSWER, resp.decision


def test_question_shape_junk_needs_strong_rerank_when_entity_uncovered():
    """The Narnia class: evidence matches the question SHAPE (partial coverage — the
    entity token 'zorbulon' appears nowhere) and the reranker is mid-confident (0.5,
    above rerank_min) — still must abstain; only rerank_strong admits partial coverage."""
    h = build_feature_harness()
    h.ood_threshold = 0.99
    h.rerank_veto = 0.0
    h.reranker = _CertainReranker(0.5)           # fooled by shape, not strongly confident
    resp = h.answer(Query(text="what is qdrant zorbulon"))
    assert resp.decision == Decision.ABSTAIN_OOD, resp.decision
    assert resp.trace.ood.signal == "lexical_unconfirmed", resp.trace.ood.signal


def test_full_coverage_rescue_admits_at_rerank_min():
    """Full content-token coverage (a true lexical rescue) still admits at the ordinary
    rerank_min bar — the strong bar applies only to partial coverage."""
    h = build_feature_harness()
    h.ood_threshold = 0.99
    h.rerank_veto = 0.0
    h.reranker = _CertainReranker(0.5)           # >= rerank_min (0.2), < rerank_strong
    resp = h.answer(Query(text="what is qdrant"))  # 'qdrant' fully covered by d2
    assert resp.decision == Decision.ANSWER, resp.decision


def test_zero_overlap_rescue_composes_verbatim_when_rerank_strong():
    """The typo/romanized signature ('wat is fotosynthesis' class): gates admit, the
    cross-encoder is CERTAIN (>= rerank_strong), but the lexical composer scores nothing.
    The verbatim-lead rescue must answer, cited, instead of abstain_ungrounded."""
    h = build_feature_harness()
    h.ood_threshold = 0.0                        # dense arm admits (typos embed near enough)
    h.rerank_veto = 0.0
    h.reranker = _CertainReranker(0.9)           # >= rerank_strong (0.8): certain
    resp = h.answer(Query(text="wat iz kdrant"))  # zero content-token overlap with corpus
    assert resp.decision == Decision.ANSWER, (resp.decision, resp.reason)
    assert resp.citations, "verbatim rescue must cite its source chunk"
    stages = [s.stage for s in resp.trace.stages]
    assert "generate_fallback" in stages, stages


def test_zero_overlap_still_abstains_when_rerank_not_strong():
    """Same zero-overlap shape but the cross-encoder is only mid-confident (< rerank_strong):
    the rescue must NOT fire — borderline admissions keep abstaining (grounded, or nothing)."""
    h = build_feature_harness()
    h.ood_threshold = 0.0
    h.rerank_veto = 0.0
    h.reranker = _CertainReranker(0.5)           # admitted, but not certain
    resp = h.answer(Query(text="wat iz kdrant"))
    assert resp.decision == Decision.ABSTAIN_UNGROUNDED, resp.decision


def test_verbatim_lead_skips_interrogative_and_cites():
    from generation import verbatim_lead

    class _C:
        def __init__(self, chunk_id, text, payload=None):
            self.chunk_id, self.text, self.payload = chunk_id, text, payload or {}

    out = verbatim_lead("wat iz fotosynthesis", [
        _C("c1", "What is photosynthesis? Photosynthesis is the process plants use to "
                 "convert sunlight into energy. It occurs in chloroplasts.")])
    assert out.text.startswith("Photosynthesis is the process")   # question sentence skipped
    assert out.text.rstrip().endswith("[1]")
    assert out.cited_chunk_ids == ["c1"]


def test_greeting_variants_route_to_small_talk():
    r = HeuristicRouter()
    for g in ["hello there", "hi there", "hey", "hello", "hi!", "namaste"]:
        assert r.route(g).intent in ("chitchat", "meta"), (g, r.route(g).intent)
    # a real question that merely starts with a greeting word is NOT small talk
    assert r.route("hello what is a corporation").intent == "qa"


def test_real_question_containing_help_is_not_small_talk():
    """'what foods help lower cholesterol' (5 words, bare \\bhelp\\b) hit the canned
    meta reply 3/3 live — 'help' is meta only as the whole utterance now."""
    r = HeuristicRouter().route("what foods help lower cholesterol")
    assert r.intent not in ("meta", "greeting"), r.intent
    assert HeuristicRouter().route("help").intent == "meta"       # bare help stays meta
    assert HeuristicRouter().route("what is this disease called").intent \
        not in ("meta", "greeting")


def test_verbatim_lead_language_pick_respects_min_score():
    """A same-language chunk at position 2 with junk rerank score must NOT ride the top
    chunk's certainty into a verbatim answer (P0 from the core review)."""
    from generation import verbatim_lead

    class _C:
        def __init__(self, chunk_id, text, lang, rerank):
            self.chunk_id, self.text = chunk_id, text
            self.payload = {"language": lang}
            self.rerank_score = rerank

    top = _C("en1", "Diabetes is a metabolic disease marked by high blood sugar.", "en", 0.97)
    junk_hi = _C("hi9", "यह लेख ग्रीक देवताओं के राजा के बारे में है।", "hi", 0.08)
    out = verbatim_lead("मधुमेह", [top, junk_hi], min_score=0.90)
    assert out.cited_chunk_ids == ["en1"]        # junk hi chunk skipped despite lang match
    out2 = verbatim_lead("मधुमेह", [top, junk_hi])   # no bar -> old preference (unit only)
    assert out2.cited_chunk_ids == ["hi9"]


def test_rerank_timeout_keeps_lexical_admissions_honest():
    """Reranker CONFIGURED but failing this request: a lexical admission whose
    confirmation bar cannot be applied must abstain, not answer at 0.34 overlap."""

    class _BrokenReranker:
        def rerank(self, query, chunks, top_n=8):
            raise RuntimeError("simulated rerank timeout")

    h = build_feature_harness()
    h.ood_threshold = 0.99          # dense fails -> lexical override path
    h.rerank_veto = 0.0
    h.reranker = _BrokenReranker()
    resp = h.answer(Query(text="what is qdrant"))
    assert resp.decision == Decision.ABSTAIN_OOD, resp.decision
    assert resp.trace.ood.signal == "lexical_unconfirmed", resp.trace.ood.signal


def test_indic_function_words_are_not_content_tokens():
    from guardrails import _content_tokens
    toks = _content_tokens("मधुमेह के लक्षण क्या हैं")
    assert "मधुमेह" in toks and "लक्षण" in toks
    assert not toks & {"के", "क्या", "हैं"}, toks   # case-markers/copulas are not evidence


def test_devanagari_mr_ne_disambiguation():
    assert detect_language("पाणी म्हणजे काय आहे") == "mr"
    assert detect_language("यो कसरी गर्ने हो") == "ne"
    assert detect_language("मधुमेह के लक्षण क्या हैं") == "hi"    # plain Hindi unchanged


def test_safety_catches_cbrn_and_indic_starters():
    s = KeywordSafety()
    assert not s.check("how to make ricin at home").safe
    assert not s.check("बम कैसे बनाएं").safe
    assert s.check("what is the history of the atom bomb").safe   # informational stays open
    assert s.check("ricin poisoning symptoms treatment").safe     # medical query stays open


def test_is_romanized_indic_gate_rejects_english_and_brands():
    """The gate is the ONLY defense against transliterating English into Devanagari garbage,
    so it MUST NOT flag grammatical English or brand/name queries (red-team measured these
    as false positives on the old gate)."""
    from router import is_romanized_indic
    # must NOT flag (would be corrupted) — English + brand/name collisions
    for q in ["the KI salt and the ache it causes", "ki and ache in chemistry",
              "karo syrup mera brand", "what is a corporation", "how does inflation work",
              "why does my body ache today", "amar chitra katha comics"]:
        assert is_romanized_indic(q)[0] is False, q
    # native / mixed script -> never transliterate
    assert is_romanized_indic("what is मधुमेह disease")[0] is False
    assert is_romanized_indic("ذیابیطس کیا ہے")[0] is False
    # genuine romanized Hindi -> flag (2 clean function words, or 1 function + 1 content gloss)
    assert is_romanized_indic("madhumeh kya hai kaise")[0] is True
    assert is_romanized_indic("bukhar ka ilaj batao")[0] is True   # batao + bukhar/ilaj gloss
    assert is_romanized_indic("diabetes kaise hota hai")[0] is True   # 3 clean function words


def test_transliterator_caches_and_rejects_non_native():
    from stt import SarvamTransliterator

    class _Resp:
        def __init__(self, p): self._p = p
        def raise_for_status(self): pass
        def json(self): return self._p

    class _Fake:
        def __init__(self, out): self.out = out; self.calls = 0
        def post(self, url, **kw): self.calls += 1; return _Resp({"transliterated_text": self.out})

    x = SarvamTransliterator(api_key="k")
    x._httpx = _Fake("मधुमेह क्या है")
    assert x.transliterate("madhumeh kya hai", "hi") == "मधुमेह क्या है"
    assert x.transliterate("madhumeh kya hai", "hi") == "मधुमेह क्या है"   # 2nd = cache
    assert x._httpx.calls == 1, "second call must hit the cache, not the API"
    # a non-native (still-Latin) result is rejected -> input returned unchanged
    x2 = SarvamTransliterator(api_key="k"); x2._httpx = _Fake("madhumeh kya hai")
    assert x2.transliterate("madhumeh kya hai", "hi") == "madhumeh kya hai"
    # an unsupported language -> unchanged, no call
    x3 = SarvamTransliterator(api_key="k"); x3._httpx = _Fake("x")
    assert x3.transliterate("hello", "en") == "hello" and x3._httpx.calls == 0


def test_harness_transliterates_typed_romanized_and_skips_dualarm():
    """A typed romanized-Hindi query is transliterated to native script (a 'transliterate'
    stage appears), which puts it on the native path — the romanized dual-arm (embed_roman/
    retrieve_roman) must NOT fire. A transliterator failure keeps the dual-arm (no regression)."""
    class _FakeXlit:
        def __init__(self, mapping): self.mapping = mapping
        def transliterate(self, text, lang): return self.mapping.get(text, text)

    h = build_feature_harness()
    h.transliterator = _FakeXlit({"qdrant kya hai": "क्यूड्रैंट क्या है"})
    resp = h.answer(Query(text="qdrant kya hai"))     # LID: romanized hi (kya, hai)
    stages = [s.stage for s in resp.trace.stages]
    assert "transliterate" in stages, stages
    assert "embed_roman" not in stages and "retrieve_roman" not in stages, stages

    # transliterator returns input unchanged (failure) -> dual-arm fallback still runs
    h2 = build_feature_harness()
    h2.transliterator = _FakeXlit({})                 # no mapping -> returns input, no-op
    resp2 = h2.answer(Query(text="qdrant kya hai"))
    st2 = [s.stage for s in resp2.trace.stages]
    assert "embed_roman" in st2, st2                  # fell back to the dual-arm


def test_romanized_indic_transcript_is_transliterated_to_native():
    """The user's bug: Sarvam sometimes returns Hindi speech in Latin script, and the
    Devanagari corpus can't match it. SarvamSTT must transliterate a Latin transcript back
    to native script when an Indic language is detected — and leave native/English alone."""
    from stt import SarvamSTT

    class _Resp:
        def __init__(self, payload): self._p = payload
        def raise_for_status(self): pass
        def json(self): return self._p

    class _FakeHttpx:
        def __init__(self, stt_payload):
            self.stt_payload = stt_payload
            self.xlit_calls = []
        def post(self, url, **kw):
            if url.endswith("/transliterate"):
                self.xlit_calls.append(kw.get("json", {}))
                return _Resp({"transliterated_text": "मधुमेह के लक्षण क्या है"})
            return _Resp(self.stt_payload)

    # (a) romanized Hindi (lang hi-IN, Latin text) -> transliterated to Devanagari
    stt = SarvamSTT(api_key="x")
    stt._httpx = _FakeHttpx({"transcript": "madhumeh ke lakshan kya hai",
                             "language_code": "hi-IN", "language_probability": 0.9})
    t = stt.transcribe(b"audio")
    assert any(0x900 <= ord(c) <= 0x97f for c in t.text), t.text   # now Devanagari
    assert len(stt._httpx.xlit_calls) == 1                          # transliterate was called

    # (b) already-Devanagari transcript -> NOT transliterated (no wasted call, no corruption)
    stt2 = SarvamSTT(api_key="x")
    stt2._httpx = _FakeHttpx({"transcript": "मधुमेह के लक्षण", "language_code": "hi-IN"})
    t2 = stt2.transcribe(b"audio")
    assert t2.text == "मधुमेह के लक्षण"
    assert stt2._httpx.xlit_calls == []

    # (c) English transcript -> never transliterated
    stt3 = SarvamSTT(api_key="x")
    stt3._httpx = _FakeHttpx({"transcript": "what is a corporation", "language_code": "en-IN"})
    t3 = stt3.transcribe(b"audio")
    assert t3.text == "what is a corporation"
    assert stt3._httpx.xlit_calls == []


def test_voice_without_stt_returns_typed_error_not_500():
    """A tier with no STT (the vectorless edge) must answer voice requests with a typed
    ERROR response — a raise here becomes a 500, which the frontend reads as backend-DOWN
    and dead-marks a backend that answers text queries fine."""
    h = build_feature_harness()
    h.stt = None
    resp = h.answer_voice(b"\x1aE\xdf\xa3fake-webm-bytes", language="hi")
    assert resp.decision == Decision.ERROR
    assert resp.abstained is True
    assert "stt" in (resp.reason or "")
    assert resp.answer                     # localized error copy, never empty/raw traceback


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} feature tests passed")
