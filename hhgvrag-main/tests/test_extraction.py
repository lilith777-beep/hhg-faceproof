"""Extraction quality — regression tests from REAL failures seen live on 2026-08-15.
The old scorer returned passage headings / query echoes as answers."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generation import ExtractiveGenerator  # noqa: E402
from schemas import RetrievedChunk          # noqa: E402


def _chunk(cid, text):
    return RetrievedChunk(chunk_id=cid, text=text, score=1.0)


GEN = ExtractiveGenerator()


def test_heading_echo_is_not_the_answer():
    """LIVE BUG: 'how long does it take to get a tax refund' answered with the passage's own
    heading 'How Long Does it Take to Get a Refund? [2]'."""
    ctx = [_chunk("c1",
                  "How Long Does it Take to Get a Refund? "
                  "If you e-file and use direct deposit, the IRS issues most refunds within "
                  "21 days of accepting your return. Paper returns can take six to eight weeks.")]
    out = GEN.generate("how long does it take to get a tax refund", ctx)
    assert not out.text.split(" [")[0].rstrip().endswith("?"), \
        f"echoed a heading back: {out.text}"
    assert "21 days" in out.text or "weeks" in out.text, f"missed the answer: {out.text}"


def test_definition_beats_trivia():
    """LIVE BUG: 'what is a corporation' answered with McDonald's trivia over the definition."""
    ctx = [_chunk("c1",
                  "McDonald's Corporation is one of the most recognizable corporations in "
                  "the world. A corporation is a company or group of people authorized to "
                  "act as a single entity and recognized as such in law.")]
    out = GEN.generate("what is a corporation", ctx)
    assert "authorized to act as a single entity" in out.text, \
        f"picked trivia over definition: {out.text}"


def test_wrong_topic_heading_loses():
    """LIVE BUG: 'what is photosynthesis' answered 'What is Biomass?' from a ranked-lower
    passage because 'what is' matched."""
    ctx = [_chunk("c1", "What is Biomass? Biomass is organic material from plants and animals."),
           _chunk("c2", "Photosynthesis is the process by which plants convert sunlight, "
                        "water and carbon dioxide into oxygen and energy as sugar.")]
    out = GEN.generate("what is photosynthesis", ctx)
    assert "photosynthesis" in out.text.lower() and "[2]" in out.text, \
        f"wrong passage won: {out.text}"


def test_short_best_gets_followup_sentence():
    ctx = [_chunk("c1", "Qdrant supports hybrid search. It fuses dense vector similarity with "
                        "sparse lexical signals using reciprocal rank fusion for better recall.")]
    out = GEN.generate("does qdrant support hybrid search", ctx)
    assert "reciprocal rank fusion" in out.text, f"follow-up not appended: {out.text}"


def test_no_content_overlap_returns_empty_for_abstain():
    """Empty output -> the harness grounding gate abstains in the ASKER'S language.
    (The old canned English IDK string leaked through grounding as a fake ANSWER.)"""
    ctx = [_chunk("c1", "Bananas are rich in potassium and grow in tropical climates.")]
    out = GEN.generate("quantum chromodynamics lagrangian", ctx)
    assert out.text == "" and out.cited_chunk_ids == []


def test_meta_framing_loses_to_direct_statement():
    """LIVE BUG (defense-in-depth): RAPTOR-summary-style meta prose won as the answer."""
    ctx = [_chunk("c1",
                  "The passage discusses various methods for receiving a tax refund quickly. "
                  "Direct deposit is the fastest method, delivering most refunds within 21 days.")]
    out = GEN.generate("fastest way to receive a tax refund", ctx)
    assert "Direct deposit is the fastest" in out.text, f"meta prose won: {out.text}"
    assert not out.text.lower().startswith("the passage")


def test_citation_index_matches_chunk():
    ctx = [_chunk("first", "Nothing relevant in this one at all, just filler text here."),
           _chunk("second", "The tax refund arrives within 21 days for e-filed returns.")]
    out = GEN.generate("when does the tax refund arrive", ctx)
    assert out.cited_chunk_ids == ["second"], f"wrong citation: {out.cited_chunk_ids}"
    assert "[2]" in out.text


def test_composes_multi_sentence_answer_with_context():
    """USER FEEDBACK 2026-08-16: one-line answers read thin. The composer must add
    supporting context from the evidence — each segment cited — not just the lead."""
    ctx = [_chunk("c1",
                  "A corporation is a company or group of people authorized to act as a "
                  "single entity and recognized as such in law. Corporations enjoy most of "
                  "the rights and responsibilities that individuals possess."),
           _chunk("c2",
                  "Early corporations were established by charter. Most jurisdictions now "
                  "allow the creation of new corporations through registration with the state.")]
    out = GEN.generate("what is a corporation", ctx)
    assert out.text.startswith("A corporation is a company"), f"lead wrong: {out.text}"
    n_sents = out.text.count(". ") + out.text.count(".  ") + 1
    assert len(out.text) > 150, f"answer still thin ({len(out.text)} chars): {out.text}"
    assert "[1]" in out.text, "lead must be cited"
    assert len(out.cited_chunk_ids) >= 1


def test_no_repetitive_padding():
    """MMR novelty: near-duplicate sentences must not be repeated as 'context'."""
    ctx = [_chunk("c1",
                  "Direct deposit is the fastest way to get a tax refund. "
                  "Direct deposit is the fastest method for receiving a tax refund. "
                  "Getting a refund by direct deposit is the fastest option available.")]
    out = GEN.generate("fastest way to get a tax refund", ctx)
    assert out.text.lower().count("fastest") <= 2, f"padded with duplicates: {out.text}"


def test_irrelevant_chunks_contribute_nothing():
    ctx = [_chunk("c1", "Photosynthesis converts sunlight, water and carbon dioxide into "
                        "oxygen and glucose in plant cells."),
           _chunk("c2", "Bananas are yellow fruits rich in potassium and easy to digest.")]
    out = GEN.generate("how does photosynthesis work", ctx)
    assert "banana" not in out.text.lower(), f"junk support leaked in: {out.text}"


def test_abbreviations_do_not_fracture_sentences():
    """LIVE ROUGHNESS 2026-08-16: answer segment ended '…established by charter (i.e. [1]'."""
    ctx = [_chunk("c1",
                  "Early incorporated entities were established by charter (i.e. by an ad hoc "
                  "act granted by a monarch or passed by a parliament). Most jurisdictions now "
                  "allow incorporation through registration.")]
    out = GEN.generate("how were early corporations established", ctx)
    assert "(i.e. [1]" not in out.text and not out.text.rstrip("[1] ").endswith("(i.e."), \
        f"fractured at abbreviation: {out.text}"
    assert "ad hoc act" in out.text, f"the i.e. clause should stay attached: {out.text}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} extraction tests passed")
