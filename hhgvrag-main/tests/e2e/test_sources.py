"""
Sources / evidence view: auto-open on the first cited answer (then respect the user's
choice), per-segment verbatim highlights inside citation cards with multi-citation [n]
attribution, de-emphasized chunk ids, no sources section without citations, and
XSS-hostile citation text staying inert inside <mark> construction.
"""
import json

CITE1 = "Intro sentence. Segment one alpha extract. Trailing words here."
CITE2 = "Other doc opening. Segment two beta extract. And a coda."


def _payload(answer, cites, decision="answer"):
    return {"answer": answer, "decision": decision, "citations": cites,
            "trace": {"query": "q", "total_ms": 40.0,
                      "stages": [{"stage": "retrieve", "ms": 6, "ok": True}],
                      "route": {"language": "en"}}}


def _two_cite_payload():
    return _payload(
        "Segment one alpha extract. [1] Segment two beta extract. [2]",
        [{"chunk_id": "d1::p::0", "doc_id": "d1", "passage_id": 0, "text": CITE1},
         {"chunk_id": "d2::p::4", "doc_id": "d2", "passage_id": 4, "text": CITE2}])


def _route(page, holder):
    page.route("**/ask_text", lambda r: r.fulfill(
        status=200, content_type="application/json", body=holder["body"]))


def _ask(page, text="q"):
    page.fill("#q", text)
    page.click("#ask")


def test_multi_citation_cards_and_per_segment_highlights(page):
    _route(page, {"body": json.dumps(_two_cite_payload())})
    _ask(page)
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)

    cards = page.query_selector_all("#cites .cite")
    assert len(cards) == 2                                        # every citation renders a card
    assert [c.query_selector(".idx").text_content() for c in cards] == ["[1]", "[2]"]

    # per-segment attribution: each card highlights exactly its own verbatim extract
    marks0 = [m.text_content() for m in cards[0].query_selector_all("mark")]
    marks1 = [m.text_content() for m in cards[1].query_selector_all("mark")]
    assert marks0 == ["Segment one alpha extract."]
    assert marks1 == ["Segment two beta extract."]
    # surrounding source text is preserved around the mark (split text nodes, nothing dropped)
    assert "Intro sentence." in cards[0].inner_text()
    assert "Trailing words here." in cards[0].inner_text()
    # the visible answer keeps its [n] markers (the mapping cue to the cards)
    assert "[1]" in page.inner_text("#answer")
    # chunk_id lives in the de-emphasized .cid span
    assert cards[0].query_selector(".meta .cid").text_content() == "d1::p::0"
    assert "doc d1" in cards[0].query_selector(".meta").text_content()
    assert page.errors == []


def test_auto_open_first_answer_then_respects_user_choice(page):
    holder = {"body": json.dumps(_two_cite_payload())}
    _route(page, holder)
    _ask(page, "first")
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    assert page.eval_on_selector("#citebox", "e => e.open") is True    # discoverable at a glance

    page.eval_on_selector("#citebox", "e => e.open = false")           # user collapses it
    holder["body"] = json.dumps(_payload(
        "SECOND run segment. [1]",
        [{"chunk_id": "d9::p::9", "doc_id": "d9", "passage_id": 9,
          "text": "Lead-in. SECOND run segment. Tail."}]))
    _ask(page, "second")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('SECOND')", timeout=5000)
    assert page.eval_on_selector("#citebox", "e => e.open") is False   # unobtrusive afterwards
    # and the new card still carries its highlight
    assert page.query_selector("#cites mark").text_content() == "SECOND run segment."


def test_no_citations_hides_sources(page):
    _route(page, {"body": json.dumps(_payload("", [], decision="abstain_ood"))})
    _ask(page)
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    assert page.eval_on_selector("#citebox", "e => getComputedStyle(e).display") == "none"
    assert page.eval_on_selector("#citebox", "e => e.open") is False   # nothing auto-opened
    assert page.query_selector_all("#cites .cite") == []


def test_unmatched_and_out_of_range_markers_are_graceful(page):
    _route(page, {"body": json.dumps(_payload(
        "Missing extract not in the source. [1] Ghost segment. [7]",
        [{"chunk_id": "c1", "doc_id": "d1", "passage_id": 1,
          "text": "Completely different passage text with no overlap."}]))})
    _ask(page)
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    cards = page.query_selector_all("#cites .cite")
    assert len(cards) == 1
    assert cards[0].query_selector_all("mark") == []     # indexOf miss + [7] out of range → no marks
    assert "Completely different passage text" in cards[0].inner_text()
    assert page.errors == []


def test_hostile_citation_text_stays_inert_inside_marks(page):
    hostile_seg = '<img src=x onerror="window.__xss=1"> corporations act'
    _route(page, {"body": json.dumps(_payload(
        hostile_seg + " [1]",
        [{"chunk_id": "cx", "doc_id": "dx", "passage_id": 2,
          "text": "Preamble. " + hostile_seg + " suffix."}]))})
    _ask(page)
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    page.wait_for_timeout(150)
    assert page.evaluate("() => window.__xss") is None            # nothing executed
    assert page.query_selector("#cites img") is None              # markup never became elements
    assert page.query_selector("[onerror]") is None
    mark = page.query_selector("#cites mark")
    assert mark is not None
    assert "<img" in mark.text_content()                          # highlighted as inert TEXT
    assert page.errors == []
