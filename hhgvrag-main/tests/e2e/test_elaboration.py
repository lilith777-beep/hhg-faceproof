"""
Streaming quality mode (two-phase): the grounded card always lands first and never
changes; the elaboration streams in after via /ask_stream SSE, falling back to the
non-streaming /ask_text quality path when the endpoint is absent. Covers: SSE parser
unit behavior, both transports, abort-on-new-ask, quality-off mid-stream, unverified
badge, unavailable note, XSS-hostile tokens, the 2000-char display cap, and the voice
two-phase flow. /ask_stream does not exist on the live backend yet — everything here
is stubbed; the SSE path lights up automatically once the endpoint deploys.
"""
import json
from conftest import CANNED_JSON
from test_recorder import GUM_IMMEDIATE, _fake_recorder, _press
from test_a11y import _axe_serious_or_critical

GROUNDED_ANSWER = "A corporation is a legal entity distinct from its owners. [1]"   # from CANNED
ELAB_BODY = json.dumps({"answer": "ELAB deep prose about corporations.",
                        "decision": "answer", "citations": [],
                        "trace": {"total_ms": 3500.4, "stages": []}})


def _sse(*events):
    out = ""
    for name, data in events:
        out += "event: " + name + "\ndata: " + json.dumps(data) + "\n\n"
    return out


def _route_ask_text(page, quality_body=ELAB_BODY, grounded_body=CANNED_JSON):
    """Route /ask_text, splitting grounded (quality_mode:false) vs quality (true) calls."""
    calls = []
    def handler(route):
        body = route.request.post_data or ""
        calls.append(body)
        b = quality_body if '"quality_mode":true' in body else grounded_body
        route.fulfill(status=200, content_type="application/json", body=b)
    page.route("**/ask_text", handler)
    return calls


def _stream_404(page):
    calls = []
    def handler(route):
        calls.append(route.request.post_data or "")
        route.fulfill(status=404, content_type="text/plain", body="not found")
    page.route("**/ask_stream", handler)
    return calls


def _stream_sse(page, body):
    calls = []
    def handler(route):
        calls.append(route.request.post_data or "")
        route.fulfill(status=200, content_type="text/event-stream", body=body)
    page.route("**/ask_stream", handler)
    return calls


def _stream_hang(page):
    def handler(route):
        pass                                        # never fulfil — request stays pending
    page.route("**/ask_stream", handler)


def _ask_quality(page, text="what is a corporation?"):
    page.check("#qm")
    page.fill("#q", text)
    page.click("#ask")


def _wait_elab_text(page, needle, timeout=8000):
    page.wait_for_function(
        "(n) => document.getElementById('elabtext').textContent.includes(n)",
        arg=needle, timeout=timeout)


# ---------- SSE parser unit ----------

def test_sse_parser_unit(page):
    got = page.evaluate(r"""() => {
        const out = [];
        const feed = createSSEParser((n, d) => out.push([n, d]));
        feed("event: elab_delta\nda");                       // event split across feeds
        feed("ta: {\"t\":\"x\"}\n\nevent: elab_done\r\ndata: {\"ms\":1}\r\n");
        feed("\r\n");                                        // CRLF blank line dispatches
        feed(": a comment\n\n");                             // comment-only → no event
        feed("data: a\ndata: b\n\n");                        // multi-line data, default name
        feed("event: dangling\ndata: {\"t\":\"never\"}");    // no terminator → not dispatched
        return out;
    }""")
    assert got == [["elab_delta", '{"t":"x"}'],
                   ["elab_done", '{"ms":1}'],
                   ["message", "a\nb"]]


# ---------- fallback transport ----------

def test_two_phase_fallback_grounded_then_elaboration(page):
    stream_calls = _stream_404(page)
    text_calls = _route_ask_text(page)
    _ask_quality(page)
    _wait_elab_text(page, "ELAB deep prose")
    # phase 1 was the instant grounded path; phase 2 carried quality_mode:true
    assert len(text_calls) == 2
    assert '"quality_mode":false' in text_calls[0]
    assert '"quality_mode":true' in text_calls[1]
    assert len(stream_calls) == 1                       # SSE was attempted first (capability probe)
    # grounded card is the answer of record and is untouched by the elaboration
    assert page.inner_text("#answer") == GROUNDED_ANSWER
    assert page.inner_text("#total") == "51.4 ms"       # grounded budget, not the 3500 ms elaboration
    chip = page.inner_text("#elabchip")
    assert "3500 ms" in chip
    assert "ok" in (page.get_attribute("#elabchip", "class") or "")
    assert page.errors == []


def test_fallback_abstain_shows_quiet_unavailable(page):
    _stream_404(page)
    _route_ask_text(page, quality_body=json.dumps(
        {"answer": None, "decision": "abstain_ungrounded", "trace": {"total_ms": 900, "stages": []}}))
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent === 'elaboration unavailable'",
        timeout=8000)
    assert "dim" in (page.get_attribute("#elabchip", "class") or "")
    assert page.inner_text("#elabtext") == ""
    # the grounded card is undisturbed
    assert page.inner_text("#answer") == GROUNDED_ANSWER
    assert "Answered" in page.text_content("#decision")
    assert page.errors == []


# ---------- SSE transport ----------

def test_sse_streaming_success_verified(page):
    stream_calls = _stream_sse(page, _sse(
        ("elab_delta", {"t": "Hello "}),
        ("elab_delta", {"t": "streamed world."}),
        ("elab_done", {"ms": 2412.5, "grounded": True})))
    text_calls = _route_ask_text(page)
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent.includes('verified against sources')",
        timeout=8000)
    assert page.inner_text("#elabtext") == "Hello streamed world."
    chip = page.inner_text("#elabchip")
    assert "2413 ms" in chip                                  # 2412.5 → toFixed(0) rounds up
    assert "ok" in (page.get_attribute("#elabchip", "class") or "")
    # SSE succeeded → the fallback quality call must NOT have fired
    assert len(stream_calls) == 1
    assert not any('"quality_mode":true' in c for c in text_calls)
    # grounded card untouched
    assert page.inner_text("#answer") == GROUNDED_ANSWER
    assert page.inner_text("#total") == "51.4 ms"
    assert page.errors == []


def test_sse_unverified_badge_terracotta_and_axe(page):
    _stream_sse(page, _sse(
        ("elab_delta", {"t": "Unproven prose."}),
        ("elab_done", {"ms": 3100.0, "grounded": False})))
    _route_ask_text(page)
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent.includes('could not be verified')",
        timeout=8000)
    chip = page.inner_text("#elabchip")
    assert "could not be verified — treat as unconfirmed" in chip
    assert "3100 ms" in chip
    assert "warn" in (page.get_attribute("#elabchip", "class") or "")
    # a11y: the visible elaboration (warn chip + prose) introduces no serious/critical issues
    assert _axe_serious_or_critical(page) == []
    assert page.errors == []


def test_sse_malformed_events_ignored_then_done(page):
    _stream_sse(page,
        "event: elab_delta\ndata: not-json-at-all\n\n" +      # malformed → ignored
        _sse(("elab_delta", {"t": "Good token."}),
             ("mystery_event", {"t": "wrong name"}),          # unknown event → ignored
             ("elab_done", {"ms": "NaN", "grounded": True}))) # hostile ms → num() coercion
    _route_ask_text(page)
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent.includes('verified')", timeout=8000)
    assert page.inner_text("#elabtext") == "Good token."
    assert "0 ms" in page.inner_text("#elabchip")             # NaN coerced, never rendered as NaN
    assert "NaN" not in page.inner_text("#elabchip")
    assert page.errors == []


def test_sse_skipped_done_shows_unavailable_not_verified(page):
    # the real backend emits elab_done {"skipped": true} when quality can't run —
    # that must NEVER render as a "verified" claim about nonexistent prose
    _stream_sse(page, _sse(("elab_done", {"ms": 0, "grounded": True, "skipped": True})))
    _route_ask_text(page)
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent === 'elaboration unavailable'",
        timeout=8000)
    assert "dim" in (page.get_attribute("#elabchip", "class") or "")
    assert "verified" not in page.inner_text("#elabchip")
    assert page.inner_text("#elabtext") == ""
    assert page.errors == []


def test_sse_real_backend_event_sequence(page):
    # contract fidelity: replay the deployed /ask_stream shape — a leading `result` event
    # (full RAGResponse, ignored: the grounded card of record comes from phase 1),
    # then deltas, then done — including ensure_ascii=False style unicode
    result_event = ("result", json.loads(CANNED_JSON))
    _stream_sse(page, _sse(
        result_event,
        ("elab_delta", {"t": "कॉर्पोरेशन "}),
        ("elab_delta", {"t": "is a legal person."}),
        ("elab_done", {"ms": 2900.3, "grounded": True})))
    _route_ask_text(page)
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent.includes('verified')", timeout=8000)
    assert page.inner_text("#elabtext") == "कॉर्पोरेशन is a legal person."
    # the ignored `result` event did not disturb the grounded card
    assert page.inner_text("#answer") == GROUNDED_ANSWER
    assert page.inner_text("#total") == "51.4 ms"
    assert page.errors == []


# ---------- races + aborts ----------

def test_new_ask_aborts_prior_elaboration(page):
    state = {"n": 0}
    def stream_handler(route):
        state["n"] += 1
        if state["n"] == 1:
            pass                                               # A: hang mid-"streaming…"
        else:
            route.fulfill(status=404, content_type="text/plain", body="no")   # B: 404 → fallback
    page.route("**/ask_stream", stream_handler)
    _route_ask_text(page, quality_body=json.dumps(
        {"answer": "B ELABORATION", "decision": "answer", "trace": {"total_ms": 3200, "stages": []}}))
    _ask_quality(page, "query A")
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent.includes('streaming')", timeout=8000)
    page.fill("#q", "query B")
    page.click("#ask")
    _wait_elab_text(page, "B ELABORATION")                     # only B's elaboration renders
    assert "streaming" not in page.inner_text("#elabchip")     # A's hung stream never resurfaces
    assert page.errors == []


def test_quality_off_mid_stream_aborts_silently(page):
    _stream_hang(page)
    _route_ask_text(page)
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent.includes('streaming')", timeout=8000)
    page.uncheck("#qm")
    page.wait_for_function("() => document.getElementById('elab').hidden === true", timeout=4000)
    # grounded card untouched, no error surfaced anywhere
    assert page.inner_text("#answer") == GROUNDED_ANSWER
    assert "Answered" in page.text_content("#decision")
    assert "bad" not in (page.get_attribute("#dot", "class") or "")
    assert page.errors == []


# ---------- robustness ----------

def test_xss_stream_tokens_are_inert(page):
    _stream_sse(page, _sse(
        ("elab_delta", {"t": '<img src=x onerror="window.__xss=1"> '}),
        ("elab_delta", {"t": "</div><script>window.__xss=1</script>"}),
        ("elab_done", {"ms": 100, "grounded": True})))
    _route_ask_text(page)
    _ask_quality(page)
    _wait_elab_text(page, "<img")
    page.wait_for_timeout(150)
    assert page.evaluate("() => window.__xss") is None         # nothing executed
    assert page.query_selector("#elab img") is None            # markup never became elements
    assert page.query_selector("#elab script") is None
    assert page.query_selector("[onerror]") is None
    txt = page.inner_text("#elabtext")
    assert "<img" in txt and "<script>" in txt                 # rendered as inert TEXT
    assert page.errors == []


def test_elaboration_display_capped_with_fade(page):
    _stream_sse(page, _sse(
        ("elab_delta", {"t": "A" * 2500}),
        ("elab_done", {"ms": 1000, "grounded": True})))
    _route_ask_text(page)
    _ask_quality(page)
    page.wait_for_function(
        "() => document.getElementById('elabchip').textContent.includes('verified')", timeout=8000)
    assert page.evaluate("() => document.getElementById('elabtext').textContent.length") == 2000
    assert "clipped" in (page.get_attribute("#elabtext", "class") or "")
    assert page.errors == []


# ---------- voice two-phase ----------

def test_voice_two_phase_elaborates_transcript(harness):
    page = harness.new(init_scripts=[GUM_IMMEDIATE, _fake_recorder("audio/webm;codecs=opus")])
    voice_grounded = json.dumps({
        "answer": "Voice grounded answer.", "decision": "answer", "citations": [],
        "trace": {"query": "voice question here", "total_ms": 80.1,
                  "stages": [{"stage": "stt", "ms": 260, "ok": True}], "route": {"language": "en"}}})
    ask_posts = []
    page.route("**/ask", lambda r: (ask_posts.append(r.request.post_data_buffer),
                                    r.fulfill(status=200, content_type="application/json",
                                              body=voice_grounded)))
    _stream_404(page)
    text_calls = _route_ask_text(page, quality_body=json.dumps(
        {"answer": "VOICE ELAB", "decision": "answer", "trace": {"total_ms": 3400, "stages": []}}))
    page.check("#qm")
    _press(page, 420)
    _wait_elab_text(page, "VOICE ELAB")
    # phase 1 (multipart) never carried quality_mode
    assert b'name="quality_mode"' not in (ask_posts[0] or b"")
    # phase 2 elaborated the TRANSCRIPT via the text path
    quality = [c for c in text_calls if '"quality_mode":true' in c]
    assert len(quality) == 1
    assert '"text":"voice question here"' in quality[0]
    # grounded voice card intact
    assert page.inner_text("#answer") == "Voice grounded answer."
    assert page.errors == []
