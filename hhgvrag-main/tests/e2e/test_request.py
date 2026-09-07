"""
Request executor: monotonic sequencing (newest wins, stale suppressed), abort of the
prior in-flight request, a 12 s deadline with NO retry, one retry on genuine network
failure only, r.ok gating, malformed-JSON handling, the IME/composition Enter guard, and
the rule that a query error must not flip the global health dot to offline.
"""
import json
import pytest


def _answer(text, decision="answer"):
    return json.dumps({"answer": text, "decision": decision,
                       "trace": {"query": "q", "total_ms": 30, "stages": []}})


def _wait_live(page):
    page.wait_for_selector("#dot.ok", timeout=5000)


def test_stale_response_suppressed_newest_wins(page):
    def handler(route):
        body = route.request.post_data or ""
        if "QUERY_A" in body:
            route.fulfill(status=200, content_type="application/json", body=_answer("STALE_OLD"))
        else:
            route.fulfill(status=200, content_type="application/json", body=_answer("FRESH_NEW"))
    page.route("**/ask_text", handler)
    _wait_live(page)
    # fire A then B in the same tick: B must abort A; only B may render
    page.evaluate("() => { window.askText('QUERY_A here'); window.askText('QUERY_B here'); }")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FRESH_NEW')", timeout=5000)
    page.wait_for_timeout(300)
    assert "STALE_OLD" not in page.inner_text("#answer")
    assert page.errors == []


def test_timeout_fires_deadline_and_does_not_retry(harness):
    calls = {"n": 0}

    def setup(p):
        p.clock.install()
        def handler(route):
            calls["n"] += 1
            # never resolve → force the client-side deadline to fire
        p.route("**/ask_text", handler)
    page = harness.new(setup=setup)
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.fill("#q", "hang forever")
    page.click("#ask")
    page.wait_for_timeout(100)          # let the request start
    page.clock.fast_forward(13000)      # jump past the 12 s deadline
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('timed out')", timeout=5000)
    assert calls["n"] == 1              # exactly one attempt — no retry on timeout
    assert "bad" not in (page.get_attribute("#dot", "class") or "")   # health not flipped offline


def test_one_retry_on_network_failure(page):
    calls = {"n": 0}

    def handler(route):
        calls["n"] += 1
        if calls["n"] == 1:
            route.abort()               # genuine network failure (fetch TypeError)
        else:
            route.fulfill(status=200, content_type="application/json", body=_answer("RECOVERED"))
    page.route("**/ask_text", handler)
    _wait_live(page)
    page.fill("#q", "flaky")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('RECOVERED')", timeout=5000)
    assert calls["n"] == 2              # one retry, then success


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_error_no_retry_and_keeps_backend_live(page, status):
    calls = {"n": 0, "urls": []}

    def handler(route):
        calls["n"] += 1
        calls["urls"].append(route.request.url.split("/ask")[0])
        route.fulfill(status=status, content_type="application/json", body='{"detail":"nope"}')
    page.route("**/ask_text", handler)
    _wait_live(page)
    page.fill("#q", "boom")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('decision').textContent === 'Error'", timeout=5000)
    if status >= 500:
        # an unapproved 5xx FAILS OVER across backends (the compiled edge slot makes two
        # candidates) — the invariant is per-backend: each host tried exactly once
        assert calls["n"] == len(set(calls["urls"])), calls["urls"]
    else:
        assert calls["n"] == 1                               # 4xx/429: a response, no retry, no failover
    assert str(status) in page.inner_text("#answer")
    dot = page.get_attribute("#dot", "class") or ""
    assert "bad" not in dot and "ok" in dot                  # a query error is NOT a health outage


def test_malformed_json_no_retry(page):
    calls = {"n": 0}

    def handler(route):
        calls["n"] += 1
        route.fulfill(status=200, content_type="application/json", body="<<<not json>>>")
    page.route("**/ask_text", handler)
    _wait_live(page)
    page.fill("#q", "garbage")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('unexpected response')",
        timeout=5000)
    assert calls["n"] == 1
    assert "bad" not in (page.get_attribute("#dot", "class") or "")


def test_ime_composition_enter_is_ignored(page):
    calls = {"n": 0}

    def handler(route):
        calls["n"] += 1
        route.fulfill(status=200, content_type="application/json", body=_answer("SENT"))
    page.route("**/ask_text", handler)
    _wait_live(page)
    # Enter while composing (IME candidate open) must NOT submit
    page.evaluate("""() => {
        const q = document.getElementById('q'); q.value = 'कॉर्पोरेशन';
        q.dispatchEvent(new KeyboardEvent('keydown',
            {key:'Enter', isComposing:true, bubbles:true, cancelable:true}));
    }""")
    page.wait_for_timeout(300)
    assert calls["n"] == 0
    # a committed Enter (not composing) submits
    page.evaluate("""() => {
        const q = document.getElementById('q'); q.value = 'कॉर्पोरेशन';
        q.dispatchEvent(new KeyboardEvent('keydown',
            {key:'Enter', isComposing:false, bubbles:true, cancelable:true}));
    }""")
    page.wait_for_function("() => document.getElementById('answer').textContent.includes('SENT')",
                           timeout=5000)
    assert calls["n"] == 1
