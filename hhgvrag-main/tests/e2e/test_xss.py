"""
XSS / untrusted-backend rendering. The backend is treated as untrusted: every string
field it can return is poisoned with HTML/JS-injection payloads and we assert nothing is
ever parsed as HTML or executed (full DOM-construction render, no innerHTML sinks).
"""
import json

TAG = '<img src=x onerror="window.__xss=1">'
ATTR = '" onmouseover="window.__xss=1'          # title/attribute breakout attempt
SCRIPT = '</div><script>window.__xss=1</script>'


def _poisoned():
    return {
        "answer": TAG,
        "decision": "answer",
        "citations": [{
            "text": TAG, "doc_id": ATTR, "passage_id": SCRIPT, "chunk_id": TAG,
        }],
        "trace": {
            "query": ATTR,
            "total_ms": 42.0,
            "stages": [
                {"stage": ATTR, "ms": 5.0, "ok": True},
                {"stage": TAG, "ms": 10.0, "ok": True},
            ],
            "ood": {"signal": TAG, "top_score": 0.5},
            "route": {"language": ATTR, "intent": TAG},
            "normalized": SCRIPT,
            "rerank_top": 0.5, "cache_hit": True, "quality_mode": True,
            "retrieval_mode": "pageindex",
        },
    }


def test_every_field_is_inert(page):
    page.route("**/ask_text", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(_poisoned())))
    page.fill("#q", "attack")
    page.click("#ask")
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)

    # open the collapsibles so citations + legend are in the DOM/painted
    page.eval_on_selector("#citebox", "e => e.open = true")
    page.eval_on_selector(".legendbox", "e => e.open = true")
    # hover a segment to trigger any (would-be) onmouseover handler
    seg = page.query_selector("#scale .seg")
    if seg:
        seg.hover()
    page.wait_for_timeout(150)

    # 1. the injection sentinel was never set
    assert page.evaluate("() => window.__xss") is None
    # 2. no element was created from the markup
    assert page.query_selector("#cites img") is None
    assert page.query_selector("#signals img") is None
    assert page.query_selector("#scale script, #cites script") is None
    # 3. no injected event-handler attribute anywhere
    assert page.query_selector("[onmouseover]") is None
    assert page.query_selector("[onerror]") is None
    # 4. the payload survived as *text* (proves it was escaped, not dropped)
    assert "<img" in page.inner_text("#answer")
    assert "onmouseover" in page.eval_on_selector("#scale .seg", "e => e.title")
    # 5. an unsafe language value is sanitised to a strict-allowlist code
    assert page.get_attribute("#answer", "lang") == "en"
    assert page.errors == []
