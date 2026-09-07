"""
Defensive rendering: hostile / missing / out-of-range numeric fields must never throw,
must coerce to finite values, and the budget-bar geometry must stay in bounds. Also pins
the limit-marker math the break-tester verified as correct.
"""
import json


def _ask(page, payload):
    page.route("**/ask_text", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(payload)))
    page.fill("#q", "q")
    page.click("#ask")
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)


def test_hostile_metrics_do_not_throw(page):
    payload = {
        "answer": "ok", "decision": "answer",
        "trace": {
            "query": "q",
            "total_ms": "NaN",
            "stages": [
                {"stage": "embed", "ms": "not-a-number", "ok": True},
                {"stage": "retrieve", "ms": float("1e9"), "ok": True},   # oversized
                {"stage": "rerank", "ms": -10, "ok": True},              # negative
                {"stage": "generate", "ms": None, "ok": False},
            ],
            "rerank_top": "foo", "ood": {"top_score": None, "signal": "dense"},
            "route": {"language": "en"},
        },
    }
    _ask(page, payload)
    # no uncaught error; total coerced to a finite string
    total = page.inner_text("#total")
    assert "NaN" not in total and "Infinity" not in total and "undefined" not in total
    # every segment width is a finite percentage within [0, 100]
    widths = page.evaluate(
        "() => Array.from(document.querySelectorAll('#scale .seg'))"
        ".map(s => parseFloat(s.style.width))")
    assert widths, "expected segments to render"
    assert all(w == w and 0 <= w <= 100 for w in widths), widths
    assert page.errors == []


def test_negative_total_clamped(page):
    _ask(page, {"answer": "x", "decision": "answer", "trace": {"total_ms": -5, "stages": []}})
    assert page.inner_text("#total").startswith("0.0")
    assert page.errors == []


def test_limit_marker_at_50_percent_when_total_400(page):
    _ask(page, {"answer": "x", "decision": "answer", "trace": {
        "total_ms": 400,
        "stages": [{"stage": "embed", "ms": 200, "ok": True},
                   {"stage": "rerank", "ms": 200, "ok": True}]}})
    ratio = page.evaluate("""() => {
      const s = document.getElementById('scale').getBoundingClientRect();
      const l = document.getElementById('limit').getBoundingClientRect();
      return (l.left - s.left) / s.width; }""")
    assert abs(ratio - 0.5) < 0.03, ratio


def test_limit_marker_at_100_percent_when_total_under_budget(page):
    _ask(page, {"answer": "x", "decision": "answer", "trace": {
        "total_ms": 100, "stages": [{"stage": "embed", "ms": 100, "ok": True}]}})
    ratio = page.evaluate("""() => {
      const s = document.getElementById('scale').getBoundingClientRect();
      const l = document.getElementById('limit').getBoundingClientRect();
      return (l.left - s.left) / s.width; }""")
    assert ratio > 0.95, ratio


def test_missing_answer_is_blank_not_undefined(page):
    _ask(page, {"decision": "abstain_ood", "trace": {"total_ms": 30, "stages": []}})
    assert page.inner_text("#answer") == ""
    assert "Declined" in page.text_content("#decision")   # text_content: raw DOM (badge is CSS-uppercased)


def test_unknown_decision_falls_back(page):
    _ask(page, {"answer": "hi", "decision": "totally_made_up", "trace": {"total_ms": 30, "stages": []}})
    assert page.inner_text("#decision").strip() == "—"
    assert page.errors == []
