"""
Accessibility + truthful-UI acceptance: axe (no serious/critical) on the empty and the
rendered states, aria-live scoped to only the decision/transcript/answer region, `lang`
from a strict allowlist, >=44px touch targets at mobile width, keyboard-only text flow,
and a focusable/labelled mic.
"""
from conftest import AXE_JS, CANNED_JSON


def _axe_serious_or_critical(page):
    page.add_script_tag(path=str(AXE_JS))
    res = page.evaluate("""async () => {
        const r = await axe.run(document, { resultTypes: ['violations'] });
        return r.violations.map(v => ({ id: v.id, impact: v.impact,
            nodes: v.nodes.map(n => n.target).slice(0, 3) }));
    }""")
    return [v for v in res if v["impact"] in ("serious", "critical")]


def test_axe_clean_on_load(page):
    page.wait_for_selector("#dot.ok", timeout=5000)
    bad = _axe_serious_or_critical(page)
    assert bad == [], bad


def test_axe_clean_after_render(page):
    page.route("**/ask_text", lambda r: r.fulfill(
        status=200, content_type="application/json", body=CANNED_JSON))
    page.fill("#q", "what is a corporation?")
    page.click("#ask")
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    page.eval_on_selector("#citebox", "e => e.open = true")
    page.eval_on_selector(".legendbox", "e => e.open = true")
    bad = _axe_serious_or_critical(page)
    assert bad == [], bad


def test_live_region_scoped_to_answer_only(page):
    assert page.get_attribute(".announce", "aria-live") == "polite"
    # everything else must NOT be a live region (or it would spam the screen reader)
    for sel in ["#result", "#budget", "#citebox", "#signals", "#why", ".status"]:
        assert page.get_attribute(sel, "aria-live") is None, sel


def test_lang_from_strict_allowlist(page):
    page.evaluate("(p) => render(p)", {"answer": "नमस्ते", "decision": "answer",
        "trace": {"route": {"language": "hi"}, "total_ms": 30, "stages": []}})
    assert page.get_attribute("#answer", "lang") == "hi"
    # an unknown code falls back to en
    page.evaluate("(p) => render(p)", {"answer": "x", "decision": "answer",
        "trace": {"route": {"language": "xx"}, "total_ms": 30, "stages": []}})
    assert page.get_attribute("#answer", "lang") == "en"
    # a hostile value is sanitised to en, never injected
    page.evaluate("(p) => render(p)", {"answer": "x", "decision": "answer",
        "trace": {"route": {"language": '"><script>alert(1)</script>'}, "total_ms": 30, "stages": []}})
    assert page.get_attribute("#answer", "lang") == "en"


def test_dir_rtl_for_urdu_scoped_to_content(page):
    # Urdu (RTL) answers + transcript get dir=rtl; the surrounding LTR chrome is untouched
    page.evaluate("(p) => render(p)", {"answer": "اردو متن", "decision": "answer",
        "trace": {"query": "اردو سوال", "route": {"language": "ur"}, "total_ms": 30, "stages": []}})
    assert page.get_attribute("#answer", "dir") == "rtl"
    assert page.get_attribute("#transcript", "dir") == "rtl"
    # a non-RTL language resets to ltr (no sticky direction across renders)
    page.evaluate("(p) => render(p)", {"answer": "hello", "decision": "answer",
        "trace": {"query": "hi", "route": {"language": "en"}, "total_ms": 30, "stages": []}})
    assert page.get_attribute("#answer", "dir") == "ltr"
    assert page.get_attribute("#transcript", "dir") == "ltr"


def test_touch_targets_at_mobile_width(harness):
    page = harness.new(viewport={"width": 390, "height": 844})
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.eval_on_selector("footer details", "d => d.open = true")   # reveal #save
    for sel in ["#ask", "#mic", ".ex", "#save", ".opt"]:
        box = page.query_selector(sel).bounding_box()
        assert box["height"] >= 44, (sel, box)
        assert box["width"] >= 44, (sel, box)


def test_keyboard_only_text_flow(page):
    page.route("**/ask_text", lambda r: r.fulfill(
        status=200, content_type="application/json", body=CANNED_JSON))
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.focus("#q")
    page.keyboard.type("what is a corporation?")
    page.keyboard.press("Enter")               # committed Enter (not composing) submits
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    assert "corporation" in page.inner_text("#answer")


def test_mic_is_focusable_and_labelled(page):
    assert page.get_attribute("#mic", "aria-label") == "Hold to talk"
    assert page.eval_on_selector("#mic", "e => e.tagName") == "BUTTON"
    page.focus("#mic")
    assert page.evaluate("() => document.activeElement.id") == "mic"
