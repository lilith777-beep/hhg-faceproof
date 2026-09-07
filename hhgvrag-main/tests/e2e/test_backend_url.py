"""
Backend-URL migration to `hhgvrag_backend_v2`: validate stored URLs (HTTPS required except
localhost), use a valid stored override, self-heal a dead override by falling back once to
the compiled default and deleting the stale key, and reject non-HTTPS on save.
"""
from conftest import DEFAULT_BACKEND, HEALTH_OK, CANNED_JSON

BK = "hhgvrag_backend_v2"


def _seed(key_value):
    return f"try{{localStorage.setItem('{BK}', {key_value!r});}}catch(e){{}}"


def test_dead_override_falls_back_to_default_and_clears_key(harness):
    def setup(p):
        p.route("https://dead.example.invalid/health", lambda r: r.abort())
        p.route(DEFAULT_BACKEND + "/health",
                lambda r: r.fulfill(status=200, content_type="application/json", body=HEALTH_OK))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed("https://dead.example.invalid")],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=8000)
    assert page.input_value("#backend") == DEFAULT_BACKEND
    assert page.evaluate(f"() => localStorage.getItem('{BK}')") is None   # stale key deleted
    assert page.inner_text("#statustext") == "live · corpus ready"


def test_invalid_stored_url_is_dropped(harness):
    page = harness.new(init_scripts=[_seed("http://evil.example.com")])   # http, non-local → invalid
    page.wait_for_selector("#dot.ok", timeout=5000)
    assert page.input_value("#backend") == DEFAULT_BACKEND
    assert page.evaluate(f"() => localStorage.getItem('{BK}')") is None


def test_valid_stored_override_is_used_for_requests(harness):
    custom = "https://custom.example.com"

    def setup(p):
        p.route(custom + "/ask_text",
                lambda r: r.fulfill(status=200, content_type="application/json", body=CANNED_JSON))
    page = harness.new(init_scripts=[_seed(custom)], setup=setup)   # health globally stubbed
    page.wait_for_selector("#dot.ok", timeout=5000)
    assert page.input_value("#backend") == custom
    # prove api() actually targets the stored override
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    assert "corporation" in page.inner_text("#answer")


def test_save_rejects_non_https(page):
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.eval_on_selector("footer details", "d => d.open = true")
    page.fill("#backend", "http://example.com")
    page.click("#save")
    assert page.evaluate(f"() => localStorage.getItem('{BK}')") is None   # nothing saved
    assert page.get_attribute("#backend", "aria-invalid") == "true"
    assert "https" in page.inner_text("#save").lower()


def test_save_allows_localhost_http(page):
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.eval_on_selector("footer details", "d => d.open = true")
    page.fill("#backend", "http://localhost:8000")
    page.click("#save")
    assert page.evaluate(f"() => localStorage.getItem('{BK}')") == "http://localhost:8000"
