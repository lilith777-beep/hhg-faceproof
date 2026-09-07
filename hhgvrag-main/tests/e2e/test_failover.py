"""
Automatic multi-backend failover. The frontend holds a priority-ordered list of
{url, tier} backends (persisted in hhgvrag_backends_v1). Each query routes to the
highest-priority backend believed live; a genuine backend-DOWN failure (network error,
per-attempt timeout, or an unapproved 5xx) marks it dead and fails the SAME query over to
the next live backend, retrying once there. It never fails over on a normal/abstain/error
RESPONSE, a 4xx, a malformed body, a user-abort, or a superseded request. An empty fallback
slot is skipped entirely (never health-checked). A small tier badge reports which backend
answered — "full" (Forge) vs "edge" (laptop) — and the advanced list is XSS-safe.

These tests seed hhgvrag_backends_v1 with two DISTINCT hosts so health/ask can be routed
per-backend, and drive the real request executor (no monkeypatching of app internals).
"""
import json

PRIMARY = "https://primary.example.test"
SECONDARY = "https://secondary.example.test"
BK_LIST = "hhgvrag_backends_v1"


def _seed_list(entries):
    """init_script: seed the v1 backend list (entries = [(url, tier), ...])."""
    payload = json.dumps([{"url": u, "tier": t} for (u, t) in entries])
    return f"try{{localStorage.setItem('{BK_LIST}', {payload!r});}}catch(e){{}}"


def _answer(text, decision="answer"):
    return json.dumps({"answer": text, "decision": decision,
                       "trace": {"query": "q", "total_ms": 30, "stages": []}})


def _ok(route):
    route.fulfill(status=200, content_type="application/json",
                  body='{"status":"ok","budget_ms":200}')


def _fulfill(route, body):
    route.fulfill(status=200, content_type="application/json", body=body)


def _tier_text(page):
    return (page.text_content("#tier") or "").strip()


def _tier_class(page):
    return page.get_attribute("#tier", "class") or ""


# ---------- 1. primary live → uses primary, tier "full" ----------

def test_primary_live_uses_primary_full_tier(harness):
    calls = {"primary": 0, "secondary": 0}

    def setup(p):
        p.route(PRIMARY + "/health", _ok)
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text",
                lambda r: (calls.__setitem__("primary", calls["primary"] + 1),
                           _fulfill(r, _answer("FROM_PRIMARY"))))
        p.route(SECONDARY + "/ask_text",
                lambda r: (calls.__setitem__("secondary", calls["secondary"] + 1),
                           _fulfill(r, _answer("FROM_SECONDARY"))))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_PRIMARY')", timeout=5000)
    assert calls == {"primary": 1, "secondary": 0}      # highest-priority live backend, no failover
    assert _tier_text(page) == "full"
    assert "full" in _tier_class(page) and "on" in _tier_class(page)
    assert page.get_attribute("#tier", "hidden") is None
    assert page.errors == []


# ---------- 2. primary health-down → uses secondary, tier "edge" ----------

def test_primary_health_down_uses_secondary_edge_tier(harness):
    calls = {"primary": 0, "secondary": 0}

    def setup(p):
        p.route(PRIMARY + "/health", lambda r: r.abort())       # primary dead at health time
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text",
                lambda r: (calls.__setitem__("primary", calls["primary"] + 1),
                           _fulfill(r, _answer("FROM_PRIMARY"))))
        p.route(SECONDARY + "/ask_text",
                lambda r: (calls.__setitem__("secondary", calls["secondary"] + 1),
                           _fulfill(r, _answer("FROM_SECONDARY"))))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)             # secondary keeps the page live
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_SECONDARY')", timeout=5000)
    assert calls["primary"] == 0                                # dead primary is skipped, not probed
    assert calls["secondary"] == 1
    assert _tier_text(page) == "edge"
    assert "edge" in _tier_class(page)
    assert page.errors == []


# ---------- 3a. primary fails MID-request (network) → fails over to secondary ----------

def test_primary_network_fail_midrequest_fails_over(harness):
    calls = {"primary": 0, "secondary": 0}

    def setup(p):
        p.route(PRIMARY + "/health", _ok)                       # both healthy at load…
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text",                          # …but the primary query aborts (net down)
                lambda r: (calls.__setitem__("primary", calls["primary"] + 1), r.abort()))
        p.route(SECONDARY + "/ask_text",
                lambda r: (calls.__setitem__("secondary", calls["secondary"] + 1),
                           _fulfill(r, _answer("FROM_SECONDARY"))))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_SECONDARY')", timeout=6000)
    assert calls["primary"] == 2        # one attempt + the ONE network retry, on the primary
    assert calls["secondary"] == 1      # then the SAME query failed over to the secondary once
    assert _tier_text(page) == "edge"   # the badge is honest about who actually answered
    assert page.errors == []


# ---------- 3b. primary hangs MID-request (per-attempt timeout) → fails over ----------

def test_primary_timeout_midrequest_fails_over(harness):
    calls = {"primary": 0, "secondary": 0}

    def setup(p):
        p.clock.install()
        p.route(PRIMARY + "/health", _ok)
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text",                          # primary hangs forever mid-request
                lambda r: calls.__setitem__("primary", calls["primary"] + 1))
        p.route(SECONDARY + "/ask_text",
                lambda r: (calls.__setitem__("secondary", calls["secondary"] + 1),
                           _fulfill(r, _answer("FROM_SECONDARY"))))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_timeout(100)          # let the primary request start
    page.clock.fast_forward(7000)       # past the 6 s per-attempt soft timeout (still under 12 s global)
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_SECONDARY')", timeout=6000)
    assert calls["primary"] == 1        # hung once, no network retry (an abort is not a TypeError)
    assert calls["secondary"] == 1      # failed over and answered
    assert _tier_text(page) == "edge"
    assert page.errors == []


# ---------- 4. an empty fallback slot is skipped (never health-checked) ----------

def test_empty_fallback_slot_skipped_no_health_call(harness):
    health_urls = []

    def setup(p):
        p.on("request", lambda r: health_urls.append(r.url) if "/health" in r.url else None)
        p.route(PRIMARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text", lambda r: _fulfill(r, _answer("FROM_PRIMARY")))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), ("", "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    # ONLY the non-empty primary was ever probed — the empty edge slot triggered no /health fetch
    assert health_urls, "expected at least one health probe"
    assert all("primary.example.test" in u for u in health_urls), health_urls
    # and the app is fully functional on the primary alone
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_PRIMARY')", timeout=5000)
    assert _tier_text(page) == "full"
    assert page.errors == []


# ---------- 5. both down → honest error, no fabricated answer ----------

def test_both_down_honest_error_no_fabrication(harness):
    calls = {"primary": 0, "secondary": 0}

    def setup(p):
        p.route(PRIMARY + "/health", _ok)                       # both "live" at load…
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text",                          # …but every query aborts on both
                lambda r: (calls.__setitem__("primary", calls["primary"] + 1), r.abort()))
        p.route(SECONDARY + "/ask_text",
                lambda r: (calls.__setitem__("secondary", calls["secondary"] + 1), r.abort()))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('decision').textContent === 'Error'", timeout=6000)
    # every live backend was tried (1 + 1 retry each) then the query failed honestly
    assert calls["primary"] == 2 and calls["secondary"] == 2
    ans = page.inner_text("#answer")
    assert "unreachable" in ans                                # the honest failure copy
    assert "FROM_" not in ans                                  # NO fabricated grounded answer
    assert page.eval_on_selector("#tier", "e => e.hidden") is True   # no fidelity badge on an error
    assert page.errors == []


# ---------- 6. a newer user request cancels an in-flight failover ----------

def test_newer_request_cancels_inflight_failover(harness):
    seen = {"primary": 0, "sec_A": 0, "sec_B": 0}

    def setup(p):
        p.route(PRIMARY + "/health", _ok)
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text",                          # primary always aborts → forces failover
                lambda r: (seen.__setitem__("primary", seen["primary"] + 1), r.abort()))

        def sec(route):
            body = route.request.post_data or ""
            if "QUERY_A" in body:
                seen["sec_A"] += 1                              # A's failover reaches the secondary… and hangs
            else:
                seen["sec_B"] += 1
                _fulfill(route, _answer("FROM_SECONDARY_B"))    # B's failover answers
        p.route(SECONDARY + "/ask_text", sec)
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)

    # A: fire-and-forget (do NOT let evaluate await the hanging promise) — primary aborts, fails over
    page.evaluate("() => { window.askText('QUERY_A please'); }")
    # wait until A is genuinely in-flight ON the secondary (its failover retry)
    waited = 0
    while seen["sec_A"] < 1 and waited < 5000:
        page.wait_for_timeout(50); waited += 50
    assert seen["sec_A"] == 1, "A should be mid-failover on the secondary"

    page.fill("#q", "QUERY_B please")
    page.click("#ask")                                          # B supersedes the in-flight failover
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_SECONDARY_B')", timeout=6000)
    # only B rendered; A's hung failover was aborted and never surfaced
    assert seen["sec_B"] == 1
    assert page.inner_text("#answer").strip() == "FROM_SECONDARY_B"
    assert _tier_text(page) == "edge"
    assert page.errors == []


# ---------- 7. XSS-inert backend URLs in the advanced list ----------

def test_backend_list_is_xss_inert(harness):
    hostile = 'https://evil.example.test/<img src=x onerror="window.__xss=1">'

    def setup(p):
        # a syntactically-valid https URL whose path carries an injection payload
        p.route("**/health", _ok)
    page = harness.new(
        stub_health=False,
        init_scripts=[_seed_list([(hostile, '<script>bad</script>'), (SECONDARY, "edge")])],
        setup=setup)
    page.wait_for_timeout(200)
    page.eval_on_selector("footer details", "d => d.open = true")   # reveal the list
    page.wait_for_timeout(100)
    # 1. nothing executed
    assert page.evaluate("() => window.__xss") is None
    # 2. the markup never became elements / handlers inside the list
    assert page.query_selector("#backendlist img") is None
    assert page.query_selector("#backendlist [onerror]") is None
    assert page.query_selector("#backendlist script") is None
    # 3. the hostile URL survived as an inert input VALUE (a property, never parsed as HTML)
    primary_val = page.input_value("#backend")
    assert "<img" in primary_val and "onerror" in primary_val
    # 4. a hostile tier is sanitised to the strict allowlist, never injected as text/markup
    tiers = page.eval_on_selector_all("#backendlist .bktier", "els => els.map(e => e.textContent)")
    assert all(t in ("full", "edge") for t in tiers), tiers
    assert page.errors == []


# ---------- 8z. a 200 decision:"error" from the primary fails over to the secondary ----------

def test_decision_error_fails_over_to_secondary(harness):
    """A funnel/relay hiccup makes the primary return HTTP 200 with decision:"error" — a
    typed server failure, not a real answer. The SAME query must fail over to a healthy
    secondary instead of showing 'Error', and the primary must NOT be dead-marked."""
    calls = {"primary": 0, "secondary": 0}

    def setup(p):
        p.route(PRIMARY + "/health", _ok)
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text",
                lambda r: (calls.__setitem__("primary", calls["primary"] + 1),
                           _fulfill(r, _answer("PRIMARY_ERR", decision="error"))))
        p.route(SECONDARY + "/ask_text",
                lambda r: (calls.__setitem__("secondary", calls["secondary"] + 1),
                           _fulfill(r, _answer("FROM_SECONDARY"))))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_SECONDARY')", timeout=6000)
    assert calls["primary"] == 1                 # tried once, decision:error -> failover (no net retry)
    assert calls["secondary"] == 1
    assert _tier_text(page) == "edge"
    # primary is NOT dead-marked: a query error isn't a backend outage
    assert page.evaluate("() => backends.find(b => b.tier==='full').live") is not False
    assert page.errors == []


def test_decision_error_on_all_backends_renders_honest_error(harness):
    """When every backend returns decision:error, the last typed error renders as-is
    (localized error copy) — never a fabricated answer, never a blank card."""
    def setup(p):
        p.route(PRIMARY + "/health", _ok)
        p.route(SECONDARY + "/health", _ok)
        p.route(PRIMARY + "/ask_text", lambda r: _fulfill(r, _answer("E1", decision="error")))
        p.route(SECONDARY + "/ask_text", lambda r: _fulfill(r, _answer("E2", decision="error")))
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('decision').textContent === 'Error'", timeout=6000)
    assert page.errors == []


# ---------- 8a. a stale stored list (saved before the edge slot shipped) still fails over ----------

def test_stale_stored_list_merges_compiled_edge_slot(harness):
    """A one-row list persisted before the edge tier existed must not pin the page to a
    dead primary: the compiled-in edge default is merged at load (runtime-only) and the
    stored list itself is NOT rewritten by the merge."""
    calls = {"edge_ask": 0}

    def setup(p):
        p.route(PRIMARY + "/health", lambda r: r.abort())        # stored primary is down

        def edge(route):
            if route.request.url.endswith("/health"):
                _ok(route)
            else:
                calls["edge_ask"] += 1
                _fulfill(route, _answer("FROM_COMPILED_EDGE"))
        p.route("https://*.trycloudflare.com/**", edge)          # compiled slot, any rotation
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full")])],   # pre-edge-slot save
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)              # merged edge keeps the page live
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_COMPILED_EDGE')",
        timeout=6000)
    assert calls["edge_ask"] == 1
    assert _tier_text(page) == "edge"
    # runtime-only: the merge never writes storage (future bundles must keep winning)
    stored = page.evaluate(f"() => JSON.parse(localStorage.getItem('{BK_LIST}'))")
    assert [e["url"] for e in stored] == [PRIMARY]
    assert page.errors == []


# ---------- 8b. a rotated quick-tunnel URL in a stored edge row is swapped for the compiled one ----------

def test_stale_trycloudflare_edge_url_swapped_for_compiled(harness):
    """Quick-tunnel URLs rotate on restart; a stored edge row holding an old rotation is
    swapped at load for the bundle's fresher compiled URL. The custom primary row and the
    old dead URL are never probed."""
    OLD = "https://old-rotation-gone.trycloudflare.com"
    old_hits = []

    def setup(p):
        p.route(PRIMARY + "/health", lambda r: r.abort())

        def edge(route):
            if "old-rotation-gone" in route.request.url:
                old_hits.append(route.request.url)
                route.abort()
            elif route.request.url.endswith("/health"):
                _ok(route)
            else:
                _fulfill(route, _answer("FROM_FRESH_EDGE"))
        p.route("https://*.trycloudflare.com/**", edge)
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (OLD, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)
    page.fill("#q", "hi")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('FROM_FRESH_EDGE')",
        timeout=6000)
    assert old_hits == []                                        # dead rotation never touched
    assert _tier_text(page) == "edge"
    assert page.errors == []


# ---------- 8. the tier badge reflects the answering backend (full ↔ edge, hidden on error) ----------

def test_tier_badge_tracks_answering_backend(harness):
    """A run that answers on the edge shows edge/amber; a later error hides the badge entirely."""
    state = {"mode": "answer"}

    def setup(p):
        p.route(PRIMARY + "/health", lambda r: r.abort())       # force routing to the edge
        p.route(SECONDARY + "/health", _ok)

        def sec(route):
            if state["mode"] == "answer":
                _fulfill(route, _answer("EDGE_ANSWER"))
            else:
                route.abort()
        p.route(SECONDARY + "/ask_text", sec)
    page = harness.new(stub_health=False,
                       init_scripts=[_seed_list([(PRIMARY, "full"), (SECONDARY, "edge")])],
                       setup=setup)
    page.wait_for_selector("#dot.ok", timeout=6000)

    # edge answer → amber "edge" badge, visible + labelled
    page.fill("#q", "one")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('answer').textContent.includes('EDGE_ANSWER')", timeout=5000)
    assert _tier_text(page) == "edge"
    assert "edge" in _tier_class(page) and "on" in _tier_class(page)
    assert (page.get_attribute("#tier", "aria-label") or "").find("edge") != -1

    # now the edge also fails → honest error, and the fidelity badge is withdrawn
    state["mode"] = "down"
    page.fill("#q", "two")
    page.click("#ask")
    page.wait_for_function(
        "() => document.getElementById('decision').textContent === 'Error'", timeout=6000)
    assert page.eval_on_selector("#tier", "e => e.hidden") is True
    assert page.errors == []
