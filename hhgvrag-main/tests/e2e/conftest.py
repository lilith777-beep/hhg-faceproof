"""
Playwright e2e harness for web/index.html (hhgvrag Opus-UI acceptance suite).

No pytest-playwright dependency: we drive playwright.sync_api directly and own the
browser/context/page lifecycle here. Most tests stub the backend with page.route;
a small number of `live` tests hit the real Forge backend for happy paths only.

Run:  .venv/Scripts/python -m pytest tests/e2e -v
"""
import json
import pathlib
import pytest
from playwright.sync_api import sync_playwright

E2E_DIR = pathlib.Path(__file__).resolve().parent
REPO = E2E_DIR.parent.parent
PAGE = REPO / "web" / "index.html"
PAGE_URL = PAGE.as_uri()                      # file:///F:/HHG/hhgvrag/web/index.html
AXE_JS = E2E_DIR / "vendor" / "axe.min.js"
SILENCE_WAV = E2E_DIR / "fixtures" / "silence.wav"

DEFAULT_BACKEND = "https://goquest-z790-aorus-elite-ax.tail16e418.ts.net"

# Chromium flags that grant a fake mic + auto-accept the permission prompt.
BASE_ARGS = [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
]

# A minimal but schema-faithful /ask_text (and /ask) response for stubbed renders.
CANNED = {
    "answer": "A corporation is a legal entity distinct from its owners. [1]",
    "decision": "answer",
    "citations": [{"chunk_id": "en-10::passage::0", "doc_id": "en-10", "passage_id": 0,
                   "text": "A corporation is a company or group authorized to act as a single entity."}],
    "abstained": False, "reason": None,
    "trace": {
        "query": "what is a corporation?", "decision": "answer", "total_ms": 51.4,
        "stages": [
            {"stage": "safety", "ms": 0.05, "ok": True},
            {"stage": "embed", "ms": 13.5, "ok": True},
            {"stage": "retrieve", "ms": 6.7, "ok": True},
            {"stage": "rerank", "ms": 26.1, "ok": True},
            {"stage": "generate", "ms": 0.3, "ok": True},
            {"stage": "ground_check", "ms": 0.16, "ok": True},
        ],
        "ood": {"in_domain": True, "top_score": 0.69, "signal": "dense"},
        "route": {"language": "en", "intent": "qa"},
        "normalized": None, "rerank_top": 0.999, "cache_hit": False,
        "quality_mode": False, "retrieval_mode": "hybrid", "n_retrieved": 24,
    },
}
CANNED_JSON = json.dumps(CANNED)

HEALTH_OK = '{"status":"ok","collection":"msmarco_xi__raptor","budget_ms":200}'


def _fulfill_json(route, body, status=200):
    route.fulfill(status=status, content_type="application/json", body=body)


@pytest.fixture(scope="session")
def pw():
    with sync_playwright() as p:
        yield p


class Harness:
    """Factory that spins up isolated browser/context/page trios with custom args,
    permissions, pre-navigation init scripts and routes; tears them all down."""

    def __init__(self, pw):
        self.pw = pw
        self._alive = []

    def new(self, *, extra_args=(), permissions=("microphone",), viewport=None,
            init_scripts=(), setup=None, goto=True, stub_health=True):
        browser = self.pw.chromium.launch(args=BASE_ARGS + list(extra_args))
        context = browser.new_context(
            permissions=list(permissions),
            viewport=viewport or {"width": 1000, "height": 900},
        )
        page = context.new_page()
        page.errors = []
        page.on("pageerror", lambda e: page.errors.append(str(e)))
        # console.error is collected, EXCEPT the browser's own network resource-load reports
        # (e.g. "Failed to load resource: ... 404") — expected in negative-path/capability-probe
        # tests and not a page defect; real JS exceptions always arrive via pageerror.
        page.on("console", lambda m: page.errors.append("console.error: " + m.text)
                if m.type == "error" and "Failed to load resource" not in m.text else None)
        for s in init_scripts:
            page.add_init_script(s)
        if stub_health:
            page.route("**/health", lambda route: _fulfill_json(route, HEALTH_OK))
        if setup:
            setup(page)
        if goto:
            page.goto(PAGE_URL)
        self._alive.append((browser, context))
        return page

    def close(self):
        for browser, context in self._alive:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass
        self._alive = []


@pytest.fixture
def harness(pw):
    h = Harness(pw)
    yield h
    h.close()


@pytest.fixture
def page(harness):
    """Convenience: a loaded page with /health stubbed OK. Add /ask_text routes after load."""
    return harness.new()
