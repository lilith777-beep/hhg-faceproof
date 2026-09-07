"""Smoke: page loads, brand intact, stubbed health goes live, no JS errors."""
from conftest import CANNED_JSON


def test_page_loads_and_health_ok(page):
    assert "hhgvrag" in page.title()
    # brand preserved
    assert page.inner_text(".mark").strip().startswith("hhg")
    assert "#RAGInGoa" in page.inner_text(".brand")
    # stubbed /health → neutral live copy (never the collection name)
    page.wait_for_selector("#dot.ok", timeout=5000)
    assert page.inner_text("#statustext") == "live · corpus ready"
    assert "msmarco" not in page.inner_text("#statustext")
    assert page.errors == []


def test_text_ask_renders(page):
    page.route("**/ask_text", lambda r: r.fulfill(
        status=200, content_type="application/json", body=CANNED_JSON))
    page.fill("#q", "what is a corporation?")
    page.click("#ask")
    page.wait_for_function("() => document.getElementById('decision').textContent.includes('Answered')",
                           timeout=5000)
    assert "corporation" in page.inner_text("#answer")
    assert "51.4 ms" in page.inner_text("#total")
    assert page.errors == []
