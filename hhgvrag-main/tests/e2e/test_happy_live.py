"""
Happy paths against the REAL Forge backend (no stubs). Guarded by a reachability check so
the suite stays green offline. Deliberately few requests — happy paths only, no hammering.
"""
import urllib.request
import pytest
from conftest import DEFAULT_BACKEND

KNOWN_DECISIONS = {"Answered", "Declined", "Refused", "Assistant", "Error"}


def _backend_up():
    try:
        with urllib.request.urlopen(DEFAULT_BACKEND + "/health", timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _backend_up(), reason="live Forge backend unreachable")


def _decision_settled(page, timeout=20000):
    page.wait_for_function(
        "() => { const d = document.getElementById('decision').textContent.trim();"
        " return d && d !== '—' && d !== '…'; }", timeout=timeout)


def test_live_text_english(harness):
    page = harness.new(stub_health=False)
    page.wait_for_selector("#dot.ok", timeout=12000)
    assert page.inner_text("#statustext") == "live · corpus ready"
    page.fill("#q", "what is a corporation?")
    page.click("#ask")
    _decision_settled(page)
    assert page.text_content("#decision").split("·")[0].strip() in KNOWN_DECISIONS
    assert page.inner_text("#answer").strip() != ""       # this query answers
    assert "ms" in page.inner_text("#total")
    assert page.errors == []


def test_live_text_hindi(harness):
    page = harness.new(stub_health=False)
    page.wait_for_selector("#dot.ok", timeout=12000)
    page.fill("#q", "कॉर्पोरेशन क्या है?")
    page.click("#ask")
    _decision_settled(page)
    assert page.text_content("#decision").split("·")[0].strip() in KNOWN_DECISIONS
    assert "ms" in page.inner_text("#total")
    assert page.errors == []

# NOTE: the audio path against the real backend is intentionally not a happy-path test —
# the Chromium fake device emits a tone, not speech, so the backend's response to non-speech
# is not a meaningful "happy" assertion. The full record→upload→render audio pipeline is
# covered deterministically (real MediaRecorder, stubbed /ask) in
# test_recorder.py::test_real_recording_uploads_webm_and_renders.
