"""
Hold-to-talk recorder state machine. Covers the generation-guarded races and the
"no STT spend on empty/short/silent" guarantees.

`page.mouse.down/up` over the disc produces trusted pointer events (Chromium synthesizes
pointerdown/up from mouse input), which is what the state machine listens to. Real
recordings use the Chromium fake mic; where first-call getUserMedia latency would make
sub-second holds nondeterministic (negotiation / short-tap / release-timing), we stub
getUserMedia + MediaRecorder so the state transition under test is exercised exactly.
"""
from conftest import CANNED_JSON, SILENCE_WAV

# --- init scripts (run before the page script) -------------------------------

# getUserMedia that resolves only after 300 ms, returning a fake stream whose tracks
# count their own stop() calls — lets us prove a stale (released) grant is cleaned up.
GUM_DELAYED_COUNT_STOPS = """
window.__trackStops = 0;
navigator.mediaDevices.getUserMedia = () => new Promise(res => setTimeout(() => {
  res({ getTracks: () => [{ stop(){ window.__trackStops++; }, kind: 'audio' }] });
}, 300));
"""

# getUserMedia that resolves immediately with a fake stream (createMediaStreamSource will
# throw on it → the RMS meter simply disables; the recording path is unaffected).
GUM_IMMEDIATE = """
window.__fakeTracks = [{ stop(){}, kind: 'audio' }];
navigator.mediaDevices.getUserMedia = () => Promise.resolve({ getTracks: () => window.__fakeTracks });
"""

GUM_DENY = """
navigator.mediaDevices.getUserMedia = () =>
  Promise.reject(new DOMException('denied', 'NotAllowedError'));
"""

NO_MEDIA_RECORDER = "window.MediaRecorder = undefined;"

# The recorder reads performance.now() only to measure clip duration. Mock it so the
# <300 ms boundary is exact and independent of page.mouse/wait_for_timeout jitter.
PERF_NOW_MOCK = "window.__now = 1000; performance.now = () => window.__now;"

# getUserMedia that COUNTS acquisitions and returns a persistently-live fake stream. Each track
# ends only when stopped — lets us prove the warm stream is REUSED across holds (one acquisition)
# and RE-WARMED after an idle release. __tracks records every stream ever handed out.
GUM_COUNT_LIVE = """
window.__gumCalls = 0;
window.__tracks = [];
navigator.mediaDevices.getUserMedia = () => {
  window.__gumCalls++;
  const track = { kind: 'audio', readyState: 'live', stop(){ this.readyState = 'ended'; } };
  window.__tracks.push(track);
  return Promise.resolve({ getTracks: () => [track] });
};
"""

# Same, but resolves after 300 ms so the interim "arming…" cue is observable before capture is live.
GUM_DELAYED_LIVE = """
window.__gumCalls = 0;
window.__tracks = [];
navigator.mediaDevices.getUserMedia = () => new Promise(res => setTimeout(() => {
  window.__gumCalls++;
  const track = { kind: 'audio', readyState: 'live', stop(){ this.readyState = 'ended'; } };
  window.__tracks.push(track);
  res({ getTracks: () => [track] });
}, 300));
"""

def _fake_recorder(supported_mime):
    # a MediaRecorder stand-in: only `supported_mime` negotiates true; emits one 4 KB chunk.
    # start() sets state='recording' synchronously (as the real API does) then dispatches an
    # async `start` event — the honest go-cue is gated on it. window.__stateAtStart captures the
    # recorder state at the instant `start` fired, proving capture is live before the cue shows.
    return """
    class FakeRec {
      constructor(stream, opts){ this.stream = stream;
        this.mimeType = (opts && opts.mimeType) || %r; this.state = 'inactive'; }
      static isTypeSupported(m){ return m === %r; }
      start(){ this.state = 'recording';
        setTimeout(() => { window.__stateAtStart = this.state; if (this.onstart) this.onstart({}); }, 0);  // real fires `start`
        setTimeout(() => { if (this.ondataavailable)
          this.ondataavailable({ data: new Blob([new Uint8Array(4096)], { type: this.mimeType }) }); }, 20); }
      stop(){ this.state = 'inactive'; setTimeout(() => { if (this.onstop) this.onstop(); }, 0); }  // async like real
    }
    window.MediaRecorder = FakeRec;
    """ % (supported_mime, supported_mime)


def _count_ask(page):
    posts = []
    page.route("**/ask", lambda r: (posts.append(r.request.post_data_buffer),
                                    r.fulfill(status=200, content_type="application/json",
                                              body=CANNED_JSON)))
    return posts


def _press(page, hold_ms):
    page.hover("#mic")
    page.mouse.down()
    page.wait_for_timeout(hold_ms)
    page.mouse.up()


def _wait_posts(page, posts, n, timeout=6000):
    # poll the Python-side upload list while pumping the browser event loop (route handlers fire
    # during wait_for_timeout) — used where the assertion is on the number of /ask uploads.
    waited = 0
    while len(posts) < n and waited < timeout:
        page.wait_for_timeout(50)
        waited += 50
    assert len(posts) == n, f"expected {n} uploads, got {len(posts)}"


def _warm_mic(page):
    # Initialise the fake device so the app's getUserMedia resolves promptly and a sub-second
    # hold reliably starts a real recording (first-call latency otherwise nears ~800ms).
    page.evaluate("""async () => {
        try { const s = await navigator.mediaDevices.getUserMedia({ audio: true });
              s.getTracks().forEach(t => t.stop()); } catch (e) {}
    }""")


def test_release_before_permission_stops_tracks_and_uploads_nothing(harness):
    page = harness.new(init_scripts=[GUM_DELAYED_COUNT_STOPS])
    posts = _count_ask(page)
    page.hover("#mic")
    page.mouse.down()                       # pointerdown → holding set synchronously (before permission)
    assert "rec" in (page.get_attribute("#mic", "class") or "")   # pointer events reach the machine
    page.wait_for_timeout(60)
    page.mouse.up()                         # release BEFORE the 300 ms grant resolves
    page.wait_for_timeout(600)              # grant resolves into a now-stale generation
    assert page.evaluate("() => window.__trackStops") >= 1        # acquired tracks were stopped
    assert posts == []                                            # no STT spend
    assert "rec" not in (page.get_attribute("#mic", "class") or "")


def test_permission_denied_is_graceful(harness):
    page = harness.new(init_scripts=[GUM_DENY])
    posts = _count_ask(page)
    _press(page, 80)
    page.wait_for_timeout(200)
    assert page.inner_text("#hint") == "microphone permission needed"   # notice persists after release
    assert posts == []
    assert "rec" not in (page.get_attribute("#mic", "class") or "")


def test_no_media_recorder_degrades_to_text(harness):
    page = harness.new(init_scripts=[NO_MEDIA_RECORDER])
    posts = _count_ask(page)
    assert page.get_attribute("#mic", "aria-disabled") == "true"   # signalled on load
    _press(page, 80)
    page.wait_for_timeout(150)
    assert page.inner_text("#hint") == "recording not supported — type instead"
    assert posts == []


def test_mp4_only_negotiation_uploads_matching_mime_and_extension(harness):
    page = harness.new(init_scripts=[GUM_IMMEDIATE, _fake_recorder("audio/mp4")])
    posts = _count_ask(page)
    _press(page, 420)
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=5000)
    assert len(posts) == 1
    body = posts[0] or b""
    assert b'filename="audio.mp4"' in body      # extension matches the negotiated container
    assert b"audio/mp4" in body                 # part carries the actual MIME


def test_pointercancel_discards_recording(harness):
    # a system pointercancel (palm rejection, notification) must discard, never upload
    page = harness.new(init_scripts=[GUM_IMMEDIATE, _fake_recorder("audio/webm;codecs=opus")])
    posts = _count_ask(page)
    page.hover("#mic")
    page.mouse.down()                            # pointerdown → recording starts
    page.wait_for_timeout(60)
    page.dispatch_event("#mic", "pointercancel")  # → cancelRec → discard
    page.mouse.up()
    page.wait_for_timeout(200)
    assert posts == []
    assert "rec" not in (page.get_attribute("#mic", "class") or "")


def test_short_tap_is_discarded(harness):
    page = harness.new(init_scripts=[PERF_NOW_MOCK, GUM_IMMEDIATE,
                                     _fake_recorder("audio/webm;codecs=opus")])
    posts = _count_ask(page)
    page.hover("#mic")
    page.mouse.down()
    page.wait_for_timeout(40)                    # flush gum microtask → rec.start captures startedAt
    page.evaluate("() => { window.__now = 1150; }")   # 150 ms elapsed (mocked) → under the 300 ms floor
    page.mouse.up()
    page.wait_for_timeout(100)
    assert posts == []                           # short clip → no STT spend
    assert "hold a little longer" in page.inner_text("#hint")


def test_silent_recording_spends_no_stt(harness):
    page = harness.new(extra_args=[f"--use-file-for-fake-audio-capture={SILENCE_WAV.as_posix()}"])
    posts = _count_ask(page)
    _warm_mic(page)
    _press(page, 800)                           # long enough, but silent input
    page.wait_for_timeout(500)
    assert posts == []                          # silence/empty gate → no upload
    assert page.inner_text("#hint") in ("didn’t catch any speech — try again", "hold a little longer")


def test_real_recording_uploads_webm_and_renders(harness):
    # the audio happy path: a REAL fake-device recording, negotiated container, stubbed /ask
    page = harness.new()
    posts = _count_ask(page)
    _warm_mic(page)
    page.hover("#mic")
    page.mouse.down()
    page.wait_for_timeout(800)
    page.mouse.up()
    page.wait_for_function("() => !document.getElementById('result').hidden", timeout=6000)
    assert len(posts) == 1
    body = posts[0] or b""
    assert b'filename="audio.webm"' in body      # negotiated WebM/Opus → matching extension
    assert b'name="session_id"' in body
    assert "corporation" in page.inner_text("#answer")


def test_15s_auto_stop_uploads(harness):
    page = harness.new()
    _count_ask(page)
    page.hover("#mic")
    page.mouse.down()                           # hold and never release
    try:
        page.wait_for_function(                 # auto-stop at 15 s → onstop → upload → render
            "() => document.getElementById('decision').textContent.includes('Answered')",
            timeout=18000)
        assert "rec" not in (page.get_attribute("#mic", "class") or "")
    finally:
        page.mouse.up()


# --- mic pre-warming + honest readiness cue (first-word-cutoff fix) ------------

def test_warm_stream_reused_across_consecutive_holds(harness):
    # Pre-warming: the first hold acquires the mic; the second REUSES the still-live stream and
    # must NOT call getUserMedia again — so rec.start() fires on an already-live stream (near-zero
    # capture-start latency). Both holds still record and upload.
    page = harness.new(init_scripts=[GUM_COUNT_LIVE, _fake_recorder("audio/webm;codecs=opus")])
    posts = _count_ask(page)
    _press(page, 420)
    _wait_posts(page, posts, 1)
    assert page.evaluate("() => window.__gumCalls") == 1              # first hold acquired
    assert page.evaluate("() => window.__tracks[0].readyState") == "live"   # kept live for reuse
    _press(page, 420)
    _wait_posts(page, posts, 2)
    assert page.evaluate("() => window.__gumCalls") == 1             # second hold REUSED — no re-acquire
    assert page.evaluate("() => window.__tracks.length") == 1        # only ever one stream acquired
    assert page.errors == []


def test_idle_release_then_rewarm(harness):
    # After a hold the warm stream persists (not stopped in onstop); when the idle timeout fires it
    # is stopped for privacy, and the NEXT hold re-acquires a fresh stream.
    page = harness.new(init_scripts=[GUM_COUNT_LIVE, _fake_recorder("audio/webm;codecs=opus")])
    posts = _count_ask(page)
    _press(page, 420)
    _wait_posts(page, posts, 1)
    assert page.evaluate("() => window.__tracks[0].readyState") == "live"   # survives the hold
    # fire what the WARM_IDLE_MS timer does → warm stream stopped + dropped
    page.evaluate("() => releaseWarm()")
    assert page.evaluate("() => window.__tracks[0].readyState") == "ended"  # released for privacy
    # the next hold must RE-WARM (fresh getUserMedia) and still upload
    _press(page, 420)
    _wait_posts(page, posts, 2)
    assert page.evaluate("() => window.__gumCalls") == 2             # re-acquired after release
    assert page.evaluate("() => window.__tracks.length") == 2
    assert page.evaluate("() => window.__tracks[1].readyState") == "live"
    assert page.errors == []


def test_arming_then_listening_cue_gated_on_start_event(harness):
    # Honest readiness cue: during acquisition the hint is the interim "arming…" (NEVER a go-cue);
    # the "listening…" go-cue appears ONLY after the MediaRecorder `start` event fires, by which
    # point the recorder is already capturing — so the first word is never lost.
    page = harness.new(init_scripts=[GUM_DELAYED_LIVE, _fake_recorder("audio/webm;codecs=opus")])
    posts = _count_ask(page)
    page.hover("#mic")
    page.mouse.down()
    # while getUserMedia is pending: the interim cue, not hot, and not the go-cue
    page.wait_for_function("() => document.getElementById('hint').textContent === 'arming mic…'",
                           timeout=2000)
    assert "hot" not in (page.get_attribute("#hint", "class") or "")   # arming is not a go-cue
    # once the start event fires: the go-cue appears, hot, and the recorder was recording when it fired
    page.wait_for_function(
        "() => document.getElementById('hint').textContent === 'listening… release to ask'",
        timeout=3000)
    assert "hot" in (page.get_attribute("#hint", "class") or "")
    assert page.evaluate("() => window.__stateAtStart") == "recording"   # capture live BEFORE the go-cue
    page.wait_for_timeout(400)                                           # hold past the 300 ms floor
    page.mouse.up()
    _wait_posts(page, posts, 1)                                          # the held clip still uploads
    assert page.errors == []
