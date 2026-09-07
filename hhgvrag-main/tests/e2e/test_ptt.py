"""
Spacebar hold-to-talk (page-global push-to-talk). SPACE anywhere except a text field / IME
is the SAME push-to-talk trigger as the mic disc: keydown arms a hold, keyup stops it and
uploads, OS auto-repeat is ignored, and a space typed into the query input records nothing
(it types a space instead). This suite drives only the SPACE entry point — the recorder
state machine it reuses (warm stream, arming->listening cue, generation counter, <300ms
discard, silence guard, 15s auto-stop) is covered in test_recorder.py.
"""
from test_recorder import (GUM_IMMEDIATE, GUM_COUNT_LIVE, _fake_recorder,
                           _count_ask, _wait_posts)

REC = _fake_recorder("audio/webm;codecs=opus")

# dispatch a synthetic Space keydown on document with caller-supplied flags (repeat / IME)
DISPATCH_SPACE = """
(opts) => document.dispatchEvent(new KeyboardEvent('keydown',
    Object.assign({code: 'Space', key: ' ', bubbles: true, cancelable: true}, opts || {})))
"""


def test_space_keydown_starts_and_keyup_uploads(harness):
    # keydown (SPACE, nothing focused) arms a hold; keyup stops it and uploads the held clip
    page = harness.new(init_scripts=[GUM_IMMEDIATE, REC])
    posts = _count_ask(page)
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.keyboard.down("Space")
    page.wait_for_function(
        "() => document.getElementById('mic').classList.contains('rec')", timeout=3000)
    page.wait_for_timeout(420)                       # hold past the 300 ms floor
    page.keyboard.up("Space")
    _wait_posts(page, posts, 1)                      # released -> the held clip uploaded via /ask
    assert "rec" not in (page.get_attribute("#mic", "class") or "")
    assert page.errors == []


def test_space_start_then_stop_state(harness):
    # explicit keydown-starts / keyup-stops via real key events (observed on the mic .rec state)
    page = harness.new(init_scripts=[GUM_COUNT_LIVE, REC])
    _count_ask(page)
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.keyboard.down("Space")
    page.wait_for_function(                                             # keydown armed the hold
        "() => document.getElementById('mic').classList.contains('rec')", timeout=3000)
    assert page.evaluate("() => window.__gumCalls") == 1               # one acquisition, same machine
    page.keyboard.up("Space")
    page.wait_for_function(                                             # keyup released it
        "() => !document.getElementById('mic').classList.contains('rec')", timeout=3000)
    assert page.errors == []


def test_space_autorepeat_keydown_is_ignored(harness):
    # OS key-repeat (keydown with e.repeat=true) must NEVER arm a hold
    page = harness.new(init_scripts=[GUM_COUNT_LIVE, REC])
    posts = _count_ask(page)
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.evaluate(DISPATCH_SPACE, {"repeat": True})
    page.wait_for_timeout(150)
    assert "rec" not in (page.get_attribute("#mic", "class") or "")
    assert page.evaluate("() => window.__gumCalls") == 0               # nothing acquired the mic
    assert posts == []
    assert page.errors == []


def test_space_while_typing_in_input_records_nothing(harness):
    # SPACE inside the query field types a space; it must not hijack into a recording
    page = harness.new(init_scripts=[GUM_COUNT_LIVE, REC])
    posts = _count_ask(page)
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.focus("#q")
    page.keyboard.type("what is")                    # the embedded space stays in the field
    page.wait_for_timeout(120)
    assert "rec" not in (page.get_attribute("#mic", "class") or "")
    assert page.evaluate("() => window.__gumCalls") == 0
    assert page.input_value("#q") == "what is"       # the space was typed, not swallowed
    assert posts == []
    assert page.errors == []


def test_space_during_ime_composition_is_ignored(harness):
    # a composing keydown (IME candidate window open) must not trigger PTT
    page = harness.new(init_scripts=[GUM_COUNT_LIVE, REC])
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.evaluate(DISPATCH_SPACE, {"isComposing": True})
    page.wait_for_timeout(120)
    assert "rec" not in (page.get_attribute("#mic", "class") or "")
    assert page.evaluate("() => window.__gumCalls") == 0
    assert page.errors == []


def test_space_on_focused_mic_not_double_started(harness):
    # when the mic button itself is focused, its own handler owns SPACE; the global handler
    # must defer (no double start / no error), and the hold still records + uploads on release
    page = harness.new(init_scripts=[GUM_COUNT_LIVE, REC])
    posts = _count_ask(page)
    page.wait_for_selector("#dot.ok", timeout=5000)
    page.focus("#mic")
    page.keyboard.down("Space")
    page.wait_for_function(
        "() => document.getElementById('mic').classList.contains('rec')", timeout=3000)
    assert page.evaluate("() => window.__gumCalls") == 1               # exactly one acquisition
    page.wait_for_timeout(420)
    page.keyboard.up("Space")
    _wait_posts(page, posts, 1)
    assert page.evaluate("() => window.__gumCalls") == 1               # still one — no double start
    assert page.errors == []
