"""Bridge translation tests: feed rendercanvas dicts, observe typed bus
emissions and InputState mutations. Uses a stub engine to avoid GPU."""
import pytest

from manifoldx.input import (
    InputState,
    KeyEvent,
    PointerEvent,
    ResizeEvent,
    WheelEvent,
    _InputBridge,
)


class _StubEngine:
    """Mimics the surface of `Engine` that `_InputBridge` touches."""

    def __init__(self) -> None:
        self.emitted: list[tuple[str, object]] = []

    def emit(self, event: str, payload) -> None:
        self.emitted.append((event, payload))


@pytest.fixture
def setup():
    state = InputState()
    engine = _StubEngine()
    bridge = _InputBridge(engine, state)
    return engine, state, bridge


def test_key_down_translates_and_records(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "key_down",
        "key": "w",
        "modifiers": (),
    })
    assert len(engine.emitted) == 1
    name, ev = engine.emitted[0]
    assert name == "key_down"
    assert isinstance(ev, KeyEvent)
    assert ev.key == "w"
    assert ev.is_down is True
    assert state.is_pressed("w")


def test_key_up_translates_and_records(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "key_down", "key": "w", "modifiers": ()})
    bridge._on_event({"event_type": "key_up", "key": "w", "modifiers": ()})
    name, ev = engine.emitted[-1]
    assert name == "key_up"
    assert isinstance(ev, KeyEvent)
    assert ev.is_down is False
    assert not state.is_pressed("w")


def test_pointer_down_carries_zero_delta(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "pointer_down",
        "x": 100.0, "y": 200.0,
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    name, ev = engine.emitted[0]
    assert name == "pointer_down"
    assert isinstance(ev, PointerEvent)
    assert ev.dx == 0.0 and ev.dy == 0.0
    assert ev.phase == "down"
    assert state.is_mouse_pressed(1)


def test_pointer_move_computes_dx_dy_from_last_pointer_pos(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "pointer_move",
        "x": 100.0, "y": 100.0,
        "button": 0, "buttons": (), "modifiers": (),
        "ntouches": 0, "touches": {},
    })
    # First move: no prior position → dx/dy are 0.
    _, ev1 = engine.emitted[0]
    assert ev1.dx == 0.0 and ev1.dy == 0.0
    bridge._on_event({
        "event_type": "pointer_move",
        "x": 110.0, "y": 105.0,
        "button": 0, "buttons": (), "modifiers": (),
        "ntouches": 0, "touches": {},
    })
    _, ev2 = engine.emitted[1]
    assert ev2.dx == 10.0 and ev2.dy == 5.0


def test_pointer_down_resets_last_pointer_pos(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "pointer_move", "x": 100.0, "y": 100.0,
                      "button": 0, "buttons": (), "modifiers": (),
                      "ntouches": 0, "touches": {}})
    bridge._on_event({"event_type": "pointer_down", "x": 200.0, "y": 200.0,
                      "button": 1, "buttons": (1,), "modifiers": (),
                      "ntouches": 0, "touches": {}})
    bridge._on_event({"event_type": "pointer_move", "x": 205.0, "y": 200.0,
                      "button": 0, "buttons": (1,), "modifiers": (),
                      "ntouches": 0, "touches": {}})
    # After the down event, the bridge anchors at (200, 200), so the next
    # move's delta is (5, 0), NOT (105, 100).
    _, ev = engine.emitted[-1]
    assert ev.dx == 5.0 and ev.dy == 0.0


def test_wheel_event_translates(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "wheel",
        "dx": 0.0, "dy": 100.0, "x": 50.0, "y": 50.0,
        "buttons": (), "modifiers": (),
    })
    name, ev = engine.emitted[0]
    assert name == "wheel"
    assert isinstance(ev, WheelEvent)
    assert ev.dy == 100.0


def test_resize_event_translates_and_updates_viewport(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "resize",
        "width": 1024, "height": 768, "pixel_ratio": 2.0,
    })
    name, ev = engine.emitted[0]
    assert name == "resize"
    assert isinstance(ev, ResizeEvent)
    assert ev.width == 1024
    assert state.viewport_size == (1024, 768)


def test_unknown_event_type_is_ignored(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "pointer_enter"})
    bridge._on_event({"event_type": "double_click", "x": 0, "y": 0})
    bridge._on_event({"event_type": "char", "char_str": "a"})
    assert engine.emitted == []


def test_begin_frame_triggers_state_swap(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "key_down", "key": "w", "modifiers": ()})
    assert not state.just_pressed("w")
    bridge.begin_frame()
    assert state.just_pressed("w")


def _key(bridge, et, key, modifiers=()):
    bridge._on_event({"event_type": et, "key": key, "modifiers": modifiers})


def test_letter_pressed_under_shift_is_recorded_lowercase(setup):
    engine, state, bridge = setup
    _key(bridge, "key_down", "Shift", ("Shift",))
    _key(bridge, "key_down", "W", ("Shift",))
    assert state.is_pressed("w")
    assert engine.emitted[-1][1].key == "w"
    _key(bridge, "key_up", "w", ())
    assert not state.is_pressed("w")


def test_space_is_named_space_and_accepts_the_blank_alias(setup):
    engine, state, bridge = setup
    _key(bridge, "key_down", " ")
    bridge.begin_frame()
    assert engine.emitted[-1][1].key == "Space"
    assert state.is_pressed("Space") and state.is_pressed(" ")
    assert state.just_pressed("Space") and state.just_pressed(" ")
    _key(bridge, "key_up", " ")
    bridge.begin_frame()
    assert not state.is_pressed("Space")
    assert state.just_released(" ")


class _StubWindowCanvas:
    """A canvas with the private glfw window rendercanvas keeps in `_window`."""

    def __init__(self) -> None:
        self._window = object()
        self.handlers = []

    def add_event_handler(self, fn, *types):
        self.handlers.append(fn)

    def remove_event_handler(self, fn, *types):
        self.handlers.remove(fn)


def test_capture_pointer_without_a_window_is_a_noop(setup):
    _, state, _ = setup
    assert state.capture_pointer(True) is False
    assert state.pointer_captured is False


def test_capture_pointer_sets_glfw_modes_and_resets_the_anchor(setup, monkeypatch):
    engine, state, bridge = setup
    calls = []
    import manifoldx.input as inp

    monkeypatch.setattr(inp, "_glfw_set_capture", lambda win, on: calls.append((win, on)))
    canvas = _StubWindowCanvas()
    bridge.attach(canvas)
    bridge._on_event({"event_type": "pointer_move", "x": 10.0, "y": 10.0,
                      "buttons": (), "modifiers": ()})
    assert state.capture_pointer(True) is True
    assert state.pointer_captured
    assert calls == [(canvas._window, True)]
    # glfw jumps the virtual cursor on capture; the first move must not count it.
    bridge._on_event({"event_type": "pointer_move", "x": 900.0, "y": 700.0,
                      "buttons": (), "modifiers": ()})
    assert engine.emitted[-1][1].dx == 0.0
    assert state.capture_pointer(False) is True
    assert not state.pointer_captured
    assert calls[-1] == (canvas._window, False)


def test_detach_releases_capture(setup, monkeypatch):
    _, state, bridge = setup
    calls = []
    import manifoldx.input as inp

    monkeypatch.setattr(inp, "_glfw_set_capture", lambda win, on: calls.append(on))
    canvas = _StubWindowCanvas()
    bridge.attach(canvas)
    state.capture_pointer(True)
    bridge.detach(canvas)
    assert calls == [True, False]
    assert not state.pointer_captured
    assert state.capture_pointer(True) is False
