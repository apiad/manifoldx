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
