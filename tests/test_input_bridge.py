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


from manifoldx.input import GamepadButtonEvent, GamepadInfo, GamepadSnapshot


class _FakePoller:
    """Returns the queued snapshot tuples, one per poll; then repeats the last."""

    def __init__(self, *frames) -> None:
        self.frames = list(frames)
        self.polls = 0

    def poll(self):
        self.polls += 1
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]


_INFO = GamepadInfo(index=0, name="pad", guid="guid", mapped=True)


def _snap(*buttons, **axes) -> GamepadSnapshot:
    return GamepadSnapshot(info=_INFO, buttons=frozenset(buttons), axes=axes)


def test_begin_frame_polls_gamepads_and_emits_events():
    engine, state = _StubEngine(), InputState()
    poller = _FakePoller((_snap("a"),), (_snap(),), ())
    bridge = _InputBridge(engine, state, gamepad_poller=poller)

    bridge.begin_frame()
    assert state.gamepad_just_pressed("a")
    assert engine.emitted == [
        ("gamepad_connected", _INFO),
        ("gamepad_down", GamepadButtonEvent(pad=0, button="a", is_down=True)),
    ]

    engine.emitted.clear()
    bridge.begin_frame()
    assert state.gamepad_just_released("a")
    assert engine.emitted == [
        ("gamepad_up", GamepadButtonEvent(pad=0, button="a", is_down=False)),
    ]

    engine.emitted.clear()
    bridge.begin_frame()
    assert engine.emitted == [("gamepad_disconnected", _INFO)]
    assert poller.polls == 3


def test_virtual_direction_from_stick_emits_on_bus():
    engine, state = _StubEngine(), InputState()
    bridge = _InputBridge(engine, state, gamepad_poller=_FakePoller((_snap(left_x=-0.9),)))
    bridge.begin_frame()
    assert ("gamepad_down", GamepadButtonEvent(pad=0, button="left", is_down=True)) in engine.emitted


def test_no_poller_means_zero_pads_and_no_errors(setup):
    engine, state, bridge = setup
    bridge.begin_frame()
    bridge.begin_frame()
    assert state.gamepads == ()
    assert not state.gamepad_pressed("a")
    assert engine.emitted == []


class _StubCanvas:
    def add_event_handler(self, *args) -> None:
        pass


def test_attach_to_non_glfw_canvas_creates_no_poller(setup):
    engine, state, bridge = setup
    bridge.attach(_StubCanvas())
    assert bridge._gamepad_poller is None


def test_attach_to_glfw_canvas_creates_glfw_poller(setup, monkeypatch):
    import manifoldx.input as input_mod

    class GlfwRenderCanvas(_StubCanvas):  # matched by class name
        pass

    class MyCanvas(GlfwRenderCanvas):
        pass

    created = []
    monkeypatch.setattr(input_mod, "_GlfwGamepadPoller", lambda: created.append(1) or "poller")
    engine, state, bridge = setup
    bridge.attach(MyCanvas())
    assert bridge._gamepad_poller == "poller"
    assert created == [1]
