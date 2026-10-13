"""End-to-end input tests: inject events via canvas.submit_event, drive
frames through engine._draw_frame, observe both bus dispatch and
engine.input state."""
import pytest


def _make_offscreen_engine(width: int = 64, height: int = 64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_synthetic_key_down_drives_polling_state():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "key_down",
        "key": "w",
        "modifiers": (),
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert engine.input.is_pressed("w")
    assert engine.input.just_pressed("w")
    engine._draw_frame()
    assert engine.input.is_pressed("w")
    assert not engine.input.just_pressed("w")


def test_synthetic_key_up_clears_polling_state():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "key_down", "key": "w", "modifiers": (),
    })
    engine._render_canvas.submit_event({
        "event_type": "key_up", "key": "w", "modifiers": (),
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert not engine.input.is_pressed("w")
    assert engine.input.just_released("w")


def test_synthetic_pointer_event_dispatches_on_bus():
    engine = _make_offscreen_engine()
    from manifoldx.input import PointerEvent
    received: list[PointerEvent] = []

    @engine.on("pointer_down")
    def on_down(ev):
        received.append(ev)

    engine._render_canvas.submit_event({
        "event_type": "pointer_down",
        "x": 50.0, "y": 75.0,
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert len(received) == 1
    assert received[0].x == 50.0
    assert received[0].phase == "down"
    assert engine.input.is_mouse_pressed(1)


def test_synthetic_pointer_move_accumulates_delta():
    engine = _make_offscreen_engine()
    # The offscreen EventEmitter merges consecutive pointer_move events that
    # share the same (ntouches, buttons, modifiers) key — later events replace
    # earlier ones with empty accum_keys, so only the last position survives
    # a batch.  Flush between each submit so each event is dispatched alone.
    for x in (10.0, 15.0, 25.0):
        engine._render_canvas.submit_event({
            "event_type": "pointer_move",
            "x": x, "y": 0.0,
            "button": 0, "buttons": (), "modifiers": (),
            "ntouches": 0, "touches": {},
        })
        engine._render_canvas._process_events()
    engine._draw_frame()
    # First move yields dx=0 (no prior anchor); subsequent moves yield 5 + 10.
    assert engine.input.mouse_delta == (15.0, 0.0)
    # Second frame with no events resets the delta.
    engine._draw_frame()
    assert engine.input.mouse_delta == (0.0, 0.0)


def test_synthetic_wheel_event_accumulates_delta():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "wheel",
        "dx": 0.0, "dy": 100.0, "x": 0.0, "y": 0.0,
        "buttons": (), "modifiers": (),
    })
    engine._render_canvas.submit_event({
        "event_type": "wheel",
        "dx": 0.0, "dy": -50.0, "x": 0.0, "y": 0.0,
        "buttons": (), "modifiers": (),
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert engine.input.wheel_delta == (0.0, 50.0)


def test_synthetic_resize_updates_viewport_size():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "resize",
        "width": 1024, "height": 768, "pixel_ratio": 1.0,
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert engine.input.viewport_size == (1024, 768)


def test_offscreen_engine_has_no_gamepads():
    engine = _make_offscreen_engine()
    engine._draw_frame()
    assert engine._input_bridge._gamepad_poller is None
    assert engine.input.gamepads == ()
    assert not engine.input.gamepad_pressed("a")


def test_injected_gamepad_poller_drives_state_and_bus_same_frame():
    from manifoldx.input import GamepadButtonEvent, GamepadInfo, GamepadSnapshot

    engine = _make_offscreen_engine()
    info = GamepadInfo(index=0, name="pad", guid="guid", mapped=True)
    frames = [
        (GamepadSnapshot(info, frozenset({"a"}), {"left_y": -1.0}),),
        (GamepadSnapshot(info, frozenset(), {}),),
    ]

    class _Poller:
        def poll(self):
            return frames.pop(0)

    engine._input_bridge._gamepad_poller = _Poller()
    received: list = []
    seen_in_handler: list[bool] = []

    @engine.on("gamepad_connected")
    def on_connected(ev):
        received.append(ev)

    @engine.on("gamepad_down")
    def on_down(ev):
        received.append(ev)
        seen_in_handler.append(engine.input.gamepad_just_pressed(ev.button))

    engine._draw_frame()
    assert received == [
        info,
        GamepadButtonEvent(pad=0, button="a", is_down=True),
        GamepadButtonEvent(pad=0, button="up", is_down=True),
    ]
    # Delivered on the frame of the poll, so polling state agrees.
    assert seen_in_handler == [True, True]
    assert engine.input.gamepad_pressed("up")
    engine._draw_frame()
    assert engine.input.gamepad_just_released("a")
    assert not engine.input.gamepad_pressed("up")
