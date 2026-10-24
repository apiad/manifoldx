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


def test_viewport_size_is_known_before_any_resize_event():
    engine = _make_offscreen_engine(width=80, height=48)
    assert engine.input.viewport_size == (80, 48)


def test_gui_ignores_the_pointer_while_it_is_captured():
    from manifoldx.gui import Panel
    from manifoldx.gui.widgets import Widget

    class _Marker(Widget):
        _is_gui_interactive = True

        def intrinsic_size(self):
            return (10.0, 10.0)

    engine = _make_offscreen_engine()
    engine.gui.append(Panel(children=[_Marker()], style_overrides={"padding": 0}))
    engine.input._pointer_captured = True
    engine._render_canvas.submit_event({
        "event_type": "pointer_down", "x": 2.0, "y": 2.0, "button": 1,
        "buttons": (1,), "modifiers": (),
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert engine.gui.pointer_over_gui is False
