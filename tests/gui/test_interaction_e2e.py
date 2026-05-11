"""End-to-end GUI interaction tests via synthetic pointer events."""

import pytest

from manifoldx.gui import Button, Panel, Slider, Toggle, style


def _make_offscreen_engine(width: int = 256, height: int = 128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine, canvas


def setup_function(_):
    style.reset()


def _click(engine, x, y):
    engine._render_canvas.submit_event({
        "event_type": "pointer_down",
        "x": float(x), "y": float(y),
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_up",
        "x": float(x), "y": float(y),
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    # First frame: pointer events are dispatched → widget handlers run → gui
    # events are emitted into _pending.
    engine._draw_frame()
    # Second frame: gui events (e.g. gui:button:...:click) are dispatched →
    # user @engine.on(...) handlers fire.
    engine._draw_frame()


def test_button_click_fires_handler():
    style.set_theme({"padding": 0, "gap": 0})
    engine, _ = _make_offscreen_engine()
    received = []

    @engine.on("gui:button:reset:click")
    def _(payload):
        received.append(payload)

    panel = Panel(
        children=[Button(name="reset", label="Reset")],
        anchor="top-left", offset=(0, 0),
        style_overrides={"width": 100, "height": 30, "padding": 0},
    )
    engine.gui.append(panel)
    _click(engine, 50, 15)
    assert received == [{}]


def test_toggle_click_flips_and_emits():
    style.set_theme({"padding": 0, "gap": 0})
    engine, _ = _make_offscreen_engine()
    received = []

    @engine.on("gui:toggle:trails:change")
    def _(payload):
        received.append(payload)

    tg = Toggle(name="trails", value=False, label="Trails")
    panel = Panel(
        children=[tg],
        anchor="top-left", offset=(0, 0),
        style_overrides={"width": 100, "height": 30, "padding": 0},
    )
    engine.gui.append(panel)
    _click(engine, 10, 15)
    assert received == [{"value": True}]
    assert tg.value is True


def test_slider_drag_emits_change_and_commit():
    style.set_theme({"padding": 0, "gap": 0})
    engine, _ = _make_offscreen_engine(width=200, height=64)
    changes, commits = [], []

    @engine.on("gui:slider:G:change")
    def _on_change(payload):
        changes.append(payload["value"])

    @engine.on("gui:slider:G:commit")
    def _on_commit(payload):
        commits.append(payload["value"])

    slider = Slider(name="G", min=0.0, max=100.0, value=0.0, label="G")
    panel = Panel(
        children=[slider],
        anchor="top-left", offset=(0, 0),
        style_overrides={"width": 200, "height": 30, "padding": 0},
    )
    engine.gui.append(panel)

    engine._render_canvas.submit_event({
        "event_type": "pointer_down", "x": 20.0, "y": 15.0,
        "button": 1, "buttons": (1,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_move", "x": 100.0, "y": 15.0,
        "button": 0, "buttons": (1,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_up", "x": 100.0, "y": 15.0,
        "button": 1, "buttons": (0,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    # First frame: pointer events dispatched → widget emits change/commit into _pending.
    engine._draw_frame()
    # Second frame: change/commit events dispatched → user handlers fire.
    engine._draw_frame()

    assert len(changes) >= 2
    assert commits == [50.0]
    assert slider.value == 50.0


def test_slider_hover_move_does_not_emit_change():
    """pointer_move over a Slider WITHOUT prior pointer_down must not emit change."""
    style.set_theme({"padding": 0, "gap": 0})
    engine, _ = _make_offscreen_engine(width=200, height=64)
    changes = []

    @engine.on("gui:slider:G:change")
    def _on_change(payload):
        changes.append(payload["value"])

    slider = Slider(name="G", min=0.0, max=100.0, value=0.0, label="G")
    panel = Panel(
        children=[slider],
        anchor="top-left", offset=(0, 0),
        style_overrides={"width": 200, "height": 30, "padding": 0},
    )
    engine.gui.append(panel)

    # Move over the slider without any pointer_down first.
    engine._render_canvas.submit_event({
        "event_type": "pointer_move", "x": 50.0, "y": 15.0,
        "button": 0, "buttons": (0,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_move", "x": 100.0, "y": 15.0,
        "button": 0, "buttons": (0,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_move", "x": 150.0, "y": 15.0,
        "button": 0, "buttons": (0,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    engine._draw_frame()

    # No change events must have fired — only hover, no drag.
    assert changes == []
    assert slider.value == 0.0
