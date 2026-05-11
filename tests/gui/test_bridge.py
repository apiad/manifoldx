"""Tests for manifoldx.gui.bridge — pointer routing skeleton."""

import pytest

from manifoldx.gui import Panel, style
from manifoldx.gui.bridge import _GuiBridge


def setup_function(_):
    style.reset()


def test_bridge_constructs_with_engine_and_subscribes_to_pointer_events():
    received = []

    class _Stub:
        def __init__(self):
            self.gui = _GuiRootStub()
            self._handlers = {}
        def on(self, event_name):
            def decorator(fn):
                self._handlers.setdefault(event_name, []).append(fn)
                return fn
            return decorator
        def emit(self, name, payload):
            received.append((name, payload))

    class _GuiRootStub:
        def __init__(self):
            self.pointer_over_gui = False
            self._panels = []
        def __iter__(self):
            return iter(self._panels)
        def __len__(self):
            return len(self._panels)

    eng = _Stub()
    bridge = _GuiBridge(eng)
    assert "pointer_down" in eng._handlers
    assert "pointer_move" in eng._handlers
    assert "pointer_up" in eng._handlers


def test_bridge_begin_frame_clears_pointer_over_gui_flag():
    class _G:
        pointer_over_gui = True
        def __iter__(self):
            return iter([])
        def __len__(self):
            return 0

    class _Stub:
        def __init__(self):
            self.gui = _G()
            self._handlers = {}
        def on(self, event_name):
            def deco(fn):
                self._handlers.setdefault(event_name, []).append(fn)
                return fn
            return deco
        def emit(self, *a, **k):
            pass
    eng = _Stub()
    bridge = _GuiBridge(eng)
    eng.gui.pointer_over_gui = True
    bridge.begin_frame()
    assert eng.gui.pointer_over_gui is False


def test_pointer_event_over_interactive_widget_sets_flag():
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    from manifoldx.gui.widgets import Widget

    class _Marker(Widget):
        _is_gui_interactive = True
        def intrinsic_size(self):
            return (10.0, 10.0)

    engine = mx.Engine("test", width=128, height=128)
    engine._init_canvas(canvas)
    engine._running = True

    panel = Panel(
        children=[_Marker()],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 50, "height": 50, "padding": 0},
    )
    engine.gui.append(panel)

    engine._render_canvas.submit_event({
        "event_type": "pointer_down",
        "x": 10.0, "y": 10.0,
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert engine.gui.pointer_over_gui is True


def test_pointer_move_without_capture_does_not_dispatch_to_widget():
    """A raw pointer_move (no prior down) must not call widget event hooks."""
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    from manifoldx.gui.widgets import Widget

    dispatched = []

    class _Marker(Widget):
        _is_gui_interactive = True
        _gui_captures_pointer = True

        def intrinsic_size(self):
            return (10.0, 10.0)

        def _on_pointer_move(self, ev, engine):
            dispatched.append("move")

    engine = mx.Engine("test", width=128, height=128)
    engine._init_canvas(canvas)
    engine._running = True

    panel = Panel(
        children=[_Marker()],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 50, "height": 50, "padding": 0},
    )
    engine.gui.append(panel)

    # Submit move event without any prior down — should not dispatch.
    engine._render_canvas.submit_event({
        "event_type": "pointer_move",
        "x": 10.0, "y": 10.0,
        "button": 0, "buttons": (0,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert dispatched == []
