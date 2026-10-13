"""Engine.w / Engine.h follow canvas resize events (logical pixels)."""

import manifoldx as mx


def _resize(engine, w, h, ratio=1.0):
    engine._input_bridge._on_event(
        {"event_type": "resize", "width": w, "height": h, "pixel_ratio": ratio}
    )


def test_resize_event_updates_engine_size():
    engine = mx.Engine("t", width=800, height=600)
    _resize(engine, 1920.0, 1080.0, ratio=2.0)
    assert (engine.w, engine.h) == (1920, 1080)
    assert isinstance(engine.w, int) and isinstance(engine.h, int)


def test_resize_moves_gui_layout_viewport():
    from manifoldx.gui import Panel
    from manifoldx.render.passes.gui import layout_gui

    engine = mx.Engine("t", width=800, height=600)
    engine.gui.append(Panel(children=[], style_overrides={"padding": 0}))
    _resize(engine, 1280.0, 720.0)
    painter = layout_gui(engine)
    box = painter.rect_ops[0].box
    assert (box.w, box.h) == (1280.0, 720.0)


def test_offscreen_still_ignores_supersampled_canvas_size():
    """render_frame draws on a (w*ss, h*ss) canvas; its resize event must not
    change the logical frame size."""
    engine = mx.Engine("t", width=64, height=48)
    engine._follow_resize = False  # what _ensure_offscreen sets
    _resize(engine, 128.0, 96.0)
    assert (engine.w, engine.h) == (64, 48)
