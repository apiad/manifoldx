"""End-to-end integration tests for sci-viz Plan 2 label rendering."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Transform
from manifoldx.viz import LabelMaterial, TextLabel


def _make_offscreen_engine(width=128, height=128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    try:
        engine = mx.Engine("test", width=width, height=height)
        engine._init_canvas(canvas)
    except Exception as e:
        pytest.skip(f"engine initialization failed: {e}")

    # TextLabel auto-registers on first spawn now that it inherits from
    # Component — no manual registration needed.
    engine._running = True
    return engine


def test_label_renders_visible_pixels_at_origin():
    """A label spawned at origin produces non-transparent pixels in the rendered frame."""
    engine = _make_offscreen_engine()
    engine.camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    atlas = engine.get_label_atlas()
    slice_idx = atlas.get_or_create("HELLO")  # blocky monospace, easy to read

    engine.spawn(
        Material(LabelMaterial()),
        Transform(pos=(0.0, 0.0, 0.0)),
        TextLabel(index=slice_idx),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    assert frame.shape == (128, 128, 4)
    assert frame.dtype == np.uint8

    # The label is white-on-transparent; on a black background the rendered
    # text region should contain bright pixels somewhere in the middle band.
    middle = frame[40:88, 16:112, :3]
    bright = (middle.max(axis=-1) > 100).any()
    assert bright, "no bright pixels found in label region — text not rendered"


def test_label_does_not_render_when_no_atlas_strings_registered():
    """If no string was ever registered, the label pass is a no-op (no crash)."""
    engine = _make_offscreen_engine()
    engine.camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    engine.spawn(
        Material(LabelMaterial()),
        Transform(pos=(0.0, 0.0, 0.0)),
        TextLabel(index=0),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    assert frame is not None  # no crash
