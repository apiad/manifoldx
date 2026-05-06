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


def test_screen_anchored_label_lands_in_target_ndc_quadrant():
    """anchor_mode='screen' interprets Transform.pos.xy as an NDC anchor in
    [-1, 1]. A label anchored at (+0.5, +0.5) with the camera looking in a
    completely unrelated direction should still land in the upper-right
    quadrant of the rendered frame."""
    engine = _make_offscreen_engine(width=128, height=128)
    # Camera looks AWAY from origin — confirms screen-anchored bypasses view/proj.
    engine.camera.position = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    engine.camera.target = np.array([200.0, 200.0, 200.0], dtype=np.float32)

    atlas = engine.get_label_atlas()
    slice_idx = atlas.get_or_create("HELLO")

    engine.spawn(
        # Label sized to fit inside the upper-right quadrant of a 128×128
        # frame: NDC anchor (+0.5, +0.5) = pixel (96, 32), label 64×24 px
        # spans roughly pixel x[64..128], y[20..44] — entirely inside UR.
        Material(LabelMaterial(pixel_width=64, pixel_height=24, anchor_mode="screen")),
        Transform(pos=(0.5, 0.5, 0.0)),
        TextLabel(index=slice_idx),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    assert frame.shape == (128, 128, 4)

    # Upper-right quadrant: rows 0..63 (top half — wgpu Y-up means NDC +y = top
    # = small row index), cols 64..127. Use threshold 200 so we count only the
    # near-white label glyph pixels, not the offscreen-clear gray background
    # (which sits around 124).
    upper_right = frame[0:64, 64:128, :3]
    other_quadrants = [
        frame[0:64, 0:64, :3],
        frame[64:128, 0:64, :3],
        frame[64:128, 64:128, :3],
    ]
    bright_ur = (upper_right.max(axis=-1) > 150).sum()
    bright_others = sum((q.max(axis=-1) > 150).sum() for q in other_quadrants)

    assert bright_ur > 5, f"too few bright pixels in target quadrant: {bright_ur}"
    assert bright_others == 0, (
        f"label leaked outside target quadrant: {bright_ur} bright in UR vs "
        f"{bright_others} elsewhere"
    )
