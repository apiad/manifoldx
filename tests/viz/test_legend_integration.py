"""Integration tests for the colormap_legend rendering capability.

Plan 3 part 2 doesn't add a new material or pipeline for legends — it
hijacks the existing label atlas: a colormap LUT is rasterized into one
of the 256×64 atlas slices and rendered via the standard screen-anchored
LabelMaterial path. Plan 4's `colormap_legend(...)` shim will compose
this with `TextLabel` tick annotations.
"""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Transform
from manifoldx.viz import LabelMaterial, TextLabel
from manifoldx.viz.colormaps import get_colormap


def _make_offscreen_engine(width=128, height=128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_register_colormap_legend_returns_atlas_slot():
    """The atlas exposes a method to register a colormap LUT as a slice."""
    engine = _make_offscreen_engine()
    atlas = engine.get_label_atlas()
    slot = atlas.register_colormap_legend("viridis")
    assert isinstance(slot, int)
    assert slot >= 0


def test_register_colormap_legend_idempotent():
    """Calling register_colormap_legend twice for the same cmap returns the
    same slot (no atlas-cap pressure for duplicate registrations)."""
    engine = _make_offscreen_engine()
    atlas = engine.get_label_atlas()
    a = atlas.register_colormap_legend("viridis")
    b = atlas.register_colormap_legend("viridis")
    assert a == b


def test_colormap_legend_renders_with_expected_gradient():
    """Render a viridis legend at NDC (+0.7, 0). The leftmost column of
    rendered legend pixels should match viridis[0] (dark blue), the
    rightmost should match viridis[255] (yellow). This validates that the
    LUT bytes survived the atlas → GPU → fragment-shader sample roundtrip."""
    engine = _make_offscreen_engine(width=128, height=128)
    # Camera looks at unrelated location — legend rendering is screen-space.
    engine.camera.position = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    engine.camera.target = np.array([200.0, 200.0, 200.0], dtype=np.float32)

    atlas = engine.get_label_atlas()
    slot = atlas.register_colormap_legend("viridis")

    # Big legend so the leftmost / rightmost column samples are unambiguous.
    engine.spawn(
        Material(LabelMaterial(pixel_width=80, pixel_height=20, anchor_mode="screen")),
        Transform(pos=(0.0, 0.0, 0.0)),
        TextLabel(index=slot),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    assert frame.shape == (128, 128, 4)

    # The legend center is at NDC (0, 0) = pixel (64, 64). With pixel_width=80
    # and pixel_height=20 the legend covers x ∈ [24, 104], y ∈ [54, 74].
    legend_strip = frame[60:68, 28:100, :3]  # well-inside band

    # Viridis: LUT[0] is deep blue-purple, LUT[255] is bright yellow.
    # Sampled in the rendered legend, the left edge should be more blue than
    # red; the right edge should be more red+green (yellow) than blue.
    left_col = legend_strip[:, :8].mean(axis=(0, 1))   # avg over leftmost columns
    right_col = legend_strip[:, -8:].mean(axis=(0, 1))  # rightmost columns

    # Left edge: viridis[0] ≈ (68, 1, 84) sRGB → strongly blue/purple.
    assert left_col[2] > left_col[0], (
        f"leftmost legend column not blue-dominant: rgb={tuple(left_col.astype(int))}"
    )

    # Right edge: viridis[255] ≈ (253, 231, 37) → yellow (R≫B, G≫B).
    assert right_col[0] > right_col[2] + 50, (
        f"rightmost legend column not yellow: rgb={tuple(right_col.astype(int))}"
    )
    assert right_col[1] > right_col[2] + 50, (
        f"rightmost legend column not yellow: rgb={tuple(right_col.astype(int))}"
    )
