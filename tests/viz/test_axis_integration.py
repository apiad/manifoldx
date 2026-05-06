"""End-to-end integration test for AxisFrame + AxisMaterial line rendering."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.viz import AxisFrame, AxisMaterial


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


def test_three_axes_render_distinct_colored_lines():
    """Spawn X (red), Y (green), Z (blue) axes; verify each color shows up
    as bright pixels in the rendered frame."""
    engine = _make_offscreen_engine()
    # Camera offset so all three axes are visible (not edge-on).
    engine.camera.position = np.array([6.0, 4.0, 6.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Look up the three built-in axis line geometries by name.
    geom_x = engine._geometry_registry.get_by_name("axis_line_x")
    geom_y = engine._geometry_registry.get_by_name("axis_line_y")
    geom_z = engine._geometry_registry.get_by_name("axis_line_z")
    assert geom_x is not None and geom_y is not None and geom_z is not None

    # Three entities, one per axis. Transform.scale extends the unit line to
    # ±extent on its axis; the AxisFrame component tags it for the line path.
    extent = 3.0
    engine.spawn(
        Mesh(geom_x),
        Material(AxisMaterial(color="#ff0000")),
        Transform(pos=(0.0, 0.0, 0.0), scale=(extent, 1.0, 1.0)),
        AxisFrame(extent=extent),
        n=1,
    )
    engine.spawn(
        Mesh(geom_y),
        Material(AxisMaterial(color="#00ff00")),
        Transform(pos=(0.0, 0.0, 0.0), scale=(1.0, extent, 1.0)),
        AxisFrame(extent=extent),
        n=1,
    )
    engine.spawn(
        Mesh(geom_z),
        Material(AxisMaterial(color="#0000ff")),
        Transform(pos=(0.0, 0.0, 0.0), scale=(1.0, 1.0, extent)),
        AxisFrame(extent=extent),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    assert frame.shape == (128, 128, 4)

    rgb = frame[..., :3].astype(np.int32)
    # A pixel is "red-dominant" when r > g and r > b by a clear margin (the
    # background gray won't satisfy this). Same for green and blue.
    red_dominant = ((rgb[..., 0] - rgb[..., 1] > 30) & (rgb[..., 0] - rgb[..., 2] > 30)).sum()
    green_dominant = ((rgb[..., 1] - rgb[..., 0] > 30) & (rgb[..., 1] - rgb[..., 2] > 30)).sum()
    blue_dominant = ((rgb[..., 2] - rgb[..., 0] > 30) & (rgb[..., 2] - rgb[..., 1] > 30)).sum()

    assert red_dominant > 0, f"X axis (red) not rendered: {red_dominant} red-dominant px"
    assert green_dominant > 0, f"Y axis (green) not rendered: {green_dominant} green-dominant px"
    assert blue_dominant > 0, f"Z axis (blue) not rendered: {blue_dominant} blue-dominant px"


def test_screen_anchored_axis_renders_at_ndc_anchor():
    """AxisMaterial(anchor_mode='screen') interprets Transform.pos as the NDC
    anchor and the entity's geometry as a screen-space line — independent
    of camera. Spawns a horizontal screen-anchored bar at NDC (0, -0.7)
    (lower middle of the frame) with the camera pointed somewhere
    irrelevant; the bar should land in the lower half of the rendered
    frame, not anywhere else."""
    engine = _make_offscreen_engine(width=128, height=128)
    # Camera looks at a totally unrelated location.
    engine.camera.position = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    engine.camera.target = np.array([200.0, 200.0, 200.0], dtype=np.float32)

    geom_x = engine._geometry_registry.get_by_name("axis_line_x")
    # In screen mode the entity's Transform.scale.x sets the half-length of
    # the bar in NDC units (the unit-line vertices ±1 get multiplied by it).
    # A scale.x of 0.4 = 80% of viewport width.
    engine.spawn(
        Mesh(geom_x),
        Material(AxisMaterial(color="#ffff00", anchor_mode="screen")),
        Transform(pos=(0.0, -0.7, 0.0), scale=(0.4, 1.0, 1.0)),
        AxisFrame(extent=1.0),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    rgb = frame[..., :3].astype(np.int32)
    yellow = (rgb[..., 0] > 150) & (rgb[..., 1] > 150) & (rgb[..., 2] < 80)

    # The bar at NDC y = -0.7 sits in the lower 15% of the frame
    # (NDC -1 = bottom row, NDC +1 = top row; -0.7 → row ≈ 0.85 * 128 ≈ 109).
    lower = yellow[96:128].sum()
    upper = yellow[0:96].sum()
    assert lower > 0, f"screen-anchored bar not rendered in lower band: {lower} yellow px"
    assert lower > upper, (
        f"bar leaked outside lower band: {lower} yellow in lower vs {upper} elsewhere"
    )
