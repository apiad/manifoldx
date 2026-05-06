"""End-to-end integration tests for sci-viz Plan 1 primitives."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Transform, Material
from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue


def _make_offscreen_engine():
    """Create an Engine in headless / offscreen mode for tests.

    Skips if the wgpu backend or offscreen canvas is unavailable.
    """
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=128, height=128)
    except (ImportError, Exception) as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    try:
        engine = mx.Engine("test", width=128, height=128)
        engine._init_canvas(canvas)
    except Exception as e:
        pytest.skip(f"engine initialization failed: {e}")

    engine.store.register_component("PointCloud", np.dtype("f4"), (0,))
    engine.store.register_component("ScalarValue", np.dtype("f4"), (1,))
    engine.store.register_component("Radius", np.dtype("f4"), (1,))

    engine._running = True

    return engine


def test_spawn_point_cloud_no_crash():
    """Spawn 100 sprites, render one frame, verify no errors."""
    engine = _make_offscreen_engine()

    N = 100
    rng = np.random.default_rng(42)
    positions = rng.standard_normal((N, 3)).astype(np.float32) * 2.0
    masses = rng.exponential(1.0, N).astype(np.float32)

    engine.spawn(
        PointCloud(),
        Material(ColormapMaterial(cmap="viridis", vmin=0.0, vmax=3.0)),
        Transform(pos=positions),
        ScalarValue(value=masses),
        Radius(radius=0.1),
        n=N,
    )

    engine._draw_frame()

    frame = engine._render_canvas.draw()
    assert frame is not None
    assert frame.shape == (128, 128, 4)
    assert frame.dtype == np.uint8


def test_spawn_point_cloud_entity_count():
    """Verify entity store registered N entities with correct component shapes."""
    engine = _make_offscreen_engine()

    N = 50
    engine.spawn(
        PointCloud(),
        Material(ColormapMaterial(cmap="magma", vmin=0.0, vmax=1.0)),
        Transform(pos=np.zeros((N, 3), dtype=np.float32)),
        ScalarValue(value=0.5),
        Radius(radius=0.1),
        n=N,
    )

    store = engine.store
    assert "PointCloud" in store._components
    assert "ScalarValue" in store._components
    assert "Radius" in store._components
    assert np.sum(store._alive) == N


def test_scalar_value_update_per_frame():
    """A scalar mutated between frames produces a different framebuffer color.

    Spawns one large sprite at the origin. Renders frame 0 with scalar=0.0
    (should sample LUT[0] → dark purple for viridis). Mutates scalar to 1.0,
    renders frame 1 (should sample LUT[255] → yellow). The center pixel must
    change between frames, and each frame's center pixel must be close to the
    expected colormap output.
    """
    from manifoldx.viz import colormaps

    engine = _make_offscreen_engine()
    # Camera looking at the origin from +Z so the sprite at origin is centered.
    engine.camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    engine.spawn(
        PointCloud(),
        Material(ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)),
        Transform(pos=(0.0, 0.0, 0.0)),
        ScalarValue(value=0.0),
        Radius(radius=2.0),  # large enough to fill center pixel
        n=1,
    )

    # Frame 0 — scalar = 0.0, expect LUT[0]
    engine._draw_frame()
    img0 = engine._render_canvas.draw()
    assert img0.shape == (128, 128, 4)
    center0 = img0[64, 64]
    expected0 = colormaps.lookup("viridis", 0.0)

    # Mutate the scalar directly in the store and re-render.
    # The renderer re-uploads scalar_values from the ECS array each frame, so
    # this should produce a different color.
    engine.store._components["ScalarValue"][:, 0] = 1.0
    engine._draw_frame()
    img1 = engine._render_canvas.draw()
    center1 = img1[64, 64]
    expected1 = colormaps.lookup("viridis", 1.0)

    # Pixels must have actually changed.
    assert not np.array_equal(center0, center1), (
        f"scalar update did not propagate: center pixel {center0} == {center1}"
    )

    # Each rendered center pixel must be close to its expected colormap output.
    # Tolerance accounts for sphere-imposter shading (sRGB/linear conversion,
    # imposter-disk anti-aliasing at quad edges).
    def _close(a, b, atol=20):
        return np.all(np.abs(a.astype(int) - b.astype(int)) <= atol)

    assert _close(center0[:3], expected0[:3]), (
        f"frame 0 center pixel {center0[:3]} not close to LUT[0] {expected0[:3]}"
    )
    assert _close(center1[:3], expected1[:3]), (
        f"frame 1 center pixel {center1[:3]} not close to LUT[255] {expected1[:3]}"
    )
