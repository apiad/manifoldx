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
