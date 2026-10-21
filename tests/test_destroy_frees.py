"""EntityHandle.destroy frees its geometry's GPU buffers."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial


def _offscreen_engine():
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=32, height=32)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("destroy-frees", width=32, height=32)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def _quad(k):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], np.float32) + k * 1e-3
    return {"positions": pos, "normals": np.tile([0, 1, 0], (3, 1)).astype(np.float32),
            "colors": np.ones((3, 3), np.float32), "indices": np.array([0, 1, 2], np.uint32)}


def test_ten_thousand_meshes_created_and_destroyed_leave_no_buffers():
    engine = _offscreen_engine()
    reg = engine._geometry_registry
    mat = StandardMaterial("#ffffff", vertex_colors=True)
    engine.camera.set_pose(position=(1.0, 8.0, 1.5), target=(1.0, 0.0, 1.0))   # every quad in view: none culled
    engine._draw_frame()
    start_buffers, start_geos = len(reg._gpu_buffers), len(reg._geometries)
    made = 0
    for _ in range(10):
        handles = [engine.spawn(Mesh(_quad(k)), Material(mat), Transform()) for k in range(1000)]
        engine._draw_frame()  # draws them: their buffers exist now
        made += sum(reg.get_gpu_buffers(h._geometry_id) is not None for h in handles)
        for h in handles:
            h.destroy()
        engine._draw_frame()
    assert made == 10_000
    assert len(reg._gpu_buffers) == start_buffers
    assert len(reg._geometries) == start_geos


def test_shared_geometry_stays_while_another_entity_draws_it():
    engine = _offscreen_engine()
    reg = engine._geometry_registry
    mesh = Mesh(_quad(0))
    mat = Material(StandardMaterial("#ffffff", vertex_colors=True))
    a = engine.spawn(mesh, mat, Transform())
    b = engine.spawn(mesh, mat, Transform())
    engine._draw_frame()
    a.destroy()
    engine._draw_frame()
    assert reg.get_gpu_buffers(b._geometry_id) is not None
    b.destroy()
    engine._draw_frame()
    assert reg.get(b._geometry_id) is None
    # The same Mesh spawned again registers anew and draws.
    c = engine.spawn(mesh, mat, Transform())
    engine._draw_frame()
    assert reg.get_gpu_buffers(c._geometry_id) is not None
