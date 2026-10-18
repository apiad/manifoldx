"""Engine.origin and set_origin: float64 world positions, float32 relative rows."""

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform


def _spawn(engine, pos):
    return engine.spawn(Mesh(mx.geometry.cube(1, 1, 1)), Material(mx.material.standard("#fff")),
                        Transform(pos=pos))


def test_origin_starts_at_zero():
    assert mx.Engine("origin").origin == (0.0, 0.0)


def test_set_origin_keeps_world_positions_and_shifts_rows():
    engine = mx.Engine("origin")
    h = _spawn(engine, (10.0, 2.0, -5.0))
    engine.camera.set_pose((10, 12, 6), (10, 0, -5))
    engine.set_origin(8.0, -4.0)
    assert engine.origin == (8.0, -4.0)
    assert np.allclose(h.transform.pos, [10.0, 2.0, -5.0])
    assert np.allclose(engine.store._components["Transform"][h.index, 0:3], [2.0, 2.0, -1.0])
    assert np.allclose(engine.camera.position, [2, 12, 10])
    assert np.allclose(engine.camera.target, [2, 0, -1])


def test_far_positions_keep_float64_precision():
    engine = mx.Engine("origin")
    far = 50_000.123
    engine.set_origin(far - 3.0, 0.0)
    h = _spawn(engine, (far, 0.0, 0.0))
    assert h.transform.pos.dtype == np.float64
    assert abs(h.transform.pos[0] - far) < 1e-5     # float32 alone is off by ~4e-3 here
    h.transform.pos = (far + 0.001, 0.0, 0.0)
    assert abs(h.transform.pos[0] - (far + 0.001)) < 1e-5
    assert np.allclose(engine.store._components["Transform"][h.index, 0], 3.001)
