import numpy as np
import manifoldx as mx
from manifoldx.sky import starfield


def test_starfield_spawns_stars_on_sphere():
    engine = mx.Engine("stars")
    starfield(engine, count=300, radius=100.0, seed=1)
    assert int(engine.store._alive.sum()) == 300
    pos = engine.store._components["Transform"][engine.store._alive][:, 0:3]
    r = np.linalg.norm(pos, axis=1)
    assert np.allclose(r, 100.0, atol=1e-3)     # stars sit on the sky sphere
