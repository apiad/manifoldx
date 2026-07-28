import numpy as np
import pytest
import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh


def test_spawn_returns_handle_and_transform_writes():
    engine = mx.Engine("handle-test")
    cube = mx.geometry.cube(1, 1, 1)
    mat = mx.material.standard("#ffffff")
    h = engine.spawn(Mesh(cube), Material(mat), Transform(pos=(1, 2, 3)))
    assert h is not None and h.index >= 0
    assert np.allclose(h.transform.pos, [1, 2, 3])
    h.transform.pos = (4, 5, 6)
    assert np.allclose(engine.store._components["Transform"][h.index, 0:3], [4, 5, 6])


def test_set_geometry_rejects_vertex_count_mismatch():
    engine = mx.Engine("handle-test2")
    geo = GeoMesh.icosphere(subdivisions=2).to_geometry()
    h = engine.spawn(Mesh(geo), Material(mx.material.standard("#fff")), Transform())
    with pytest.raises(ValueError):
        h.set_geometry(GeoMesh.icosphere(subdivisions=3))   # different vertex count
