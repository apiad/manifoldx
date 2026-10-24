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


def test_set_material_writes_the_registered_id():
    engine = mx.Engine("handle-test3")
    cube = mx.geometry.cube(1, 1, 1)
    h = engine.spawn(Mesh(cube), Material(mx.material.standard("#ffffff")), Transform())
    red = mx.material.basic("#ff0000")
    h.set_material(red)
    reg_id = engine.store._components["Material"][h.index, 0]
    assert engine._material_registry.get(int(reg_id)) is red
    h.set_material(red)  # the same object keeps its id
    assert engine.store._components["Material"][h.index, 0] == reg_id


def test_set_material_to_another_type_changes_the_drawn_colour():
    from manifoldx.resources import BasicMaterial, PhongMaterial, sphere

    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("handle-test4", width=64, height=64)
    engine._init_canvas(canvas)
    engine._running = True
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    h = engine.spawn(Mesh(sphere(1.0, 16)), Material(PhongMaterial("#ff0000")), Transform())
    engine._draw_frame()
    before = np.asarray(canvas.draw()).astype(int)[32, 32]
    h.set_material(BasicMaterial("#00ff00"))
    engine._draw_frame()
    after = np.asarray(canvas.draw()).astype(int)[32, 32]
    assert before[0] > 100 and before[1] < 40
    assert after[1] > 100 and after[0] < 40
