"""StandardMaterial(ripple=...): bands of light drift across a surface, like water."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial


def _scene(ripple, at=(0.0, 0.0, 0.0), vertex_colors=False):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("ripple", width=96, height=96)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.2)))
    pool = {"positions": np.array([(-4, 0, -4), (4, 0, -4), (4, 0, 4), (-4, 0, 4)], np.float32),
            "normals": np.tile(np.array([0, 1, 0], np.float32), (4, 1)),
            "indices": np.array([0, 2, 1, 0, 3, 2], np.uint32)}
    if vertex_colors:
        pool["colors"] = np.ones((4, 3), np.float32)
    mat = StandardMaterial("#3f97d6", roughness=0.75, ripple=ripple, vertex_colors=vertex_colors)
    e.spawn(Mesh(pool), Material(mat), Transform(pos=at))
    e.camera.set_pose(position=(at[0], 6.0, at[2] + 3), target=(at[0], 0.0, at[2]))
    e.render_frame(supersample=1)   # the first frame sets the engine up and restarts its clock
    return e


def _at(e, t):
    e.elapsed = t
    return e.render_frame(supersample=1).astype(np.int16)


def test_rippling_water_changes_over_time():
    e = _scene(ripple=1.0)
    assert np.abs(_at(e, 1.0) - _at(e, 2.5)).max() > 30


def test_the_bands_cover_part_of_the_water_not_all():
    lit = _at(_scene(ripple=1.0), 1.0)
    flat = _at(_scene(ripple=0.0), 1.0)
    brighter = (lit - flat).max(axis=-1) > 20
    assert 0.03 < brighter.mean() < 0.6


def test_vertex_coloured_water_ripples_too():
    e = _scene(ripple=1.0, vertex_colors=True)
    assert np.abs(_at(e, 1.0) - _at(e, 2.5)).max() > 30


def test_without_ripple_the_water_holds_still():
    e = _scene(ripple=0.0)
    assert np.abs(_at(e, 1.0) - _at(e, 2.5)).max() == 0


def test_moving_the_origin_does_not_jump_the_ripples():
    e = _scene(ripple=1.0, at=(100.0, 0.0, 37.0))
    before = _at(e, 1.7)
    e.set_origin(91.0, 30.0)
    assert np.abs(_at(e, 1.7) - before).max() <= 2


def test_ripple_reaches_the_material_uniforms():
    data = StandardMaterial("#ffffff", ripple=0.7).get_data(1, None)
    assert data.shape == (1, 12) and data[0, 8] == pytest.approx(0.7)
