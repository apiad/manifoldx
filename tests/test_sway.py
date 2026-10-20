"""StandardMaterial(sway=...): vertices sway with height, in time and across the world."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial


def _scene(sway, at=(0.0, 0.0, 0.0)):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("sway", width=96, height=96)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -0.7, -0.5)))
    pole = mx.geometry.cube(0.2, 3.0, 0.2)
    pole["positions"] = pole["positions"] + np.array([0.0, 1.5, 0.0], np.float32)  # base on y=0
    e.spawn(Mesh(pole), Material(StandardMaterial("#cc3333", sway=sway)), Transform(pos=at))
    e.camera.set_pose(position=(at[0], 1.5, at[2] + 5), target=(at[0], 1.5, at[2]))
    e.render_frame(supersample=1)   # the first frame sets the engine up and restarts its clock
    return e


def _at(e, t):
    e.elapsed = t
    return e.render_frame(supersample=1).astype(np.int16)


def test_a_swaying_pole_moves_over_time():
    e = _scene(sway=0.08)
    assert np.abs(_at(e, 1.0) - _at(e, 2.5)).max() > 40


def test_without_sway_nothing_moves():
    e = _scene(sway=0.0)
    assert np.abs(_at(e, 1.0) - _at(e, 2.5)).max() == 0


def test_moving_the_origin_does_not_jump_the_sway():
    e = _scene(sway=0.08, at=(100.0, 0.0, 37.0))
    before = _at(e, 1.7)
    e.set_origin(91.0, 30.0)
    assert np.abs(_at(e, 1.7) - before).max() <= 2


def test_sway_reaches_the_material_uniforms():
    data = StandardMaterial("#ffffff", sway=0.05).get_data(1, None)
    assert data[0, 6] == pytest.approx(0.05)
