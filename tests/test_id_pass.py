"""Id pass: one exact label per group, scene restored afterwards."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, sphere


def _scene():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("ids", width=96, height=48)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -0.7, -0.5)))
    a = e.spawn(Mesh(sphere(0.8, 32)), Material(StandardMaterial(color="#cc3333")), Transform(pos=(-1.2, 0, 0)))
    b = e.spawn(Mesh(sphere(0.8, 32)), Material(StandardMaterial(color="#3333cc")), Transform(pos=(1.2, 0, 0)))
    c = e.spawn(Mesh(sphere(0.3, 16)), Material(StandardMaterial(color="#33cc33")), Transform(pos=(0, 1.0, 0)))
    e.camera.set_pose(position=(0, 0, 4), target=(0, 0, 0))
    return e, a.index, b.index, c.index


def test_each_group_gets_exactly_one_label():
    e, a, b, _ = _scene()
    labels = e.render_frame(supersample=1, pass_="ids", groups=[[a], [b]])
    assert labels.shape == (48, 96) and labels.dtype == np.int32
    left, right = labels[:, :40], labels[:, 56:]
    assert set(np.unique(left)) == {0, 1}   # lit sphere, still one label
    assert set(np.unique(right)) == {0, 2}


def test_unlabelled_entities_read_as_background():
    e, a, b, c = _scene()
    labels = e.render_frame(supersample=1, pass_="ids", groups=[[a], [b]])
    assert labels[8, 48] == 0  # the small sphere c sits here, in no group


def test_beauty_is_unchanged_by_an_id_pass():
    e, a, b, _ = _scene()
    before = e.render_frame(supersample=1)
    e.render_frame(supersample=1, pass_="ids", groups=[[a], [b]])
    assert np.array_equal(before, e.render_frame(supersample=1))


def test_skybox_does_not_leak_into_the_background():
    e, a, b, _ = _scene()
    # Present one frame before enabling the skybox: enabling it before the
    # first present locks up some drivers (CHANGELOG, IBL known limitation).
    e.render_frame(supersample=1)
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_sky(zenith=(0.2, 0.4, 0.9), horizon=(0.8, 0.8, 0.9), ground=(0.2, 0.2, 0.2))
    env.show_skybox = True
    e.set_environment(env)
    labels = e.render_frame(supersample=1, pass_="ids", groups=[[a], [b]])
    assert labels[2, 48] == 0
