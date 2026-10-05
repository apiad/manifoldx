"""Engine.render_frame: one still frame, headless."""
import numpy as np
import pytest

import manifoldx as mx


def _engine(w=64, h=48):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    return mx.Engine("test", width=w, height=h)


def test_returns_rgb_at_engine_size():
    e = _engine()
    img = e.render_frame(supersample=2)
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8


def test_empty_scene_is_the_background_colour():
    e = _engine()
    e.background_color = (1.0, 0.0, 0.0)
    img = e.render_frame(supersample=1)
    assert tuple(img[24, 32]) == (255, 0, 0)


def test_consecutive_frames_are_identical():
    e = _engine()
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube
    e.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial(color="#3366cc")), Transform(pos=(0, 0, 0)))
    e.camera.set_pose(position=(2, 2, 3), target=(0, 0, 0))
    assert np.array_equal(e.render_frame(supersample=1), e.render_frame(supersample=1))


def test_supersample_is_fixed_per_engine():
    e = _engine()
    e.render_frame(supersample=2)
    with pytest.raises(ValueError, match="supersample"):
        e.render_frame(supersample=1)


def test_unknown_pass_is_rejected():
    e = _engine()
    with pytest.raises(ValueError, match="pass_"):
        e.render_frame(pass_="depth")
