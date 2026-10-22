"""Surface patterns: a per-vertex id picks a world-space pattern round the vertex colour."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import SURFACE_PATTERNS, DirectionalLight, StandardMaterial, material_rgb


def _ground(sid, size=12.0, with_key=True):
    h = size / 2
    geo = {
        "positions": np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h]], np.float32),
        "normals": np.tile([0, 1, 0], (4, 1)).astype(np.float32),
        "colors": np.tile(material_rgb("#7fb653"), (4, 1)).astype(np.float32),
        "indices": np.array([0, 2, 1, 0, 3, 2], np.uint32),
    }
    if with_key:
        geo["surfaces"] = np.full(4, sid, np.int32)
    return geo


def _render(geo, eye=(0, 3, 4), target=(0, 0, 0), patterns=None):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=96, height=96)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=96, height=96)
    eng._init_canvas(canvas)
    eng._running = True
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.4)))
    mat = StandardMaterial(color="#ffffff", vertex_colors=True, surface_patterns=patterns)
    eng.spawn(Mesh(geo), Material(mat), Transform(pos=(0, 0, 0)))
    eng.camera.set_pose(position=eye, target=target)
    eng._draw_frame()
    return np.asarray(canvas.draw())[..., :3].astype(int)


def test_a_pattern_varies_the_colour_and_id_zero_stays_flat():
    flat = _render(_ground(0))[40:90, 10:86]
    grass = _render(_ground(1))[40:90, 10:86]
    assert flat.std(axis=(0, 1)).max() < 4  # one flat colour, shaded smoothly
    assert grass.std(axis=(0, 1)).max() > 8  # tones and tufts
    assert np.abs(flat.mean(axis=(0, 1)) - grass.mean(axis=(0, 1))).max() < 25  # round the same colour


def test_geometry_without_surfaces_draws_flat():
    a = _render(_ground(0, with_key=False))
    b = _render(_ground(0))
    assert np.abs(a - b).max() <= 1


def test_a_far_surface_fades_to_its_flat_colour():
    far = dict(eye=(0, 8, 30), target=(0, 0, 0))  # a cell spans 2-3 pixels: under the fade
    grass = _render(_ground(1, size=200.0), **far)[50:90, 10:86]
    flat = _render(_ground(0, size=200.0), **far)[50:90, 10:86]
    assert np.abs(flat.mean(axis=(0, 1)) - grass.mean(axis=(0, 1))).max() < 3
    assert grass.std(axis=(0, 1)).max() < flat.std(axis=(0, 1)).max() + 2  # no pattern left


def test_patterns_ride_in_the_material_uniforms():
    m = StandardMaterial(color="#ffffff", vertex_colors=True, surface_patterns={1: (2.0, 0.3, 2, 0.1, 0.9)})
    d = m.get_data(1, None)[0]
    assert d.size == 60
    assert d[16:20].tolist() == pytest.approx([2.0, 0.3, 2, 0.1])  # grass, overridden
    assert d[44] == pytest.approx(0.9)  # grass darkness: 16 + 28
    assert d[20:24].tolist() == pytest.approx(SURFACE_PATTERNS[2][:4])  # cobbles, default
    assert StandardMaterial(color="#ffffff").get_data(1, None).shape == (1, 16)  # no vertex colours
