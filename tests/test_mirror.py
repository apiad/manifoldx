"""StandardMaterial(mirror=y) draws a mesh reflected in the plane y; opacity < 1 blends it over
what is behind, drawn after every opaque mesh. Together: a cheap reflection on water."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("mirror", width=96, height=96)
    e.background_color = (0.0, 0.0, 0.0)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.5)))
    e.camera.set_pose(position=(0.0, 2.0, 6.0), target=(0.0, 0.0, 0.0))
    return e


def _pole(y0=0.0):
    pole = mx.geometry.cube(0.6, 1.5, 0.6)
    pole["positions"] = pole["positions"] + np.array([0.0, y0 + 0.75, 0.0], np.float32)
    return pole


def _quad(y, half=3.0):
    return {"positions": np.array([(-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half)], np.float32),
            "normals": np.tile(np.array([0, 1, 0], np.float32), (4, 1)),
            "indices": np.array([0, 2, 1, 0, 3, 2], np.uint32)}


def _red(img):
    img = img.astype(np.int16)
    return (img[..., 0] > 60) & (img[..., 0] > 2 * img[..., 2])


def _rows(mask):
    return np.nonzero(mask.any(axis=1))[0]


def test_a_mirrored_pole_shows_below_the_plane():
    e = _engine()
    e.spawn(Mesh(_pole()), Material(StandardMaterial("#cc3333")), Transform())
    alone = _red(e.render_frame(supersample=1))
    e.spawn(Mesh(_pole()), Material(StandardMaterial("#cc3333", mirror=0.0)), Transform())
    both = _red(e.render_frame(supersample=1))
    assert _rows(both).max() > _rows(alone).max() + 10   # the reflection hangs below the pole


def test_what_lies_below_the_plane_is_not_mirrored():
    e = _engine()
    e.spawn(Mesh(_pole(y0=-2.0)), Material(StandardMaterial("#cc3333", mirror=0.0)), Transform())
    assert not _red(e.render_frame(supersample=1)).any()


def _pond(opacity, pole_first=False):
    e = _engine()
    water = (Mesh(_quad(0.0)), Material(StandardMaterial("#3355ff", opacity=opacity)), Transform())
    pole = (Mesh(_pole(y0=-2.0)), Material(StandardMaterial("#cc3333")), Transform())
    for parts in (pole, water) if pole_first else (water, pole):
        e.spawn(*parts)
    return e.render_frame(supersample=1).astype(np.int16)[..., 0]


def test_a_translucent_surface_shows_what_is_behind_it():
    assert (_pond(0.5) - _pond(1.0)).max() > 30


def test_translucent_meshes_draw_last_whatever_the_spawn_order():
    assert np.abs(_pond(0.5) - _pond(0.5, pole_first=True)).max() <= 2


def test_mirrored_and_translucent_meshes_cast_no_shadow():
    e = _engine()
    e.enable_shadows(extent=4.0)
    e.spawn(Mesh(_pole()), Material(StandardMaterial("#cc3333", mirror=0.0)), Transform())
    e.spawn(Mesh(_quad(0.5)), Material(StandardMaterial("#3355ff", opacity=0.5)), Transform())
    e.render_frame(supersample=1)
    assert e._render_pipeline._collect_mesh_instances(e) == {}


def test_mirror_and_opacity_reach_the_material_uniforms():
    data = StandardMaterial("#ffffff", mirror=0.03, opacity=0.6).get_data(1, None)
    assert data[0, 9] == 1.0 and data[0, 10] == pytest.approx(0.03) and data[0, 11] == pytest.approx(0.6)
    assert StandardMaterial("#ffffff").get_data(1, None)[0, 9:12].tolist() == [0.0, 0.0, 1.0]


def test_a_cut_drops_what_lies_inside_its_rectangle():
    e = _engine()
    mat = StandardMaterial("#3355ff")
    e.spawn(Mesh(_quad(0.0)), Material(mat), Transform())
    whole = e.render_frame(supersample=1).astype(np.int16)[..., 2] > 30
    mat.cut = (-1.0, -1.0, 1.0, 1.0)
    holed = e.render_frame(supersample=1).astype(np.int16)[..., 2] > 30
    assert 0 < whole.sum() - holed.sum() < whole.sum() / 2
    assert StandardMaterial("#ffffff", cut=(1, 2, 3, 4)).get_data(1, None)[0, 12:].tolist() == [1, 2, 3, 4]
