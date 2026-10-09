"""Render quality v1, phase 1: reversed-Z depth, tonemap curves with exposure, FXAA.

Design: .knowledge/analysis/2026-10-09-render-quality-v1-design.md (#30).
"""

import math

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, FlatMaterial, StandardMaterial, plane


def _engine(w=32, h=32):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("rq1", width=w, height=h)
    e.camera.set_pose(position=(0, 0, 3), target=(0, 0, 0))
    return e


def _linear(byte):
    c = byte / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# -- reversed-Z ---------------------------------------------------------------------------

def test_projection_maps_near_to_one_and_far_to_zero():
    from manifoldx.camera import Camera

    cam = Camera(position=(0, 0, 0), target=(0, 0, -1))
    p = cam.get_projection_matrix(aspect=1.0, near=0.5, far=1000.0)
    for z_view, depth in ((-0.5, 1.0), (-1000.0, 0.0)):
        clip = p @ np.array([0.0, 0.0, z_view, 1.0])
        assert clip[2] / clip[3] == pytest.approx(depth, abs=1e-5), z_view


def test_main_depth_is_32_bit_float():
    e = _engine()
    e.render_frame(supersample=1)
    assert "depth32float" in str(e._depth_texture.format)


def test_planes_two_centimetres_apart_at_500_metres_do_not_fight():
    """Forward depth24 with near 0.1 resolves about 15 cm at 500 m, so these two planes
    interleave; reversed-Z depth32float keeps the nearer one everywhere."""
    e = _engine(48, 48)
    e.camera.near, e.camera.far = 0.1, 2000.0
    e.camera.set_pose(position=(0, 0, 0), target=(0, 0, -1))
    e.spawn(Mesh(plane(600, 600)), Material(FlatMaterial("#ff0000")), Transform(pos=(0, 0, -500.0)))
    e.spawn(Mesh(plane(600, 600)), Material(FlatMaterial("#0000ff")), Transform(pos=(0, 0, -500.02)))
    img = e.render_frame(supersample=1).astype(int)
    red = (img[..., 0] > 200) & (img[..., 2] < 50)
    assert red.all(), f"{(~red).sum()} of {red.size} pixels show the farther plane"


# -- tonemap curves and exposure ---------------------------------------------------------------

def _lit_pixel(mode, exposure, sun, color="#808080"):
    e = _engine()
    e.set_tonemap(mode, exposure=exposure)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=sun, direction=(0, 0, -1)))
    e.spawn(Mesh(plane(10, 10)), Material(StandardMaterial(color=color, roughness=1.0)), Transform(pos=(0, 0, 0)))
    return e.render_frame(supersample=1)[16, 16].astype(int)


def test_default_tonemap_is_reinhard_at_exposure_one():
    e = _engine()
    assert e.tonemap == ("reinhard", 1.0)


def test_set_tonemap_rejects_unknown_curves_and_bad_exposure():
    e = _engine()
    with pytest.raises(ValueError):
        e.set_tonemap("filmic-ish")
    with pytest.raises(ValueError):
        e.set_tonemap("agx", exposure=0.0)


def test_reinhard_at_exposure_one_matches_the_old_shader():
    """Reinhard c/(c+1) per channel, as StandardMaterial wrote before v1."""
    explicit = _lit_pixel("reinhard", 1.0, 3.0)
    e = _engine()
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, 0, -1)))
    e.spawn(Mesh(plane(10, 10)), Material(StandardMaterial(color="#808080", roughness=1.0)), Transform(pos=(0, 0, 0)))
    default = e.render_frame(supersample=1)[16, 16].astype(int)
    assert np.array_equal(explicit, default)


@pytest.mark.parametrize("mode", ["agx", "aces"])
def test_filmic_curves_are_monotonic_and_do_not_clip(mode):
    lum = [int(_lit_pixel(mode, 1.0, s).mean()) for s in (0.25, 0.5, 1, 2, 4, 8, 16)]
    assert all(b >= a for a, b in zip(lum, lum[1:])), lum
    assert lum[-1] < 255, lum  # sixteen times brighter still has headroom
    assert lum[-1] > lum[0] + 60, lum


@pytest.mark.parametrize("mode", ["reinhard", "agx", "aces", "none"])
def test_every_curve_keeps_black_black(mode):
    assert _lit_pixel(mode, 1.0, 4.0, color="#000000").max() <= 2


def test_exposure_scales_linear_light():
    one = _lit_pixel("none", 1.0, 0.15)
    two = _lit_pixel("none", 2.0, 0.15)
    ratio = _linear(two[0]) / _linear(one[0])
    assert ratio == pytest.approx(2.0, rel=0.06), (one, two)


def test_the_skybox_uses_the_same_curve():
    from manifoldx.render.passes import skybox

    sources = [v for v in vars(skybox).values() if isinstance(v, str) and "fn fs_main" in v]
    assert sources and not any("color / (color + vec3<f32>(1.0))" in s for s in sources)


def test_unlit_colours_ignore_the_tonemap():
    e = _engine()
    e.set_tonemap("agx", exposure=3.0)
    e.spawn(Mesh(plane(10, 10)), Material(FlatMaterial("#cc2222")), Transform(pos=(0, 0, 0)))
    px = e.render_frame(supersample=1)[16, 16]
    assert all(abs(int(a) - b) <= 1 for a, b in zip(px, (0xCC, 0x22, 0x22))), px


# -- FXAA and the final pass -----------------------------------------------------------------

def _diagonal_edge(antialias):
    e = _engine(64, 64)
    e.background_color = (0.0, 0.0, 0.0)
    e.set_antialias(antialias)
    half = math.radians(30) / 2
    e.spawn(Mesh(plane(4, 4)), Material(FlatMaterial("#ffffff")),
            Transform(pos=(0.9, 0, 0), rot=(0, 0, math.sin(half), math.cos(half))))
    return e.render_frame(supersample=1)[..., 0].astype(int)


def test_without_antialiasing_the_edge_is_hard():
    img = _diagonal_edge(None)
    assert ((img <= 1) | (img >= 254)).all()


def test_fxaa_softens_the_edge_and_leaves_flat_regions_alone():
    hard, soft = _diagonal_edge(None), _diagonal_edge("fxaa")
    edge = (soft > 20) & (soft < 235)
    assert edge.sum() >= 10, "FXAA left the edge hard"
    flat = (np.abs(np.diff(hard, axis=0, prepend=hard[:1])) == 0) & (np.abs(np.diff(hard, axis=1, prepend=hard[:, :1])) == 0)
    flat[:2, :] = flat[-2:, :] = flat[:, :2] = flat[:, -2:] = False
    interior = flat & (np.abs(hard - np.roll(hard, 2, 0)) == 0) & (np.abs(hard - np.roll(hard, 2, 1)) == 0)
    assert np.abs(soft[interior] - hard[interior]).max() <= 1


def test_set_antialias_rejects_unknown_modes():
    e = _engine()
    with pytest.raises(ValueError):
        e.set_antialias("msaa8")


def test_id_pass_is_exact_with_fxaa_on():
    e = _engine()
    e.set_antialias("fxaa")
    a = e.spawn(Mesh(plane(1.2, 1.2)), Material(StandardMaterial(color="#ff0000")), Transform(pos=(-0.8, 0, 0)))
    b = e.spawn(Mesh(plane(1.2, 1.2)), Material(StandardMaterial(color="#00ff00")), Transform(pos=(0.8, 0, 0)))
    labels = e.render_frame(supersample=1, pass_="ids", groups=[[a.index], [b.index]])
    assert set(np.unique(labels)) == {0, 1, 2}


def test_gui_draws_after_the_final_pass():
    from manifoldx.gui import Panel, Text

    e = _engine(64, 64)
    e.set_antialias("fxaa")
    e.background_color = (0.0, 0.0, 0.0)
    e.gui.append(Panel(children=[Text("x")], anchor="top-left", offset=(0, 0),
                       style_overrides={"width": 40, "height": 40, "bg": "#ff0000", "padding": 0}))
    px = e.render_frame(supersample=1)[30, 5].astype(int)
    assert px[0] > 200 and px[1] < 80 and px[2] < 80, px


# -- review focus ----------------------------------------------------------------------------------

def test_a_tonemap_set_between_frames_applies_to_the_next_one():
    e = _engine()
    e.set_sun(DirectionalLight(color="#ffffff", intensity=6.0, direction=(0, 0, -1)))
    e.spawn(Mesh(plane(10, 10)), Material(StandardMaterial(color="#808080", roughness=1.0)), Transform(pos=(0, 0, 0)))
    before = e.render_frame(supersample=1)[16, 16].astype(int)
    e.set_tonemap("none", exposure=0.25)
    after = e.render_frame(supersample=1)[16, 16].astype(int)
    assert after.mean() < before.mean() - 20, (before, after)


def test_supersampled_stills_run_the_final_pass_on_the_large_target():
    e = _engine(32, 32)
    e.set_antialias("fxaa")
    e.spawn(Mesh(plane(10, 10)), Material(FlatMaterial("#cc2222")), Transform(pos=(0, 0, 0)))
    img = e.render_frame(supersample=2)
    assert img.shape == (32, 32, 3)
    assert all(abs(int(a) - b) <= 1 for a, b in zip(img[16, 16], (0xCC, 0x22, 0x22))), img[16, 16]
