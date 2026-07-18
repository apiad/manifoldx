"""GPU-gated render tests for directional sun + shadow mapping.

Gate on the offscreen wgpu backend (skip if unavailable) so the suite stays
runnable headless. Pattern mirrors tests/test_textured_material.py.
"""

import numpy as np
import pytest

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import DirectionalLight, StandardMaterial, sphere, plane


def _make_offscreen_engine(name="shadow-test", w=128, h=128):
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx

    engine = mx.Engine(name, width=w, height=h)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def _render_array(engine):
    engine._draw_frame()
    return engine._render_canvas.draw()  # HxWx4 uint8


# --------------------------------------------------------------------------
# Task 1 — directional sun
# --------------------------------------------------------------------------


def test_set_sun_stores_light():
    eng = _make_offscreen_engine()
    sun = DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -1.0, -0.3))
    eng.set_sun(sun)
    assert eng._sun is sun


def test_globals_buffer_is_352_bytes():
    eng = _make_offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, -1, 0)))
    eng.spawn(Mesh(sphere(1.0, 24)), Material(StandardMaterial(color="#ffffff", roughness=0.5)))
    _render_array(eng)
    assert eng._render_pipeline._globals_buffer.size == 352


def test_sun_lights_sphere_with_gradient():
    eng = _make_offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-1, -1, -1)))
    eng.spawn(
        Mesh(sphere(1.0, 32)),
        Material(StandardMaterial(color="#ffffff", roughness=0.5)),
        Transform(pos=(0, 0, 0)),
    )
    eng.camera.fit(radius=3.0, center=(0, 0, 0), azimuth=0, elevation=10)
    img = _render_array(eng)
    lum = img[..., :3].mean(axis=2)
    lit = lum[lum > 12]  # ignore dark-blue background
    assert lit.size > 0
    assert lit.max() - lit.min() > 20  # a Lambert gradient, not flat


# --------------------------------------------------------------------------
# Task 3 — shadow map resources
# --------------------------------------------------------------------------


def test_enable_shadows_stores_config():
    eng = _make_offscreen_engine()
    eng.enable_shadows(target=(0, 0, 0), extent=8.0, resolution=1024, bias=0.004)
    cfg = eng._shadow_config
    assert cfg["extent"] == 8.0 and cfg["resolution"] == 1024 and cfg["bias"] == 0.004


def test_shadow_map_allocated_when_enabled():
    eng = _make_offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, -1, 0)))
    eng.enable_shadows(target=(0, 0, 0), extent=10.0, resolution=512)
    eng.spawn(Mesh(plane(12, 12)), Material(StandardMaterial(color="#ffffff", roughness=0.8)))
    eng.spawn(
        Mesh(sphere(1.0, 16)), Material(StandardMaterial(color="#ffffff")), Transform(pos=(0, 2, 0))
    )
    _render_array(eng)
    tex = eng._render_pipeline._shadow_map
    assert tex is not None
    assert tex.size[0] == 512 and tex.size[1] == 512


# --------------------------------------------------------------------------
# Task 4 — cast shadow onto the ground
# --------------------------------------------------------------------------

# -90deg about X (quaternion x,y,z,w): turns plane()'s +Z normal into +Y (a floor).
_FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)


def _plane_sphere_scene(shadows, pcf=1):
    eng = _make_offscreen_engine(w=192, h=192)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.3)))
    if shadows:
        eng.enable_shadows(
            target=(0, 0, 0), extent=6.0, resolution=1024, bias=0.004, pcf_radius=pcf
        )
    eng.spawn(
        Mesh(plane(12, 12)),
        Material(StandardMaterial(color="#ffffff", roughness=0.9)),
        Transform(pos=(0, 0, 0), rot=_FLOOR_ROT),
    )
    eng.spawn(
        Mesh(sphere(1.0, 32)),
        Material(StandardMaterial(color="#ffffff", roughness=0.5)),
        Transform(pos=(0, 2, 0)),
    )
    eng.camera.fit(radius=6.0, center=(0, 0, 0), azimuth=25, elevation=55)
    return _render_array(eng)


def test_sphere_casts_shadow_on_plane():
    lit = _plane_sphere_scene(shadows=False)[..., :3].mean(axis=2).astype(np.float64)
    shadowed = _plane_sphere_scene(shadows=True)[..., :3].mean(axis=2).astype(np.float64)
    # A clearly darkened region (the cast shadow) must appear when shadows are on.
    darkening = lit - shadowed
    assert darkening.max() > 20, f"no shadow appeared (max darkening {darkening.max():.1f})"


# --------------------------------------------------------------------------
# VS2 — PCF soft shadows
# --------------------------------------------------------------------------


def test_enable_shadows_stores_pcf_radius():
    eng = _make_offscreen_engine()
    eng.enable_shadows(target=(0, 0, 0), extent=6.0, pcf_radius=3)
    assert eng._shadow_config["pcf_radius"] == 3
    # Default is a 3x3 soft kernel.
    eng.enable_shadows(target=(0, 0, 0), extent=6.0)
    assert eng._shadow_config["pcf_radius"] == 1


def _penumbra_count(lit, shadowed):
    """Pixels in the soft transition band (partial darkening) of the shadow."""
    d = lit - shadowed
    mx = d.max()
    if mx < 20:
        return 0
    return int(((d > 0.2 * mx) & (d < 0.8 * mx)).sum())


def test_pcf_widens_the_penumbra():
    lit = _plane_sphere_scene(shadows=False)[..., :3].mean(axis=2).astype(np.float64)
    hard = _plane_sphere_scene(shadows=True, pcf=0)[..., :3].mean(axis=2).astype(np.float64)
    soft = _plane_sphere_scene(shadows=True, pcf=4)[..., :3].mean(axis=2).astype(np.float64)
    hard_band = _penumbra_count(lit, hard)
    soft_band = _penumbra_count(lit, soft)
    # A wider PCF kernel must produce a wider partial-shadow (penumbra) band.
    assert soft_band > hard_band, f"pcf did not soften edges (hard={hard_band}, soft={soft_band})"
