"""Benders: swaying meshes part around points set on the engine, such as grass around
whatever walks through it. GPU-gated like the other render tests."""

import numpy as np
import pytest

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, cube


def _engine(w=128, h=128):
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx

    engine = mx.Engine("benders-test", width=w, height=h)
    engine._init_canvas(canvas)
    engine._running = True
    engine.camera.move_to((0.0, 0.4, 2.0))
    engine.camera.look_at((0.0, 0.4, 0.0))
    return engine


def _blade(engine, x=0.0, sway=1e-4):
    """A thin dark blade of grass standing from y = 0 to 0.7 at (x, 0)."""
    m = cube(0.06, 0.7, 0.06)
    m["positions"] = m["positions"] + np.array([0.0, 0.35, 0.0], dtype=np.float32)
    engine.spawn(Mesh(m), Material(StandardMaterial("#000000", sway=sway, sway_reach=0.6)),
                 Transform(pos=(x, 0.0, 0.0)))


def _top_x(engine):
    """The mean column of the blade's dark pixels in the top third of the picture's blade."""
    engine._draw_frame()
    img = engine._render_canvas.draw()[:, :, :3].astype(int)
    dark = img.sum(axis=2) < 200
    rows = np.nonzero(dark.any(axis=1))[0]
    top = rows[: max(1, len(rows) // 4)]
    return float(np.nonzero(dark[top])[1].mean())


def test_a_bender_pushes_the_top_of_a_blade_away():
    engine = _engine()
    _blade(engine)
    still = _top_x(engine)
    engine.benders = [(-0.3, 0.0, 0.8, 1.0)]       # just left of the blade: its top leans right
    pushed = _top_x(engine)
    assert pushed > still + 3


def test_a_bender_out_of_reach_or_on_a_still_mesh_does_nothing():
    engine = _engine()
    _blade(engine)
    still = _top_x(engine)
    engine.benders = [(-2.0, 0.0, 0.8, 1.0)]
    assert abs(_top_x(engine) - still) < 0.5

    engine = _engine()
    _blade(engine, sway=0.0)                       # not a swaying mesh
    still = _top_x(engine)
    engine.benders = [(-0.3, 0.0, 0.8, 1.0)]
    assert abs(_top_x(engine) - still) < 0.5


def test_benders_follow_the_floating_origin():
    engine = _engine()
    _blade(engine, x=100.0)
    engine.set_origin(100.0, 0.0)
    engine.camera.move_to((0.0, 0.4, 2.0))
    engine.camera.look_at((0.0, 0.4, 0.0))
    still = _top_x(engine)
    engine.benders = [(99.7, 0.0, 0.8, 1.0)]       # world coordinates, like Transform.pos
    assert _top_x(engine) > still + 3


def test_only_the_first_eight_count():
    engine = _engine()
    _blade(engine)
    still = _top_x(engine)
    engine.benders = [(-5.0, 0.0, 0.1, 1.0)] * 8 + [(-0.3, 0.0, 0.8, 1.0)]
    assert abs(_top_x(engine) - still) < 0.5
