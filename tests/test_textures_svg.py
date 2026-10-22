"""texture_from_array / texture_from_svg, and albedo alpha in StandardMaterial."""

import numpy as np
import pytest

ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20" viewBox="0 0 40 20">'
    '<rect width="20" height="20" fill="#ff0000"/></svg>'
)


def _engine(w=96, h=96):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("svg-texture", width=w, height=h)
    engine._init_canvas(canvas)
    engine._running = True
    return engine, canvas


def test_texture_from_array_sizes_and_rejects_bad_input():
    from manifoldx.textures import texture_from_array

    engine, _ = _engine()
    assert texture_from_array(engine, np.zeros((8, 16, 4), np.uint8)).size == (16, 8)
    assert texture_from_array(engine, np.zeros((8, 16, 3), np.uint8)).size == (16, 8)
    with pytest.raises(ValueError):
        texture_from_array(engine, np.zeros((8, 16), np.uint8))
    with pytest.raises(ValueError):
        texture_from_array(engine, np.zeros((8, 16, 4), np.float32))


def test_texture_from_svg_keeps_aspect_or_takes_exact_size():
    pytest.importorskip("thorvg_python")
    from manifoldx.textures import texture_from_svg

    engine, _ = _engine()
    assert texture_from_svg(engine, ICON, 256).size == (256, 128)
    assert texture_from_svg(engine, ICON.encode(), (64, 64)).size == (64, 64)


def _centre_and_corner(rgba):
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import PointLight, StandardMaterial, sphere
    from manifoldx.textures import texture_from_array

    engine, canvas = _engine()
    tex = texture_from_array(engine, np.full((4, 4, 4), rgba, np.uint8))
    mat = StandardMaterial(color="#ffffff", albedo_map=tex)
    engine.spawn(Mesh(sphere(1.0, segments=16)), Material(mat), Transform(pos=(0, 0, 0)))
    engine.add_light(PointLight(position=(2, 3, 3), color="#ffffff", intensity=25.0))
    engine._draw_frame()
    frame = np.asarray(canvas.draw()).astype(int)
    return frame[48, 48, :3], frame[2, 2, :3]


def test_transparent_texels_are_cut_out():
    centre, corner = _centre_and_corner((255, 255, 255, 255))
    assert np.abs(centre - corner).sum() > 60  # opaque: the sphere covers the centre
    centre, corner = _centre_and_corner((255, 255, 255, 0))
    assert np.abs(centre - corner).sum() <= 3  # transparent: the background shows
