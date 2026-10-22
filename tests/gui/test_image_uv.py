"""Image uv windows and textures.write_texture: the minimap's two needs."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.gui import Image, Panel, style
from manifoldx.textures import texture_from_array, write_texture


def setup_function(_):
    style.reset()


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=64, height=64)
    eng._init_canvas(canvas)
    eng._running = True
    return eng, canvas


def _map(eng):
    """A 64x32 map: red on the left half, green on the right."""
    rgba = np.zeros((32, 64, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[:, :32, 0] = 255
    rgba[:, 32:, 1] = 255
    return texture_from_array(eng, rgba)


def _draw(eng, canvas, img):
    eng.gui.clear()
    eng.gui.append(Panel([img], style_overrides={"padding": 0, "width": 64, "height": 64}))
    eng._draw_frame()
    return np.asarray(canvas.draw()).astype(int)


def test_uv_draws_a_window_of_the_source():
    eng, canvas = _engine()
    tex = _map(eng)
    img = Image(tex, style_overrides={"width": 64, "height": 64, "uv": (0.5, 0.0, 1.0, 1.0)})
    frame = _draw(eng, canvas, img)
    assert frame[32, 4, 1] > 200 and frame[32, 4, 0] < 40  # green across the box
    img.style_overrides["uv"] = (0.0, 0.0, 0.5, 1.0)  # pan, no upload
    frame = _draw(eng, canvas, img)
    assert frame[32, 60, 0] > 200 and frame[32, 60, 1] < 40  # now red across


def test_write_texture_patches_in_place():
    eng, canvas = _engine()
    tex = _map(eng)
    patch = np.zeros((32, 32, 4), np.uint8)
    patch[..., 2] = 255
    patch[..., 3] = 255
    write_texture(eng, tex, patch, (32, 0))
    frame = _draw(eng, canvas, Image(tex, style_overrides={"width": 64, "height": 64}))
    assert frame[32, 60, 2] > 200 and frame[32, 60, 1] < 40  # right half is blue now
    with pytest.raises(ValueError):
        write_texture(eng, tex, patch, (40, 0))
