"""GUI colours are sRGB: a hex renders as itself, unless gui.exact_colors is off."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.gui import Panel, Text, style


def setup_function(_):
    style.reset()


def _frame(exact: bool, bg="#808080ff", fg=None):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=64, height=64)
    eng._init_canvas(canvas)
    eng._running = True
    eng.gui.exact_colors = exact
    children = [Text("MMMM", style_overrides={"fg": fg, "font_size": 40})] if fg else []
    eng.gui.append(Panel(children, style_overrides={"bg": bg, "padding": 0,
                                                    "width": 64, "height": 64}))
    eng._draw_frame()
    return np.asarray(canvas.draw()).astype(int)


def test_panel_bg_renders_as_its_hex():
    assert np.abs(_frame(True)[32, 32, :3] - 128).max() <= 1


def test_opt_out_keeps_the_lighter_look():
    assert _frame(False)[32, 32, :3].min() > 170  # 0.502 as linear light encodes to ~188


def test_text_fg_renders_as_its_hex():
    frame = _frame(True, bg="#000000ff", fg="#808080")
    solid = frame[..., 0][frame[..., 0] > 0]
    assert abs(int(solid.max()) - 128) <= 1  # fully covered glyph pixels


def test_translucent_bg_blends_as_itself():
    # 50% white over black: the result is half of white's light, sRGB ~188
    frame = _frame(True, bg="#ffffff80")
    clear = _frame(True, bg="#00000000")[32, 32, :3]
    lin = lambda c: ((c / 255 + 0.055) / 1.055) ** 2.4  # noqa: E731
    expect = 0.502 * 1.0 + 0.498 * lin(clear)
    got = lin(frame[32, 32, :3])
    assert np.abs(got - expect).max() < 0.02


def test_translucent_bg_with_a_border_keeps_its_alpha():
    a = _frame(True, bg="#ffffff00")[32, 32, :3]  # transparent fill, no border
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=64, height=64)
    eng._init_canvas(canvas)
    eng._running = True
    eng.gui.append(Panel([], style_overrides={"bg": "#ffffff00", "border": 4,
                                              "border_color": "#ff0000", "padding": 0,
                                              "width": 64, "height": 64}))
    eng._draw_frame()
    frame = np.asarray(canvas.draw()).astype(int)
    assert np.abs(frame[32, 32, :3] - a).max() <= 2  # the transparent fill stays transparent
    assert frame[1, 32, 0] > 200 and frame[1, 32, 1] < 40  # the border is drawn
