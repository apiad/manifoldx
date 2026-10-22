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
