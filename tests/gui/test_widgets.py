"""Tests for manifoldx.gui.widgets — Widget base, Panel, Text."""

import pytest

from manifoldx.gui import style
from manifoldx.gui.widgets import Panel, Text, Widget, _GuiRoot


def setup_function(_):
    style.reset()


def test_text_construction_keeps_string_and_default_style():
    t = Text("hello")
    assert t.text == "hello"
    assert t.style is None
    assert t.style_overrides == {}


def test_text_effective_style_resolves_theme_then_class_then_overrides():
    style.set_theme({"font_size": 12})
    style.define("big", {"font_size": 20})
    t = Text("x", style="big", style_overrides={"font_size": 30})
    assert t.effective_style()["font_size"] == 30


def test_text_intrinsic_size_grows_with_font_size():
    # Intrinsic size is a function of font_size and char count;
    # exact values depend on the rasterizer, but the relationship must hold.
    small = Text("hello", style_overrides={"font_size": 10})
    big = Text("hello", style_overrides={"font_size": 40})
    sw, sh = small.intrinsic_size()
    bw, bh = big.intrinsic_size()
    assert bw > sw
    assert bh > sh


def test_panel_holds_children_in_order():
    a, b = Text("a"), Text("b")
    p = Panel(children=[a, b])
    assert list(p.children) == [a, b]


def test_panel_anchor_defaults_to_top_left():
    p = Panel(children=[])
    assert p.anchor == "top-left"
    assert p.offset == (0, 0)


def test_panel_rejects_unknown_anchor():
    with pytest.raises(ValueError):
        Panel(children=[], anchor="nowhere")


def test_panel_build_spec_includes_padding_gap_direction_from_style():
    style.set_theme({"padding": 8, "gap": 4, "direction": "h"})
    p = Panel(children=[Text("a"), Text("b")])
    spec = p.build_layout_spec()
    assert spec["padding"] == (8, 8, 8, 8)
    assert spec["gap"] == 4
    assert spec["direction"] == "h"
    assert len(spec["children"]) == 2


def test_widget_is_abstract():
    with pytest.raises(TypeError):
        Widget()  # type: ignore[abstract]


def test_gui_root_is_listlike_and_has_pointer_over_gui_flag():
    g = _GuiRoot()
    assert list(g) == []
    p = Panel(children=[])
    g.append(p)
    assert list(g) == [p]
    assert len(g) == 1
    assert g[0] is p
    assert g.pointer_over_gui is False
    g.pointer_over_gui = True
    assert g.pointer_over_gui is True


def test_value_display_calls_getter_each_frame_invocation():
    from manifoldx.gui import ValueDisplay
    calls = []
    def getter():
        calls.append(1)
        return "x"
    vd = ValueDisplay(getter=getter)
    vd.refresh()
    vd.refresh()
    assert len(calls) == 2
    assert vd.text == "x"


def test_value_display_min_width_locks_intrinsic_width():
    from manifoldx.gui import ValueDisplay
    vd = ValueDisplay(getter=lambda: "hello", min_width=120)
    vd.refresh()
    w, _ = vd.intrinsic_size()
    assert w >= 120


def test_value_display_intrinsic_size_does_not_change_when_string_same_length_fits_min_width():
    from manifoldx.gui import ValueDisplay
    counter = {"n": 0.0}
    def getter():
        counter["n"] += 1.0
        return f"fps: {counter['n']:.0f}"  # 1- to 3-digit fluctuation
    vd = ValueDisplay(getter=getter, min_width=200)
    vd.refresh()
    size1 = vd.intrinsic_size()
    vd.refresh()
    size2 = vd.intrinsic_size()
    # min_width clamps the width so it doesn't oscillate.
    assert size1 == size2
