"""Tests for manifoldx.gui.hit_test — topmost interactive widget under a point."""

from manifoldx.gui import Panel, Text, style
from manifoldx.gui.hit_test import hit_test
from manifoldx.gui.layout import LayoutBox, compute_layout


def setup_function(_):
    style.reset()


def test_hit_test_returns_none_when_no_panels():
    assert hit_test([], 50.0, 50.0, viewport=LayoutBox(0, 0, 100, 100)) is None


def test_hit_test_returns_none_when_point_outside_all_panels():
    p = Panel(children=[], anchor="top-left", offset=(0, 0),
              style_overrides={"width": 30, "height": 30})
    assert hit_test([p], 200.0, 200.0, viewport=LayoutBox(0, 0, 256, 256)) is None


def test_hit_test_non_interactive_widgets_dont_consume_hits():
    p = Panel(children=[Text("hi")], anchor="top-left", offset=(0, 0),
              style_overrides={"width": 100, "height": 30, "padding": 0})
    assert hit_test([p], 10.0, 10.0, viewport=LayoutBox(0, 0, 256, 256)) is None


def test_hit_test_later_panel_topmost():
    from manifoldx.gui.widgets import Widget

    class _Marker(Widget):
        def __init__(self, name):
            super().__init__()
            self._name = name
        def intrinsic_size(self):
            return (10.0, 10.0)
        @property
        def _is_gui_interactive(self):
            return True

    a, b = _Marker("a"), _Marker("b")
    p1 = Panel(children=[a], anchor="top-left", offset=(0, 0),
               style_overrides={"width": 50, "height": 50, "padding": 0})
    p2 = Panel(children=[b], anchor="top-left", offset=(0, 0),
               style_overrides={"width": 50, "height": 50, "padding": 0})
    hit = hit_test([p1, p2], 10.0, 10.0, viewport=LayoutBox(0, 0, 256, 256))
    assert hit is not None
    widget, box = hit
    assert widget is b
