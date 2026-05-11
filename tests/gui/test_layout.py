"""Tests for manifoldx.gui.layout — stack + flex algorithm."""

import pytest

from manifoldx.gui.layout import LayoutBox, compute_layout


def _leaf(width=None, height=None, flex=None):
    return {
        "direction": "v",
        "padding": (0, 0, 0, 0),
        "gap": 0,
        "width": width,
        "height": height,
        "flex": flex,
        "intrinsic": (10, 10),
        "children": [],
    }


def _container(direction="v", padding=(0, 0, 0, 0), gap=0, children=None):
    return {
        "direction": direction,
        "padding": padding,
        "gap": gap,
        "width": None,
        "height": None,
        "flex": None,
        "intrinsic": (0, 0),
        "children": children or [],
    }


def test_single_leaf_fills_container():
    root = _container(children=[_leaf(flex=1)])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 50))
    assert boxes[id(root)] == LayoutBox(0, 0, 100, 50)
    assert boxes[id(root["children"][0])] == LayoutBox(0, 0, 100, 50)


def test_vertical_stack_with_fixed_heights():
    a = _leaf(height=10)
    b = _leaf(height=20)
    root = _container(direction="v", children=[a, b])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    assert boxes[id(a)] == LayoutBox(0, 0, 100, 10)
    assert boxes[id(b)] == LayoutBox(0, 10, 100, 20)


def test_horizontal_stack_with_fixed_widths():
    a = _leaf(width=10)
    b = _leaf(width=20)
    root = _container(direction="h", children=[a, b])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 50))
    assert boxes[id(a)] == LayoutBox(0, 0, 10, 50)
    assert boxes[id(b)] == LayoutBox(10, 0, 20, 50)


def test_flex_distributes_remaining_axis_space():
    a = _leaf(height=10)             # fixed
    b = _leaf(flex=1)                # gets 1 share of the rest
    c = _leaf(flex=3)                # gets 3 shares of the rest
    root = _container(direction="v", children=[a, b, c])
    # viewport=100h, fixed=10, remaining=90, shares=4 → b=22.5, c=67.5
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    assert boxes[id(a)] == LayoutBox(0, 0, 100, 10)
    assert boxes[id(b)] == LayoutBox(0, 10, 100, pytest.approx(22.5))
    assert boxes[id(c)] == LayoutBox(0, pytest.approx(32.5), 100, pytest.approx(67.5))


def test_padding_shrinks_children_area():
    a = _leaf(flex=1)
    root = _container(padding=(2, 4, 6, 8), children=[a])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    # Children area: x=8, y=2, w=100-4-8=88, h=100-2-6=92.
    assert boxes[id(a)] == LayoutBox(8, 2, 88, 92)


def test_gap_separates_siblings_in_stack():
    a = _leaf(height=10)
    b = _leaf(height=10)
    root = _container(direction="v", gap=5, children=[a, b])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    assert boxes[id(a)].y == 0
    assert boxes[id(b)].y == 10 + 5  # gap inserted between siblings


def test_nested_panels_layout_independently():
    inner_leaf = _leaf(width=20, height=10)
    inner = _container(direction="h", padding=(1, 1, 1, 1), children=[inner_leaf])
    inner["flex"] = 1
    outer = _container(direction="v", padding=(5, 5, 5, 5), children=[inner])
    boxes = compute_layout(outer, viewport=LayoutBox(0, 0, 100, 100))
    # outer children area: x=5,y=5,w=90,h=90 → inner gets that whole box (flex=1).
    assert boxes[id(inner)] == LayoutBox(5, 5, 90, 90)
    # inner children area: x=6,y=6,w=88,h=88 → leaf has explicit width=20, fills cross-axis height.
    assert boxes[id(inner_leaf)] == LayoutBox(6, 6, 20, 88)


def test_intrinsic_size_used_when_no_explicit_size_or_flex():
    leaf = _leaf()
    leaf["intrinsic"] = (30, 8)
    root = _container(direction="v", children=[leaf])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 50))
    # On the cross axis the leaf still fills (no per-child cross-alignment in v1).
    # On the main axis it gets its intrinsic height.
    assert boxes[id(leaf)] == LayoutBox(0, 0, 100, 8)
