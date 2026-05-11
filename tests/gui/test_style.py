"""Tests for manifoldx.gui.style — theme, named classes, overrides."""

import pytest

from manifoldx.gui import style


def setup_function(_):
    style.reset()  # Clear theme + named classes between tests.


def test_default_theme_resolves_when_no_class_or_overrides():
    resolved = style.resolve(class_name=None, overrides=None)
    # Defaults from the design doc style table.
    assert resolved["bg"] == "#222"
    assert resolved["fg"] == "#ddd"
    assert resolved["font_size"] == 12
    assert resolved["padding"] == 4
    assert resolved["gap"] == 4
    assert resolved["border"] == 0
    assert resolved["radius"] == 0
    assert resolved["direction"] == "v"


def test_set_theme_overrides_defaults():
    style.set_theme({"bg": "#111", "font_size": 16})
    resolved = style.resolve(class_name=None, overrides=None)
    assert resolved["bg"] == "#111"
    assert resolved["font_size"] == 16
    # Unchanged defaults still present.
    assert resolved["fg"] == "#ddd"


def test_define_class_and_resolve_merges_theme_under_class():
    style.set_theme({"bg": "#222", "fg": "#ddd"})
    style.define("hud", {"bg": "#1a1a1aD0", "padding": 8})
    resolved = style.resolve(class_name="hud", overrides=None)
    assert resolved["bg"] == "#1a1a1aD0"
    assert resolved["padding"] == 8
    # Theme fg still present.
    assert resolved["fg"] == "#ddd"


def test_overrides_take_precedence_over_class_and_theme():
    style.set_theme({"radius": 0})
    style.define("danger", {"radius": 3})
    resolved = style.resolve(class_name="danger", overrides={"radius": 8})
    assert resolved["radius"] == 8


def test_resolve_unknown_class_raises():
    with pytest.raises(KeyError):
        style.resolve(class_name="does_not_exist", overrides=None)


def test_color_parser_short_hex():
    assert style.parse_color("#abc") == (
        pytest.approx(0xaa / 255),
        pytest.approx(0xbb / 255),
        pytest.approx(0xcc / 255),
        pytest.approx(1.0),
    )


def test_color_parser_full_hex():
    assert style.parse_color("#112233") == (
        pytest.approx(0x11 / 255),
        pytest.approx(0x22 / 255),
        pytest.approx(0x33 / 255),
        pytest.approx(1.0),
    )


def test_color_parser_hex_with_alpha():
    assert style.parse_color("#11223344") == (
        pytest.approx(0x11 / 255),
        pytest.approx(0x22 / 255),
        pytest.approx(0x33 / 255),
        pytest.approx(0x44 / 255),
    )


def test_color_parser_rejects_garbage():
    with pytest.raises(ValueError):
        style.parse_color("not a color")
    with pytest.raises(ValueError):
        style.parse_color("#xyzxyz")


def test_padding_parser_int():
    assert style.parse_padding(8) == (8, 8, 8, 8)


def test_padding_parser_single_value_string():
    assert style.parse_padding("8") == (8, 8, 8, 8)


def test_padding_parser_two_values_vertical_horizontal():
    # CSS shorthand: "v h" -> top=bottom=v, right=left=h.
    assert style.parse_padding("6 12") == (6, 12, 6, 12)


def test_padding_parser_four_values_top_right_bottom_left():
    assert style.parse_padding("1 2 3 4") == (1, 2, 3, 4)


def test_padding_parser_rejects_three_values():
    with pytest.raises(ValueError):
        style.parse_padding("1 2 3")
