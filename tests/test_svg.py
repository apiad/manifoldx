"""SVG rasterizer: source kinds, sizes, colour, and the cache."""

import numpy as np
import pytest

pytest.importorskip("thorvg_python")

from manifoldx import svg  # noqa: E402

RED_SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10">'
    '<rect width="10" height="10" fill="#ff0000"/></svg>'
)


def setup_function(_):
    svg.clear_cache()


def test_sources_give_the_same_pixels(tmp_path):
    path = tmp_path / "icon.svg"
    path.write_text(RED_SQUARE)
    a = svg.rasterize_svg(RED_SQUARE, 40, 20)
    for source in (RED_SQUARE.encode(), path, str(path)):
        assert np.array_equal(svg.rasterize_svg(source, 40, 20), a)


def test_size_and_colour():
    rgba = svg.rasterize_svg(RED_SQUARE, 40, 20)
    assert rgba.shape == (20, 40, 4) and rgba.dtype == np.uint8
    assert rgba[10, 10].tolist() == [255, 0, 0, 255]  # left half: the square
    assert rgba[10, 30, 3] == 0  # right half: transparent


def test_straight_alpha_on_a_translucent_fill():
    half = RED_SQUARE.replace('fill="#ff0000"', 'fill="#ff0000" fill-opacity="0.5"')
    r, g, b, a = svg.rasterize_svg(half, 40, 20)[10, 10].tolist()
    assert 120 <= a <= 135 and r >= 250 and g == 0 and b == 0


def test_svg_size_reads_width_and_height():
    assert svg.svg_size(RED_SQUARE) == (20.0, 10.0)


def test_cache_hits_and_new_sizes(monkeypatch):
    calls = []
    real = svg._render
    monkeypatch.setattr(svg, "_render", lambda d, w, h: calls.append((w, h)) or real(d, w, h))
    svg.rasterize_svg(RED_SQUARE, 48, 24)
    svg.rasterize_svg(RED_SQUARE.encode(), 48, 24)
    assert calls == [(48, 24)]
    svg.rasterize_svg(RED_SQUARE, 96, 48)
    assert calls == [(48, 24), (96, 48)]


def test_cache_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(svg._cache, "budget", 2 * 32 * 32 * 4)
    svg.rasterize_svg(RED_SQUARE, 32, 32)
    svg.rasterize_svg(RED_SQUARE, 32, 31)
    svg.rasterize_svg(RED_SQUARE, 32, 32)  # touch: 32x31 is now the oldest
    svg.rasterize_svg(RED_SQUARE, 31, 32)
    keys = {(w, h) for (_d, w, h) in svg._cache._items}
    assert keys == {(32, 32), (31, 32)}


def test_text_renders_with_the_preloaded_font():
    text = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<text x="4" y="40" font-family="DejaVuSansMono" font-size="24" fill="#000">Ab</text>'
        "</svg>"
    )
    assert svg.rasterize_svg(text, 64, 64)[..., 3].sum() > 0


def test_bad_size_and_bad_svg_raise():
    with pytest.raises(ValueError):
        svg.rasterize_svg(RED_SQUARE, 0, 10)
    with pytest.raises(ValueError):
        svg.rasterize_svg("<not svg", 10, 10)
