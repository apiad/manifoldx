"""The Image widget: sizes, paint, pixel size under scale and density, and a GPU draw."""

import numpy as np
import pytest

pytest.importorskip("thorvg_python")

import manifoldx as mx  # noqa: E402
from manifoldx import svg  # noqa: E402
from manifoldx.gui import Image, Panel, style  # noqa: E402
from manifoldx.gui.layout import LayoutBox  # noqa: E402
from manifoldx.render.passes.gui import image_pixel_box, layout_gui  # noqa: E402

# 20 x 10: red on the left half, transparent on the right.
ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10">'
    '<rect width="10" height="10" fill="#ff0000"/></svg>'
)
WHITE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
    '<rect width="10" height="10" fill="#ffffff"/></svg>'
)


def setup_function(_):
    style.reset()
    svg.clear_cache()


def test_intrinsic_size_keeps_aspect():
    assert Image(ICON).intrinsic_size() == (20.0, 10.0)
    assert Image(ICON, style_overrides={"width": 60}).intrinsic_size() == (60.0, 30.0)
    assert Image(ICON, style_overrides={"height": 40}).intrinsic_size() == (80.0, 40.0)
    assert Image(ICON, style_overrides={"width": 7, "height": 9}).intrinsic_size() == (7.0, 9.0)


def test_swapping_the_svg_updates_size():
    img = Image(ICON)
    img.svg = WHITE
    assert img.intrinsic_size() == (10.0, 10.0)


def _engine(w=200, h=100):
    return mx.Engine("t", width=w, height=h)


def test_paint_centres_the_art_in_a_stretched_box():
    eng = _engine()
    eng.gui.append(Panel([Image(ICON, style_overrides={"width": 40})],
                         style_overrides={"width": 100, "height": 50, "padding": 0}))
    (op,) = layout_gui(eng).image_ops
    # the cross axis stretches the leaf to 100 wide; the art stays 40 x 20, centred
    assert (op.box.x, op.box.y, op.box.w, op.box.h) == (30.0, 0.0, 40.0, 20.0)


def test_tint_and_opacity_reach_the_op():
    eng = _engine()
    eng.gui.append(Panel([Image(WHITE, style_overrides={"tint": "#00ff00", "opacity": 0.5})]))
    (op,) = layout_gui(eng).image_ops
    assert op.tint == (0.0, 1.0, 0.0, 0.5)


def test_pixel_size_follows_gui_scale_and_density():
    eng = _engine()
    eng.gui.append(Panel([Image(ICON, style_overrides={"width": 48})]))
    (op,) = layout_gui(eng).image_ops
    assert image_pixel_box(op.box, 1.0)[1:] == (48, 24)
    assert image_pixel_box(op.box, 2.0)[1:] == (96, 48)  # HiDPI or a 2x still
    eng.gui.scale = 1.5
    (op,) = layout_gui(eng).image_ops
    assert image_pixel_box(op.box, 1.0)[1:] == (72, 36)


def test_pixel_box_snaps_to_target_pixels():
    box, pw, ph = image_pixel_box(LayoutBox(10.3, 4.6, 20.2, 9.8), 2.0)
    assert (pw, ph) == (40, 20)
    assert (box.x * 2, box.y * 2, box.w * 2, box.h * 2) == (21.0, 9.0, 40.0, 20.0)


def _offscreen(w=128, h=128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=w, height=h)
    eng._init_canvas(canvas)
    eng._running = True
    return eng, canvas


def test_gpu_draws_the_svg_over_the_panel_with_alpha():
    eng, canvas = _offscreen()
    eng.gui.append(Panel([Image(ICON, style_overrides={"width": 80})],
                         offset=(10, 10),
                         style_overrides={"bg": "#0000ffff", "padding": 0,
                                          "width": 80, "height": 40}))
    eng._draw_frame()
    frame = np.asarray(canvas.draw())
    assert frame[30, 30, :3].tolist() == [255, 0, 0]  # red half of the art
    assert frame[30, 70, 2] > 200 and frame[30, 70, 0] < 30  # transparent half shows the panel


def test_gpu_tint_matches_a_white_icon():
    eng, canvas = _offscreen()
    eng.gui.append(Panel([Image(WHITE, style_overrides={"width": 40, "tint": "#00ff00"})],
                         offset=(10, 10), style_overrides={"padding": 0, "width": 40}))
    eng._draw_frame()
    frame = np.asarray(canvas.draw())
    assert frame[30, 30, :3].tolist() == [0, 255, 0]


def test_gpu_rerasterizes_at_the_new_scale():
    eng, canvas = _offscreen()
    eng.gui.append(Panel([Image(ICON, style_overrides={"width": 40})], style_overrides={"padding": 0}))
    eng._draw_frame()
    canvas.draw()
    eng.gui.scale = 2.0
    for _ in range(2):  # the frame of the change draws a bucketed raster, the next the exact one
        eng._draw_frame()
        canvas.draw()
    sizes = {(w, h) for (_d, w, h) in svg._cache._items}
    assert (80, 40) in sizes and (40, 20) in sizes


def test_paint_runs_group_same_kind_ops():
    from manifoldx.render.passes.gui import paint_runs

    order = [("rect", 0), ("rect", 1), ("image", 0), ("text", 0), ("rect", 2), ("image", 1)]
    assert paint_runs(order) == [
        ("rect", 0, 2), ("image", 0, 1), ("text", 0, 1), ("rect", 2, 1), ("image", 1, 1)
    ]


def test_later_root_panel_covers_earlier_images():
    eng, canvas = _offscreen()
    eng.gui.append(Panel([Image(WHITE, style_overrides={"width": 40, "tint": "#00ff00"})],
                         offset=(10, 10), style_overrides={"padding": 0, "width": 40}))
    eng.gui.append(Panel([Image(WHITE, style_overrides={"width": 20, "tint": "#ff0000"})],
                         offset=(30, 30),
                         style_overrides={"padding": 0, "width": 30, "height": 30, "bg": "#0000ffff"}))
    eng._draw_frame()
    frame = np.asarray(canvas.draw())
    assert frame[15, 15, :3].tolist() == [0, 255, 0]  # first icon, uncovered
    # the second panel spans 30..60; its icon x 35..55, y 30..50; the first icon 10..50
    assert frame[40, 32, 2] > 200 and frame[40, 32, 1] < 30  # second panel over the first icon
    assert frame[40, 45, :3].tolist() == [255, 0, 0]  # second icon, drawn after its panel


def test_bucket_size_steps_and_settle():
    from manifoldx.render.passes.gui import bucket_size, settle_size

    for n in (17, 48, 100, 333):
        b, _ = bucket_size(n, n)
        assert n <= b <= n * 2 ** (1 / 8) + 1
    assert bucket_size(50, 50) == bucket_size(51, 51)  # neighbours share a raster
    assert settle_size(None, 50, 50, False) == (50, 50)  # first frame: exact
    assert settle_size((50, 50), 50, 50, False) == (50, 50)  # settled: exact
    assert settle_size((48, 48), 50, 50, False) == bucket_size(50, 50)  # animating
    assert settle_size((48, 48), 50, 50, True) == (50, 50)  # exact already cached


def test_gpu_pulse_reuses_bucketed_rasters():
    eng, canvas = _offscreen()
    img = Image(ICON, style_overrides={"width": 40})
    eng.gui.append(Panel([img], style_overrides={"padding": 0}))
    widths = [40 + 8 * abs(((i % 10) - 5) / 5) for i in range(40)]  # 40..48..40, four cycles
    for w in widths:
        img.style_overrides["width"] = w
        eng._draw_frame()
        canvas.draw()
    assert len(svg._cache) <= 6
