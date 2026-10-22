"""engine.gui.scale / design_height: UI pixels times scale = screen pixels."""

from types import SimpleNamespace

from manifoldx.gui import Button, Panel, Text, style
from manifoldx.gui.layout import LayoutBox, compute_layout
from manifoldx.gui.painter import Painter, paint
from manifoldx.gui.widgets import _GuiRoot, _measure_text
from manifoldx.input import PointerEvent
from manifoldx.render.passes.gui import _anchored, layout_gui


def setup_function(_):
    style.reset()


def _engine(w=800, h=600, **gui_kw):
    gui = _GuiRoot()
    for k, v in gui_kw.items():
        setattr(gui, k, v)
    return SimpleNamespace(w=w, h=h, gui=gui)


def _panel(**kw):
    overrides = {"width": 100, "height": 50, "padding": 4, "radius": 6, "border": 1}
    return Panel(children=[Text("hi")], style_overrides=overrides, **kw)


def test_effective_scale_defaults_to_one():
    assert _GuiRoot().effective_scale(600) == 1.0


def test_design_height_overrides_scale():
    gui = _GuiRoot()
    gui.scale = 3.0
    gui.design_height = 540
    assert gui.effective_scale(1080) == 2.0


def test_scale_two_doubles_rect_and_text():
    eng = _engine(scale=2.0)
    eng.gui.append(_panel(offset=(10, 20)))
    p = layout_gui(eng)
    rect = p.rect_ops[0]
    assert rect.box == LayoutBox(20.0, 40.0, 200.0, 100.0)
    assert rect.radius == 12.0
    assert rect.border == 2.0

    eng1 = _engine()
    eng1.gui.append(_panel(offset=(10, 20)))
    t1 = layout_gui(eng1).text_ops[0]
    t2 = p.text_ops[0]
    assert t2.font_size == 2 * t1.font_size
    assert (t2.box.x, t2.box.y) == (2 * t1.box.x, 2 * t1.box.y)
    # The glyph box is measured at the scaled font size, so it rasterizes 1:1.
    assert (t2.box.w, t2.box.h) == _measure_text("hi", t2.font_size)


def test_panel_at_bottom_right_of_ui_viewport_touches_screen_corner():
    eng = _engine(w=1920, h=1080, design_height=540)  # scale 2
    eng.gui.append(_panel(offset=(1920 / 2 - 100, 540 - 50)))
    box = layout_gui(eng).rect_ops[0].box
    assert box.x + box.w == 1920.0
    assert box.y + box.h == 1080.0


def test_scale_one_matches_unscaled_path():
    eng = _engine()
    eng.gui.exact_colors = False  # compare geometry only, not the colour decode
    eng.gui.append(_panel(offset=(7, 9)))
    eng.gui.append(Panel(children=[Text("full")], offset=(3, 3)))
    viewport = LayoutBox(0.0, 0.0, 800.0, 600.0)
    ref = Painter()
    for panel in eng.gui:
        spec = panel.build_layout_spec()
        paint(panel, spec, compute_layout(spec, viewport=_anchored(panel, viewport)), ref)
    got = layout_gui(eng)
    assert got.rect_ops == ref.rect_ops
    assert got.text_ops == ref.text_ops


def test_hit_test_divides_pointer_by_scale():
    import manifoldx as mx

    engine = mx.Engine("t", width=800, height=600)
    engine.gui.scale = 2.0
    btn = Button(name="go", label="go")
    engine.gui.append(
        Panel(children=[btn], offset=(100, 100),
              style_overrides={"width": 100, "height": 40, "padding": 0})
    )

    def ev(x, y, phase):
        return PointerEvent(x=x, y=y, dx=0.0, dy=0.0, button=1, buttons=(1,),
                            modifiers=(), phase=phase)

    # (250, 220) on screen is (125, 110) in UI pixels: inside the button,
    # which unscaled would be outside the panel's (100..200, 100..140) slot.
    engine._gui_bridge._on_pointer(ev(250, 220, "down"), "down")
    assert engine.gui.pointer_over_gui is True
    assert btn.was_clicked()

    # (150, 120) is inside the unscaled slot but maps to (75, 60): a miss.
    engine.gui.pointer_over_gui = False
    engine._gui_bridge._on_pointer(ev(150, 120, "down"), "down")
    assert engine.gui.pointer_over_gui is False
    assert not btn.was_clicked()



def test_loading_panel_stays_centred_under_scale():
    import manifoldx as mx

    engine = mx.Engine("t", width=800, height=600)
    engine.gui.scale = 2.0
    with engine.loading("model"):
        engine._update_loading_overlay()
        box = layout_gui(engine).rect_ops[0].box
    assert (box.x + box.w / 2, box.y + box.h / 2) == (400.0, 300.0)


def test_text_too_wide_for_an_atlas_tile_is_stretched_not_cut():
    label = "Press + / - to scale the UI"  # ~195 px at 12 pt, ~390 px at 24 pt
    eng1 = _engine()
    eng1.gui.append(Panel(children=[Text(label)], style_overrides={"font_size": 12}))
    t1 = layout_gui(eng1).text_ops[0]
    eng2 = _engine(scale=2.0)
    eng2.gui.append(Panel(children=[Text(label)], style_overrides={"font_size": 12}))
    t2 = layout_gui(eng2).text_ops[0]
    assert t2.font_size == t1.font_size
    assert t2.box == LayoutBox(2 * t1.box.x, 2 * t1.box.y, 2 * t1.box.w, 2 * t1.box.h)
