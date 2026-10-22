"""render_to_texture views, Image over a TextureHandle with a round clip, and Text in a font file."""

from importlib.resources import files

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.camera import Camera
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gui import Image, Panel, Text, style
from manifoldx.render.passes.gui import layout_gui
from manifoldx.resources import PhongMaterial, sphere

FONT = str(files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf"))


def setup_function(_):
    style.reset()


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


def _stage(eng):
    """A red ball far off at x=1000, and a camera looking at it."""
    eng.spawn(Mesh(sphere(1.0, 16)), Material(PhongMaterial("#ff0000")), Transform(pos=(1000, 0, 0)))
    cam = Camera()
    cam.set_pose(position=(1000, 0, 4), target=(1000, 0, 0))
    return cam


def _read(eng, handle):
    w, h = handle.size
    data = eng._device.queue.read_texture(
        {"texture": handle.texture, "origin": (0, 0, 0)},
        {"bytes_per_row": w * 4, "rows_per_image": h},
        (w, h, 1),
    )
    return np.frombuffer(data, np.uint8).reshape(h, w, 4)


def test_view_sees_what_the_main_camera_does_not():
    eng, canvas = _offscreen()
    handle = eng.render_to_texture(_stage(eng), (64, 64))
    assert handle.size == (64, 64)
    eng._draw_frame()
    main = np.asarray(canvas.draw())
    view = _read(eng, handle)
    assert view[32, 32, 0] > 150 and view[32, 32, 1] < 30  # the ball
    assert main[64, 64, 0] < 150  # the main camera looks at the origin: background


def test_still_view_renders_once_and_on_request(monkeypatch):
    eng, canvas = _offscreen()
    handle = eng.render_to_texture(_stage(eng), (32, 32), live=False)
    calls = []
    real = eng._render_pipeline._render_scene_passes
    monkeypatch.setattr(eng._render_pipeline, "_render_scene_passes",
                        lambda e, rp: calls.append(e.w) or real(e, rp))
    for _ in range(3):
        eng._draw_frame()
        canvas.draw()
    assert calls.count(32) == 1  # once, then nothing
    eng.refresh_texture(handle)
    eng._draw_frame()
    canvas.draw()
    assert calls.count(32) == 2
    eng.set_texture_live(handle, True)
    eng._draw_frame()
    eng._draw_frame()
    assert calls.count(32) == 4
    eng.drop_texture(handle)
    eng._draw_frame()
    assert calls.count(32) == 4


def test_image_draws_a_view_with_a_round_clip():
    eng, canvas = _offscreen()
    handle = eng.render_to_texture(_stage(eng), (64, 64), background=(0.0, 1.0, 0.0, 1.0))
    eng.gui.append(Panel(
        [Image(handle, style_overrides={"radius": 32})],
        offset=(10, 10),
        style_overrides={"padding": 0, "width": 64, "height": 64, "bg": "#0000ffff"},
    ))
    eng._draw_frame()
    frame = np.asarray(canvas.draw())
    assert frame[42, 42, 0] > 150 and frame[42, 42, 2] < 60  # the ball, through the GUI
    assert frame[20, 42, 1] > 200  # the view's green background inside the circle
    assert frame[12, 12, 2] > 200 and frame[12, 12, 1] < 60  # corner: clipped, the blue panel


def test_text_with_a_font_takes_the_image_path_and_any_width():
    eng = mx.Engine("t", width=1600, height=200)
    line = "¡Hola! ¿Qué tal? Esta línea es más ancha que una tesela del atlas."
    eng.gui.append(Panel([Text(line, style_overrides={"font": FONT, "font_size": 40})]))
    painter = layout_gui(eng)
    assert painter.text_ops == []
    (op,) = painter.image_ops
    assert op.box.w > 1000  # far past the label atlas' 256-pixel tile
    rgba = op.raster(int(op.box.w), int(op.box.h))
    assert rgba.shape == (int(op.box.h), int(op.box.w), 4)
    assert rgba[..., 3].max() == 255 and rgba[:, -200:, 3].max() > 0  # the line's end is drawn


def test_gpu_draws_font_text_in_fg():
    eng, canvas = _offscreen(w=200, h=80)
    eng.gui.append(Panel(
        [Text("MMMM", style_overrides={"font": FONT, "font_size": 40, "fg": "#ff0000"})],
        style_overrides={"padding": 0, "bg": "#000000ff", "width": 200, "height": 80},
    ))
    eng._draw_frame()
    frame = np.asarray(canvas.draw()).astype(int)
    red = (frame[..., 0] > 200) & (frame[..., 1] < 40)
    assert red.sum() > 200


def test_view_background_is_srgb_and_takes_rgb_or_hex():
    eng, canvas = _offscreen()
    cam = _stage(eng)
    a = eng.render_to_texture(cam, (16, 16), background=(0.6, 0.8, 0.5))
    b = eng.render_to_texture(cam, (16, 16), background="#99cc80")
    with pytest.raises(ValueError):
        eng.render_to_texture(cam, (16, 16), background=(0.6, 0.8))
    eng._draw_frame()
    for h in (a, b):
        assert np.abs(_read(eng, h)[1, 1, :3].astype(int) - [153, 204, 128]).max() <= 2


def test_view_skips_world_fog_unless_asked():
    from manifoldx.resources import PointLight, StandardMaterial

    eng, canvas = _offscreen()
    eng.spawn(Mesh(sphere(8.0, 16)), Material(StandardMaterial(color="#ff0000")),
              Transform(pos=(1000, 0, 0)))  # fog applies to StandardMaterial
    eng.add_light(PointLight(position=(1000, 20, 40), color="#ffffff", intensity=4000.0))
    cam = Camera()
    cam.set_pose(position=(1000, 0, 60), target=(1000, 0, 0))  # far, like a minimap camera
    eng.enable_fog(10.0, 40.0, color=(1.0, 1.0, 1.0))
    clear = eng.render_to_texture(cam, (64, 64), background=(0, 0, 0, 1))
    foggy = eng.render_to_texture(cam, (64, 64), background=(0, 0, 0, 1), fog=True)
    eng._draw_frame()
    c, f = _read(eng, clear)[32, 32], _read(eng, foggy)[32, 32]
    assert c[0] > 120 and c[1] < 40  # the red ball as it is
    assert f[1] > 200  # fully fogged to white
