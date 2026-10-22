"""Render layers: a camera draws only entities on its layers, and skips the rest early."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.camera import Camera
from manifoldx.components import Layer, Material, Mesh, Transform
from manifoldx.resources import PhongMaterial, sphere


def _offscreen(w=96, h=96):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=w, height=h)
    eng._init_canvas(canvas)
    eng._running = True
    return eng, canvas


def _read(eng, handle):
    w, h = handle.size
    data = eng._device.queue.read_texture(
        {"texture": handle.texture, "origin": (0, 0, 0)},
        {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    return np.frombuffer(data, np.uint8).reshape(h, w, 4).astype(int)


def test_each_camera_draws_only_its_layers():
    eng, canvas = _offscreen()
    # red ball on layer 0 and green ball on layer 1, both in front of both cameras
    eng.spawn(Mesh(sphere(1.0, 16)), Material(PhongMaterial("#ff0000")), Transform(pos=(0, 0, 0)))
    eng.spawn(Mesh(sphere(1.0, 16)), Material(PhongMaterial("#00ff00")),
              Transform(pos=(0, 0, 1.5)), Layer(layer=1))
    eng.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    eng.camera.layers = {0}
    cam = Camera()
    cam.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    cam.layers = {1}
    view = eng.render_to_texture(cam, (64, 64))
    eng._draw_frame()
    main = np.asarray(canvas.draw()).astype(int)
    v = _read(eng, view)
    assert main[48, 48, 0] > 100 and main[48, 48, 1] < 40  # main: red only, green skipped
    assert v[32, 32, 1] > 100 and v[32, 32, 0] < 40  # view: green only


def test_default_layers_draw_everything():
    eng, canvas = _offscreen()
    eng.spawn(Mesh(sphere(1.0, 16)), Material(PhongMaterial("#00ff00")),
              Transform(pos=(0, 0, 0)), Layer(layer=3))
    eng.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    eng._draw_frame()
    frame = np.asarray(canvas.draw()).astype(int)
    assert frame[48, 48, 1] > 100


def test_a_view_only_culls_its_own_layer(monkeypatch):
    eng, canvas = _offscreen()
    rng = np.random.default_rng(0)
    eng.spawn(Mesh(sphere(0.3, 8)), Material(PhongMaterial("#4dabf7")),
              Transform(pos=rng.uniform(-20, 20, (500, 3))), n=500)
    eng.spawn(Mesh(sphere(1.0, 16)), Material(PhongMaterial("#ffd43b")),
              Transform(pos=(1000, 0, 0)), Layer(layer=1))
    cam = Camera()
    cam.set_pose(position=(1000, 0, 4), target=(1000, 0, 0))
    cam.layers = {1}
    eng.render_to_texture(cam, (32, 32))
    seen = []
    real = eng._render_pipeline._in_frustum
    monkeypatch.setattr(eng._render_pipeline, "_in_frustum",
                        lambda e, g, *a: seen.append((e.w, len(g))) or real(e, g, *a))
    eng._draw_frame()
    assert (32, 1) in seen  # the view tested one entity, not 501
