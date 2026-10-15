"""The interactive canvas's frame-rate cap, tested against a fake GLFW canvas."""

import sys
from types import ModuleType, SimpleNamespace

from manifoldx import Engine
from manifoldx.backends import get_desktop_canvas


def _fake_glfw_canvas(monkeypatch):
    made = []

    class Canvas:
        def __init__(self, **kw):
            made.append(kw)

    mod = ModuleType("rendercanvas.glfw")
    mod.GlfwRenderCanvas = Canvas
    monkeypatch.setitem(sys.modules, "rendercanvas.glfw", mod)
    return made


def test_the_desktop_canvas_is_not_held_to_rendercanvas_s_30_fps(monkeypatch):
    made = _fake_glfw_canvas(monkeypatch)
    get_desktop_canvas(width=64, height=36, fullscreen=False, title="t")
    assert made[0]["max_fps"] > 60


def test_the_engine_passes_its_max_fps_to_the_canvas(monkeypatch):
    made = _fake_glfw_canvas(monkeypatch)
    e = Engine("t", width=64, height=36, max_fps=48)
    monkeypatch.setattr(e, "_init_canvas", lambda canvas: (_ for _ in ()).throw(SystemExit))
    try:
        e.run()
    except SystemExit:
        pass
    assert made[0]["max_fps"] == 48
