"""Fullscreen monitor selection, tested against a fake monitor list (no GLFW)."""

import ctypes
import sys
from types import SimpleNamespace

import pytest

from manifoldx.backends import _resolve_monitor, list_monitors

NAMES = ["Built-in Display", "HDMI-1 Living Room TV", "DP-2 Dell U2720Q"]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MANIFOLDX_MONITOR", raising=False)


def test_none_falls_back_to_primary():
    assert _resolve_monitor(None, NAMES, primary=2) == 2


def test_int_index():
    assert _resolve_monitor(1, NAMES, primary=0) == 1


def test_name_substring_is_case_insensitive():
    assert _resolve_monitor("hdmi", NAMES, primary=0) == 1
    assert _resolve_monitor("u2720", NAMES, primary=0) == 2


def test_env_var_used_when_argument_is_none(monkeypatch):
    monkeypatch.setenv("MANIFOLDX_MONITOR", "living room")
    assert _resolve_monitor(None, NAMES, primary=0) == 1
    monkeypatch.setenv("MANIFOLDX_MONITOR", "2")
    assert _resolve_monitor(None, NAMES, primary=0) == 2


def test_argument_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("MANIFOLDX_MONITOR", "HDMI")
    assert _resolve_monitor(0, NAMES, primary=2) == 0


def test_no_match_lists_the_monitors():
    with pytest.raises(ValueError) as exc:
        _resolve_monitor("projector", NAMES, primary=0)
    msg = str(exc.value)
    assert "projector" in msg
    for i, name in enumerate(NAMES):
        assert f"{i}: {name}" in msg


def test_index_out_of_range_lists_the_monitors():
    with pytest.raises(ValueError, match="1: HDMI-1 Living Room TV"):
        _resolve_monitor(5, NAMES, primary=0)


def test_list_monitors_reads_glfw(monkeypatch):
    # GLFW hands out ctypes pointers, and two pointer objects to the same
    # address compare unequal, so the primary is a separate object here.
    ptr = ctypes.POINTER(ctypes.c_int)

    def addr(m):
        return ctypes.cast(m, ctypes.c_void_p).value

    mons = [ctypes.cast(0x1000, ptr), ctypes.cast(0x2000, ptr)]
    names = {0x1000: b"Built-in", 0x2000: b"HDMI-1 TV"}
    modes = {
        0x1000: SimpleNamespace(size=(2560, 1600), refresh_rate=60),
        0x2000: SimpleNamespace(size=(3840, 2160), refresh_rate=30),
    }
    fake = SimpleNamespace(
        init=lambda: True,
        get_monitors=lambda: mons,
        get_primary_monitor=lambda: ctypes.cast(0x1000, ptr),
        get_monitor_name=lambda m: names[addr(m)],
        get_video_mode=lambda m: modes[addr(m)],
    )
    monkeypatch.setitem(sys.modules, "glfw", fake)
    assert list_monitors() == [
        (0, "Built-in", (2560, 1600), 60, True),
        (1, "HDMI-1 TV", (3840, 2160), 30, False),
    ]


def test_fullscreen_canvas_goes_to_the_chosen_monitor_at_its_mode(monkeypatch):
    from manifoldx.backends import get_desktop_canvas

    ptr = ctypes.POINTER(ctypes.c_int)

    def addr(m):
        return ctypes.cast(m, ctypes.c_void_p).value

    mons = [ctypes.cast(0x1000, ptr), ctypes.cast(0x2000, ptr)]
    names = {0x1000: b"Built-in", 0x2000: b"HDMI-1 TV"}
    modes = {
        0x1000: SimpleNamespace(size=(2560, 1600), refresh_rate=60),
        0x2000: SimpleNamespace(size=(3840, 2160), refresh_rate=30),
    }
    calls = []
    fake_glfw = SimpleNamespace(
        get_monitors=lambda: mons,
        get_primary_monitor=lambda: ctypes.cast(0x1000, ptr),
        get_monitor_name=lambda m: names[addr(m)],
        get_video_mode=lambda m: modes[addr(m)],
        set_window_monitor=lambda win, m, x, y, w, h, hz: calls.append((addr(m), w, h, hz)),
        AUTO_ICONIFY=0x00020006,
        set_window_attrib=lambda *a: None,
    )

    class FakeCanvas:
        def __init__(self, size, title):
            self._window = object()

    monkeypatch.setitem(sys.modules, "glfw", fake_glfw)
    monkeypatch.setitem(sys.modules, "rendercanvas.glfw", SimpleNamespace(GlfwRenderCanvas=FakeCanvas))

    get_desktop_canvas(width=800, height=600, fullscreen=True, title="t", monitor="hdmi")
    get_desktop_canvas(width=800, height=600, fullscreen=True, title="t")
    assert calls == [(0x2000, 3840, 2160, 30), (0x1000, 2560, 1600, 60)]


def test_run_passes_monitor_to_the_desktop_canvas(monkeypatch):
    import manifoldx as mx
    import manifoldx.backends as backends

    seen = {}

    def fake_canvas(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(backends, "get_desktop_canvas", fake_canvas)
    engine = mx.Engine("t", fullscreen=True, monitor="HDMI")
    with pytest.raises(RuntimeError, match="stop"):
        engine.run()
    assert seen["monitor"] == "HDMI" and seen["fullscreen"] is True


def test_fullscreen_window_does_not_iconify_on_focus_loss(monkeypatch):
    # GLFW minimizes a fullscreen window when it loses focus (AUTO_ICONIFY), so
    # clicking another monitor would drop the app off its screen.
    from manifoldx.backends import get_desktop_canvas

    ptr = ctypes.POINTER(ctypes.c_int)
    mon = ctypes.cast(0x1000, ptr)
    attribs = []
    fake_glfw = SimpleNamespace(
        AUTO_ICONIFY=0x00020006,
        get_monitors=lambda: [mon],
        get_primary_monitor=lambda: mon,
        get_monitor_name=lambda m: b"Built-in",
        get_video_mode=lambda m: SimpleNamespace(size=(1920, 1080), refresh_rate=60),
        set_window_monitor=lambda *a: None,
        set_window_attrib=lambda win, attrib, value: attribs.append((win, attrib, value)),
    )

    class FakeCanvas:
        def __init__(self, size, title):
            self._window = object()

    monkeypatch.setitem(sys.modules, "glfw", fake_glfw)
    monkeypatch.setitem(sys.modules, "rendercanvas.glfw", SimpleNamespace(GlfwRenderCanvas=FakeCanvas))

    canvas = get_desktop_canvas(width=800, height=600, fullscreen=True, title="t")
    assert attribs == [(canvas._window, fake_glfw.AUTO_ICONIFY, False)]
    attribs.clear()
    get_desktop_canvas(width=800, height=600, fullscreen=False, title="t")
    assert attribs == []
