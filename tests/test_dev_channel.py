import json
import struct
import textwrap
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from manifoldx import cli, devchannel
from manifoldx.backends import get_offscreen_canvas
from manifoldx.hot import HotReloader

APP = """
import manifoldx as mx

engine = mx.Engine("channel", width=100, height=60)
engine.background_color = (1.0, 0.0, 0.0)
engine.state.setdefault("loads", 0)
engine.state.loads += 1
engine.state.where = {"place": "yard", "x": 2.5}

@engine.on("frame")
def frame(payload):
    engine.log("at the yard")

engine.run()
"""


def unpng(data):
    """(h, w, 3) uint8 from the PNG bytes devchannel.png writes (one IDAT, filter 0 on every row)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    at, idat = 8, b""
    while at < len(data):
        n, kind = struct.unpack(">I4s", data[at:at + 8])
        if kind == b"IDAT":
            idat += data[at + 8:at + 8 + n]
        at += 12 + n
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * 3)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(h, w, 3)


def test_png_round_trips_an_image():
    rgb = np.random.default_rng(1).integers(0, 256, (7, 13, 3), dtype=np.uint8)
    assert (unpng(devchannel.png(rgb)) == rgb).all()


@pytest.fixture
def hot(tmp_path):
    try:
        import wgpu
        wgpu.gpu.request_adapter_sync()
    except Exception as e:  # noqa: BLE001  no GPU and no software Vulkan
        pytest.skip(f"offscreen canvas unavailable: {e}")
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(APP))
    h = HotReloader(script=str(app), canvas_factory=lambda e: get_offscreen_canvas(e.w, e.h), poll=0)
    h.start()
    h.step()
    yield h
    h.channel.close()


def test_look_writes_the_presented_frame(hot, tmp_path):
    (tmp_path / ".manifoldx" / "requests" / "look").touch()
    hot.step()  # the poll asks, the frame answers
    image = unpng((tmp_path / ".manifoldx" / "look.png").read_bytes())
    assert image.shape == (60, 100, 3)  # 100 px is 400 bytes a row: not a multiple of 256
    assert (np.abs(image[30, 50].astype(int) - (255, 0, 0)) <= 2).all()
    assert not (tmp_path / ".manifoldx" / "requests" / "look").exists()


def test_state_writes_engine_state_as_json(hot, tmp_path):
    (tmp_path / ".manifoldx" / "requests" / "state").touch()
    hot.step()
    state = json.loads((tmp_path / ".manifoldx" / "state.json").read_text())
    assert state["loads"] == 1 and state["where"] == {"place": "yard", "x": 2.5}


def test_the_log_holds_the_run_the_reloads_and_the_apps_lines(hot, tmp_path):
    app = tmp_path / "app.py"
    app.write_text(app.read_text() + "\n")
    hot.reload()
    app.write_text("this is not python(")
    hot.reload()
    hot.step()
    log = (tmp_path / ".manifoldx" / "log.txt").read_text()
    assert "started" in log and "at the yard" in log
    assert "reloaded" in log and "reload failed, kept the running version" in log


def test_keys_are_logged_once_when_pressed(tmp_path, monkeypatch):
    channel = devchannel.Channel(tmp_path)
    engine = SimpleNamespace(input=SimpleNamespace(pressed_keys=frozenset()))
    try:
        for keys in ({"w"}, {"w"}, {"w", "Space"}, set(), {"w"}):
            engine.input.pressed_keys = frozenset(keys)
            channel.frame(engine)
    finally:
        channel.close()
    lines = [line.split(" ", 1)[1] for line in (tmp_path / ".manifoldx" / "log.txt").read_text().splitlines()]
    assert lines == ["keys w", "keys Space", "keys w"]


def test_ask_times_out_with_no_app(tmp_path):
    with pytest.raises(TimeoutError):
        devchannel.ask("look", timeout=0.1, root=tmp_path)
    assert not (tmp_path / ".manifoldx" / "requests" / "look").exists()


def test_cli_log_prints_the_last_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["log"]) == 1
    (tmp_path / ".manifoldx").mkdir()
    (tmp_path / ".manifoldx" / "log.txt").write_text("a\nb\nc\n")
    assert cli.main(["log", "-n", "2"]) == 0
    assert capsys.readouterr().out == "b\nc\n"
