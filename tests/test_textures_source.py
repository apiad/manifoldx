from pathlib import Path

import pytest

from manifoldx.textures import TextureHandle


def test_handle_has_optional_source():
    h = TextureHandle(id=1, texture=None, view=None, sampler=None, size=(2, 2))
    assert h.source is None
    h2 = TextureHandle(id=2, texture=None, view=None, sampler=None, size=(2, 2), source=Path("a.png"))
    assert h2.source == Path("a.png")


def test_load_texture_records_its_file(tmp_path):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    from PIL import Image

    import manifoldx as mx

    path = tmp_path / "t.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(path)
    engine = mx.Engine("t", width=64, height=64)
    engine.render_frame(supersample=1)  # creates the device
    handle = mx.load_texture(engine, path)
    assert handle.source == path
