import io
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


def _write_solid_png(path, w=2, h=2, rgba=(255, 0, 0, 255)):
    Image.new("RGBA", (w, h), rgba).save(path, format="PNG")


def test_texture_handle_fields():
    from manifoldx.textures import TextureHandle
    h = TextureHandle(id=1, texture=None, view=None, sampler=None, size=(2, 2))
    assert h.id == 1
    assert h.size == (2, 2)


def test_load_texture_uploads_to_gpu(tmp_path):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.textures import load_texture

    engine = mx.Engine("texture-upload", width=64, height=64)
    engine._init_canvas(canvas)

    png_path = tmp_path / "red.png"
    _write_solid_png(png_path)

    handle = load_texture(engine, png_path)
    assert handle.size == (2, 2)
    assert handle.texture is not None
    assert handle.view is not None
    assert handle.sampler is not None
    assert handle.id >= 1


def test_load_texture_missing_file_raises(tmp_path):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.textures import load_texture

    engine = mx.Engine("missing", width=64, height=64)
    engine._init_canvas(canvas)

    with pytest.raises(FileNotFoundError):
        load_texture(engine, tmp_path / "does-not-exist.png")
