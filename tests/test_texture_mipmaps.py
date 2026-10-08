"""load_texture builds a mip chain, so a finely tiled texture far away averages out."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, plane


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("mip", width=64, height=64)
    e.render_frame(supersample=1)  # creates the device load_texture needs
    return e


def test_mip_chain_reaches_one_pixel(tmp_path):
    from PIL import Image
    from manifoldx.textures import load_texture
    png = tmp_path / "t.png"
    Image.new("RGB", (64, 32), (200, 100, 50)).save(png)
    tex = load_texture(_engine(), png)
    assert tex.texture.mip_level_count == 7  # 64x32, 32x16, ..., 1x1


def test_tiled_checkerboard_averages_to_grey_at_a_distance(tmp_path):
    from PIL import Image
    from manifoldx.textures import load_texture
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    png = tmp_path / "checker.png"
    Image.fromarray(np.stack([checker] * 3, axis=-1)).save(png)
    e = _engine()
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, 0, -1)))
    tex = load_texture(e, png)
    quad = plane(2.0, 2.0)
    quad["uvs"] = quad["uvs"] * 40.0  # 40 repeats of a 64 px checker: ~2560 texels across 64 px
    e.spawn(Mesh(quad), Material(StandardMaterial(color="#ffffff", roughness=1.0, albedo_map=tex)),
            Transform(pos=(0, 0, 0)))
    e.camera.set_pose(position=(0, 0, 2.2), target=(0, 0, 0))
    img = e.render_frame(supersample=1).astype(float)[20:44, 20:44, 0]
    # Measured on zion: std 18.1 without mipmaps (moire), 0.37 with them.
    assert img.std() < 5, img.std()
