"""Material colours are sRGB, like textures: color= and albedo_map agree."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import (
    AtmosphereMaterial,
    BasicMaterial,
    PhongMaterial,
    PointLight,
    StandardMaterial,
    WaterMaterial,
    plane,
)


def _decode(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


EXPECTED = [_decode(0xCC / 255), _decode(0x22 / 255), _decode(0x22 / 255)]


@pytest.mark.parametrize("make", [
    lambda: BasicMaterial("#cc2222"),
    lambda: PhongMaterial("#cc2222"),
    lambda: StandardMaterial(color="#cc2222"),
    lambda: StandardMaterial(color=(0xCC / 255, 0x22 / 255, 0x22 / 255)),
])
def test_material_colours_are_decoded_from_srgb(make):
    data = make().get_data(1, None)
    assert np.allclose(data[0, :3], EXPECTED, atol=1e-6)


@pytest.mark.parametrize("cls", [WaterMaterial, AtmosphereMaterial])
def test_special_materials_decode_their_colour_too(cls):
    import inspect
    m = cls(color="#cc2222") if "color" in inspect.signature(cls).parameters else cls("#cc2222")
    data = np.asarray(m.get_data(1, None)).ravel()
    assert np.allclose(data[:3], EXPECTED, atol=1e-6)


def test_color_and_albedo_map_of_the_same_value_render_the_same(tmp_path):
    # Review F5: under one point light, color="#cc2222" rendered (111, 50, 50)
    # and a #cc2222 albedo_map rendered (98, 19, 19).
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    from PIL import Image
    from manifoldx.textures import load_texture
    png = tmp_path / "red.png"
    Image.new("RGB", (8, 8), (0xCC, 0x22, 0x22)).save(png)
    e = mx.Engine("cs", width=64, height=32)
    e.add_light(PointLight(color="#ffffff", intensity=20.0, position=(0, 0, 2.5)))
    e.spawn(Mesh(plane(1.6, 1.6)), Material(StandardMaterial(color="#cc2222", roughness=0.9)), Transform(pos=(-1, 0, 0)))
    e.render_frame(supersample=1)  # creates the device load_texture needs
    tex = load_texture(e, png)
    e.spawn(Mesh(plane(1.6, 1.6)), Material(StandardMaterial(color="#ffffff", roughness=0.9, albedo_map=tex)), Transform(pos=(1, 0, 0)))
    e.camera.set_pose(position=(0, 0, 3), target=(0, 0, 0))
    img = e.render_frame(supersample=1).astype(int)
    left, right = img[16, 16], img[16, 48]
    assert np.abs(left - right).max() <= 3, (left, right)
