import pytest
from manifoldx.resources import StandardMaterial
from manifoldx.textures import TextureHandle


def _fake_handle():
    return TextureHandle(id=1, texture=object(), view=object(),
                         sampler=object(), size=(2, 2))


def test_scalar_subtype_is_none():
    m = StandardMaterial(color="#ff0000")
    assert m.pipeline_subtype is None
    assert "@binding(4)" not in m._compile()


def test_textured_subtype_and_shader():
    m = StandardMaterial(color="#ff0000", albedo_map=_fake_handle())
    assert m.pipeline_subtype == "textured"
    src = m._compile(textured=True)
    assert "@binding(4)" in src
    assert "@binding(5)" in src
    assert "textureSample" in src
    assert "@location(2) uv" in src


def test_textured_get_texture_bindings():
    handle = _fake_handle()
    m = StandardMaterial(color="#ff0000", albedo_map=handle)
    bindings = m.get_texture_bindings()
    assert bindings == {4: handle}


def test_scalar_get_texture_bindings_empty():
    m = StandardMaterial(color="#ff0000")
    assert m.get_texture_bindings() == {}


def test_albedo_map_type_error():
    with pytest.raises(TypeError, match="TextureHandle"):
        StandardMaterial(color="#ff0000", albedo_map="not-a-handle")
