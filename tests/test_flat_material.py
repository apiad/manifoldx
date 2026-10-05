"""FlatMaterial outputs its colour with no lighting."""
import numpy as np

from manifoldx.resources import (
    _BASICMATERIAL_SHADER,
    _FLATMATERIAL_SHADER,
    BasicMaterial,
    FlatMaterial,
)


def test_shader_has_no_lighting_term():
    assert _FLATMATERIAL_SHADER != _BASICMATERIAL_SHADER
    src = FlatMaterial._compile()
    assert "light_dir" not in src and "diffuse" not in src
    assert "material.color" in src


def test_hex_and_tuple_colours_become_vec4():
    hexed = FlatMaterial("#cc2222").get_data(2, None)
    assert hexed.shape == (2, 4) and hexed.dtype == np.float32
    assert np.allclose(hexed[0], [0xCC / 255, 0x22 / 255, 0x22 / 255, 1.0])
    tup = FlatMaterial((0.1, 0.2, 0.3)).get_data(1, None)
    assert np.allclose(tup[0], [0.1, 0.2, 0.3, 1.0])


def test_basic_material_no_longer_claims_to_be_unlit():
    assert "Unlit" not in (BasicMaterial.__doc__ or "")
