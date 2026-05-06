"""Unit tests for LabelMaterial."""
import numpy as np

from manifoldx.viz import LabelMaterial


def test_label_material_defaults():
    mat = LabelMaterial()
    assert mat.pixel_width == 256.0
    assert mat.pixel_height == 64.0
    assert mat.anchor_mode == "world"


def test_label_material_uniform_data_shape():
    mat = LabelMaterial(pixel_width=128, pixel_height=32)
    data = mat.get_data(n=5)
    assert data.shape == (5, 4)
    assert data.dtype == np.float32
    # Row 0 is the canonical row used by the renderer.
    np.testing.assert_array_equal(data[0], [128.0, 32.0, 0.0, 0.0])


def test_label_material_pipeline_subtype_is_world_or_screen():
    """The cache key includes anchor_mode so world and screen pipelines diverge."""
    a = LabelMaterial(anchor_mode="world").pipeline_subtype
    # `screen` is reserved for Plan 3; LabelMaterial accepts the literal but
    # keeps the renderer-side fallback to world for v2 (Plan 2).
    assert a == "world"


def test_label_material_compile_returns_wgsl_source():
    src = LabelMaterial._compile()
    assert "@vertex" in src
    assert "@fragment" in src
    assert "atlas_texture" in src


def test_label_material_uniform_type_layout():
    fields = LabelMaterial.uniform_type()
    assert list(fields.keys()) == ["pixel_width", "pixel_height", "anchor_mode", "_pad"]
    assert all(t == "f32" for t in fields.values())


def test_label_material_invalid_anchor_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        LabelMaterial(anchor_mode="bogus")
