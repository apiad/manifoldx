"""Tests for manifoldx.viz.materials module."""
import numpy as np
import pytest

from manifoldx.viz.materials import ColormapMaterial


def test_colormap_material_construction():
    m = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    assert m.cmap == "viridis"
    assert m.vmin == 0.0
    assert m.vmax == 1.0
    assert m.lit is False


def test_colormap_material_unknown_cmap():
    with pytest.raises(KeyError, match="unknown colormap"):
        ColormapMaterial(cmap="not-a-cmap", vmin=0.0, vmax=1.0)


def test_colormap_material_uniform_data():
    """Per-batch uniform: 4 floats — vmin, vmax, lit_flag, padding."""
    m = ColormapMaterial(cmap="viridis", vmin=-2.0, vmax=3.5, lit=True)
    data = m.get_data(n=10)
    # Per-batch material uniform; 4-float layout
    assert data.shape == (10, 4)
    assert data.dtype == np.float32
    # All rows identical (per-batch uniform)
    np.testing.assert_array_equal(data[0], data[5])
    np.testing.assert_array_equal(data[0], np.array([-2.0, 3.5, 1.0, 0.0], dtype=np.float32))


def test_colormap_material_lut_bytes():
    m = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    lut = m.get_lut()
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8


def test_colormap_material_pipeline_subtype():
    """Pipeline cache key uses the cmap name as subtype."""
    m_v = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    m_m = ColormapMaterial(cmap="magma", vmin=0.0, vmax=1.0)
    assert m_v.pipeline_subtype == "viridis"
    assert m_m.pipeline_subtype == "magma"
    assert m_v.pipeline_subtype != m_m.pipeline_subtype
