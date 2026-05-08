"""VolumeMaterial: opacity LUT baking + parameter validation."""
import numpy as np
import pytest

from manifoldx.viz import VolumeMaterial


def test_default_opacity_lut_is_linear_ramp():
    """opacity_stops=None bakes alpha = linspace(0, 1, 256)."""
    m = VolumeMaterial()
    expected = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    np.testing.assert_allclose(m.opacity_lut, expected, rtol=1e-6)


def test_opacity_stops_full_range_linear():
    """[(0,0),(1,1)] piecewise-linear matches the default ramp."""
    m = VolumeMaterial(opacity_stops=[(0.0, 0.0), (1.0, 1.0)])
    expected = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    np.testing.assert_allclose(m.opacity_lut, expected, rtol=1e-6)


def test_opacity_stops_step_at_half():
    """[(0,0),(0.5,0),(0.5,1),(1,1)] produces a step at index 128."""
    m = VolumeMaterial(opacity_stops=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 1.0)])
    assert m.opacity_lut[127] == 0.0
    assert m.opacity_lut[128] == 1.0


def test_opacity_stops_array_used_as_is():
    """Pre-baked (256,) float32 array passes through unchanged."""
    arr = np.linspace(0.5, 1.0, 256, dtype=np.float32)
    m = VolumeMaterial(opacity_stops=arr)
    np.testing.assert_array_equal(m.opacity_lut, arr)


def test_opacity_stops_array_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(256,\)"):
        VolumeMaterial(opacity_stops=np.zeros(128, dtype=np.float32))


def test_opacity_stops_must_be_sorted():
    with pytest.raises(ValueError, match="ascending"):
        VolumeMaterial(opacity_stops=[(0.5, 0.0), (0.2, 1.0)])


def test_unknown_cmap_lists_available():
    with pytest.raises(ValueError, match="viridis"):
        VolumeMaterial(cmap="not-a-real-colormap")


def test_vmin_must_be_less_than_vmax():
    with pytest.raises(ValueError, match="vmin"):
        VolumeMaterial(vmin=1.0, vmax=1.0)


def test_step_size_must_be_positive():
    with pytest.raises(ValueError, match="step_size"):
        VolumeMaterial(step_size=0.0)
    with pytest.raises(ValueError, match="step_size"):
        VolumeMaterial(step_size=-0.1)


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError, match="max_steps"):
        VolumeMaterial(max_steps=0)
