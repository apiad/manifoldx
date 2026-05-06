"""Tests for manifoldx.viz.components module."""
import numpy as np

from manifoldx.viz import components as viz_components


def test_scalar_value_default_data():
    sv = viz_components.ScalarValue()
    data = sv.get_data(n=5)
    assert data.shape == (5, 1)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, np.zeros((5, 1), dtype=np.float32))


def test_scalar_value_scalar_broadcast():
    sv = viz_components.ScalarValue(value=2.5)
    data = sv.get_data(n=10)
    assert data.shape == (10, 1)
    np.testing.assert_array_equal(data, np.full((10, 1), 2.5, dtype=np.float32))


def test_scalar_value_array_input():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sv = viz_components.ScalarValue(value=values)
    data = sv.get_data(n=4)
    assert data.shape == (4, 1)
    np.testing.assert_array_equal(data[:, 0], values)
