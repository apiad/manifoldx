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


def test_radius_default_data():
    r = viz_components.Radius()
    data = r.get_data(n=5)
    assert data.shape == (5, 1)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, np.ones((5, 1), dtype=np.float32))


def test_radius_scalar_broadcast():
    r = viz_components.Radius(radius=0.05)
    data = r.get_data(n=10)
    np.testing.assert_array_equal(data, np.full((10, 1), 0.05, dtype=np.float32))


def test_radius_array_input():
    radii = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    r = viz_components.Radius(radius=radii)
    data = r.get_data(n=3)
    np.testing.assert_array_equal(data[:, 0], radii)


def test_point_cloud_marker_no_data():
    """PointCloud is a marker — get_data returns an empty (n, 0) array."""
    pc = viz_components.PointCloud()
    data = pc.get_data(n=42)
    assert data.shape == (42, 0)
    assert data.dtype == np.float32


def test_point_cloud_importable_from_viz():
    from manifoldx.viz import PointCloud, ScalarValue, Radius
    assert PointCloud is viz_components.PointCloud
    assert ScalarValue is viz_components.ScalarValue
    assert Radius is viz_components.Radius
