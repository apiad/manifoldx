"""Unit tests for the AxisFrame component."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.viz import AxisFrame


def test_axis_frame_defaults():
    af = AxisFrame()
    data = af.get_data(n=3)
    assert data.shape == (3, 2)
    assert data.dtype == np.float32
    # Default extent and thickness are both 1.0.
    np.testing.assert_array_equal(data[:, 0], np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(data[:, 1], np.ones(3, dtype=np.float32))


def test_axis_frame_scalar_broadcast():
    af = AxisFrame(extent=5.0, thickness=2.0)
    data = af.get_data(n=4)
    assert data.shape == (4, 2)
    np.testing.assert_array_equal(data[:, 0], np.full(4, 5.0, dtype=np.float32))
    np.testing.assert_array_equal(data[:, 1], np.full(4, 2.0, dtype=np.float32))


def test_axis_frame_per_entity_extent():
    extents = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    af = AxisFrame(extent=extents)
    data = af.get_data(n=3)
    np.testing.assert_array_equal(data[:, 0], extents)
    # thickness falls back to default 1.0 for all entities.
    np.testing.assert_array_equal(data[:, 1], np.ones(3, dtype=np.float32))


def test_axis_frame_spawn_into_engine():
    """Auto-registers via the Component base when spawn() sees the instance."""
    from manifoldx.components import Material, Transform
    from manifoldx.viz import AxisMaterial

    engine = mx.Engine("test")
    engine.spawn(
        Material(AxisMaterial(color="#ff0000")),
        Transform(pos=np.zeros((3, 3), dtype=np.float32)),
        AxisFrame(extent=np.array([1.0, 1.0, 1.0], dtype=np.float32)),
        n=3,
    )
    assert "AxisFrame" in engine.store._components
    assert engine.store._components["AxisFrame"].shape[1] == 2
    assert int(np.sum(engine.store._alive)) == 3


def test_axis_frame_importable_from_viz():
    from manifoldx.viz import AxisFrame as AxisFrame2
    assert AxisFrame is AxisFrame2
