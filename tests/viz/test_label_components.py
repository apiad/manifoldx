"""Unit tests for TextLabel component."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.viz import TextLabel


def _make_engine():
    # TextLabel auto-registers on first spawn via the Component base class.
    return mx.Engine("test")


def test_text_label_default_zero():
    """No `index` argument → all entities get slice 0."""
    data = TextLabel().get_data(n=10)
    assert data.shape == (10, 1)
    assert data.dtype == np.float32
    assert np.all(data == 0.0)


def test_text_label_scalar_broadcast():
    data = TextLabel(index=42).get_data(n=5)
    assert data.shape == (5, 1)
    assert np.all(data == 42.0)


def test_text_label_array_per_entity():
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    data = TextLabel(index=indices).get_data(n=4)
    assert data.shape == (4, 1)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data[:, 0], indices)


def test_text_label_shape_mismatch_raises():
    with pytest.raises(ValueError):
        TextLabel(index=np.array([1, 2, 3])).get_data(n=4)


def test_text_label_spawn_into_engine():
    engine = _make_engine()
    from manifoldx.components import Transform

    engine.spawn(
        Transform(pos=np.zeros((3, 3), dtype=np.float32)),
        TextLabel(index=np.array([0, 1, 2], dtype=np.int64)),
        n=3,
    )
    assert "TextLabel" in engine.store._components
    assert engine.store._components["TextLabel"].shape[1] == 1
    assert int(np.sum(engine.store._alive)) == 3
