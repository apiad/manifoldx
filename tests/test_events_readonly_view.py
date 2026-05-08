"""ReadOnlyView wraps a ComponentView and forbids mutation."""
import pytest
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform
from manifoldx.events import ReadOnlyView


def _engine_with_one_entity():
    engine = mx.Engine("test", width=64, height=64)
    engine.spawn(Transform(pos=(1.0, 2.0, 3.0)), n=1)
    return engine


def test_read_passes_through():
    engine = _engine_with_one_entity()
    view = engine.store.get_component_view(["Transform"], engine)
    ro = ReadOnlyView(view)
    pos_data = ro[Transform].pos.data
    assert pos_data.shape[1] == 3
    np.testing.assert_allclose(pos_data[0], [1.0, 2.0, 3.0])


def test_setitem_on_view_raises():
    engine = _engine_with_one_entity()
    view = engine.store.get_component_view(["Transform"], engine)
    ro = ReadOnlyView(view)
    with pytest.raises(RuntimeError, match="cannot mutate ECS data"):
        ro[Transform] = "anything"


def test_attribute_assignment_through_accessor_raises():
    engine = _engine_with_one_entity()
    view = engine.store.get_component_view(["Transform"], engine)
    ro = ReadOnlyView(view)
    with pytest.raises(RuntimeError, match="cannot mutate ECS data"):
        ro[Transform].pos = (9.0, 9.0, 9.0)
