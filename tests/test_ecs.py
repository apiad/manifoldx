"""Tests for manifoldx.ecs module."""
import numpy as np
import pytest


def test_entity_store_creation():
    """EntityStore should initialize with max_entities."""
    from manifoldx.ecs import EntityStore
    store = EntityStore(max_entities=1000)
    assert store.max_entities == 1000


def test_register_component():
    """Registering a component should create storage array."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    assert "velocity" in store._components
    # SoA: each component stored separately
    assert store._components["velocity"].shape == (100_000, 3)


def test_register_multiple_components():
    """Multiple components should each have separate storage."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    # Check separate arrays per component (SoA)
    assert store._components["position"].shape == (100_000, 3)
    assert store._components["velocity"].shape == (100_000, 3)


def test_spawn_single():
    """Spawning a single entity should return valid index."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=1, position=[[1, 2, 3]])
    assert len(indices) == 1
    assert store._alive[indices[0]] == True


def test_spawn_multiple():
    """Spawning multiple entities should return all indices."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=100, position=np.random.rand(100, 3))
    assert len(indices) == 100


def test_spawn_reuses_dead():
    """Spawning after destroy should reuse dead slots."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    # Spawn and kill
    idx1 = store.spawn(n=1, position=[[1, 2, 3]])
    store.destroy(idx1)
    # Spawn again - should reuse slot
    idx2 = store.spawn(n=1, position=[[4, 5, 6]])
    assert idx2[0] == idx1[0]


def test_destroy():
    """Destroying entities should mark them as dead."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=10, position=np.random.rand(10, 3))
    store.destroy(indices[:5])
    assert np.all(store._alive[indices[:5]] == False)
    assert np.all(store._alive[indices[5:]] == True)


def test_query_basic():
    """Query should return entities with matching components."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    
    # Spawn with both components
    idx = store.spawn(n=5, position=np.zeros((5, 3)), velocity=np.ones((5, 3)))
    
    view = store.get_component_view(['position', 'velocity'])
    assert len(view) == 5


def test_component_view_getitem():
    """ComponentView should allow accessing component data."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=10, position=np.random.rand(10, 3))
    
    view = store.get_component_view(['position'])
    pos_data = view.get_component_data('position')
    assert pos_data.shape == (10, 3)


# =============================================================================
# _FieldView Operator Tests
# =============================================================================

def _make_field_view(n=5):
    """Helper to create a _FieldView for testing."""
    from manifoldx.ecs import EntityStore, _FieldView
    store = EntityStore(max_entities=100)
    store.register_component("Test", np.dtype("f4"), (3,))
    indices = store.spawn(n)
    # Data: [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]]
    store._components["Test"][indices] = np.arange(n * 3).reshape(n, 3).astype(np.float32)
    return _FieldView(store, indices, "Test", 0, 3, "data", None), store._components["Test"][indices]


def test_fieldview_add_scalar():
    """_FieldView + scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv + 1
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data + 1)


def test_fieldview_add_array():
    """_FieldView + array should return ndarray."""
    fv, data = _make_field_view()
    result = fv + np.array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data + np.array([1, 2, 3]))


def test_fieldview_add_fieldview():
    """_FieldView + _FieldView should add element-wise."""
    fv, data = _make_field_view()
    result = fv + fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data + data)


def test_fieldview_sub_scalar():
    """_FieldView - scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv - 1
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data - 1)


def test_fieldview_sub_array():
    """_FieldView - array should return ndarray."""
    fv, data = _make_field_view()
    result = fv - np.array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data - np.array([1, 2, 3]))


def test_fieldview_sub_fieldview():
    """_FieldView - _FieldView should subtract element-wise."""
    fv, data = _make_field_view()
    result = fv - fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data - data)


def test_fieldview_mul_scalar():
    """_FieldView * scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv * 2
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data * 2)


def test_fieldview_mul_array():
    """_FieldView * array should multiply element-wise."""
    fv, data = _make_field_view()
    result = fv * np.array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data * np.array([1, 2, 3]))


def test_fieldview_mul_fieldview():
    """_FieldView * _FieldView should multiply element-wise."""
    fv, data = _make_field_view()
    result = fv * fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data * data)


def test_fieldview_div_scalar():
    """_FieldView / scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv / 2
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data / 2)


def test_fieldview_radd_scalar():
    """scalar + _FieldView should return ndarray."""
    fv, data = _make_field_view()
    result = 1 + fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 1 + data)


def test_fieldview_rsub_scalar():
    """scalar - _FieldView should return ndarray."""
    fv, data = _make_field_view()
    result = 10 - fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 10 - data)


def test_fieldview_rtruediv_scalar():
    """scalar / _FieldView should return ndarray."""
    fv, data = _make_field_view(5)
    # Avoid zeros in data for division
    store = fv._store
    store._components["Test"][fv._indices] = np.arange(1, 16).reshape(5, 3).astype(np.float32)
    data = store._components["Test"][fv._indices]
    result = 10 / fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 10 / data)


def test_fieldview_combined_expression():
    """Combined expressions like np.log(fv + 1) should work."""
    fv, data = _make_field_view()
    result = np.log(fv + 1)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.log(data + 1))


def test_fieldview_expression_order():
    """Expression order should follow standard python precedence."""
    fv, data = _make_field_view()
    result = 3 * fv + 2
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 3 * data + 2)
