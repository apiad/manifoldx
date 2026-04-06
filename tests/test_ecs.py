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
