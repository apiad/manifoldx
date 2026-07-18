"""Tests for manifoldx.components module."""

import numpy as np
import pytest


def test_transform_dtype():
    """Transform should store pos(3) + rot(4) + scale(3) = 10 floats."""
    from manifoldx.ecs import EntityStore
    from manifoldx.components import Transform

    store = EntityStore()
    Transform.register(store)

    # SoA: each component stored separately
    assert "Transform" in store._components
    data = store._components["Transform"]
    assert data.shape[1] == 10  # pos(3) + rot(4) + scale(3)


def test_transform_default_values():
    """Transform should have default position (0,0,0), scale (1,1,1)."""
    from manifoldx.ecs import EntityStore
    from manifoldx.components import Transform

    store = EntityStore()
    Transform.register(store)

    # Spawn with default Transform data
    default_data = Transform.get_default_data(10)
    indices = store.spawn(n=10, Transform=default_data)
    view = store.get_component_view(["Transform"])

    # Check default position is (0, 0, 0) - first 3 columns
    pos_data = view.get_component_data("Transform")
    np.testing.assert_array_almost_equal(pos_data[:, 0:3], np.zeros((10, 3)))

    # Check default scale is (1, 1, 1) - columns 7-9
    np.testing.assert_array_almost_equal(pos_data[:, 7:10], np.ones((10, 3)))


def test_transform_get_data_applies_rot():
    """Transform(rot=...) must land in the quaternion columns 3:7 (regression:
    get_data previously dropped rot, leaving every spawned entity unrotated)."""
    from manifoldx.components import Transform

    q = (-0.70710678, 0.0, 0.0, 0.70710678)  # -90deg about X
    data = Transform(pos=(0, 0, 0), rot=q).get_data(1)
    np.testing.assert_array_almost_equal(data[0, 3:7], np.asarray(q, dtype=np.float32))


def test_transform_get_data_broadcasts_rot():
    """A single rot broadcasts across n entities."""
    from manifoldx.components import Transform

    q = (0.0, 0.38268343, 0.0, 0.92387953)  # 45deg about Y
    data = Transform(rot=q).get_data(5)
    for i in range(5):
        np.testing.assert_array_almost_equal(data[i, 3:7], np.asarray(q, dtype=np.float32))


def test_mesh_reference():
    """Mesh should store just an ID (uint32)."""
    from manifoldx.ecs import EntityStore
    from manifoldx.components import Mesh

    store = EntityStore()
    Mesh.register(store)

    # Verify it stores just an ID
    data = store._components["Mesh"]
    assert data.shape[1] == 1  # Single uint32


def test_material_reference():
    """Material should store just an ID (uint32)."""
    from manifoldx.ecs import EntityStore
    from manifoldx.components import Material

    store = EntityStore()
    Material.register(store)

    data = store._components["Material"]
    assert data.shape[1] == 1  # Single uint32
