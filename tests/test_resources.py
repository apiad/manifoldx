"""Tests for manifoldx.resources module."""
import numpy as np
import pytest


class MockDevice:
    """Mock device for testing (no GPU required)."""
    pass


def test_geometry_registry_id_allocation():
    """GeometryRegistry should allocate sequential IDs."""
    from manifoldx.resources import GeometryRegistry
    registry = GeometryRegistry(MockDevice())
    geo1 = object()
    geo2 = object()
    
    id1 = registry.register(geo1)
    id2 = registry.register(geo2)
    
    assert id1 != id2
    assert id1 == 1
    assert id2 == 2


def test_geometry_cache():
    """Same geometry object should return same ID."""
    from manifoldx.resources import GeometryRegistry
    registry = GeometryRegistry(MockDevice())
    geo = object()
    
    id1 = registry.register(geo)
    id2 = registry.register(geo)  # Same object
    
    assert id1 == id2  # Should return cached ID


def test_material_registry_pipeline_creation():
    """MaterialRegistry should allocate sequential IDs."""
    from manifoldx.resources import MaterialRegistry
    registry = MaterialRegistry(MockDevice())
    mat1 = object()
    mat2 = object()
    
    id1 = registry.register(mat1)
    id2 = registry.register(mat2)
    
    assert id1 != id2
    assert id1 == 1
    assert id2 == 2


def test_cube_geometry():
    """cube() should create geometry with positions and indices."""
    from manifoldx.resources import cube
    geo = cube(1, 1, 1)
    
    assert 'positions' in geo
    assert 'indices' in geo
    # Cube should have 8 vertices
    assert geo['positions'].shape[0] == 8
    # Cube should have 12 triangles (36 indices)
    assert geo['indices'].shape[0] == 36


def test_sphere_geometry():
    """sphere() should create geometry."""
    from manifoldx.resources import sphere
    geo = sphere(1.0, segments=16)
    
    assert 'positions' in geo
    assert 'indices' in geo


def test_basic_material():
    """basic() should create unlit material."""
    from manifoldx.resources import basic
    from manifoldx.types import Color
    mat = basic(Color("#ff0000"))
    
    assert mat.color is not None
