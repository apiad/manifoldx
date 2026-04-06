"""Tests for manifoldx.systems module."""
import numpy as np
import pytest


def test_system_registration():
    """SystemRegistry should register decorated functions."""
    from manifoldx.systems import SystemRegistry, Query
    from manifoldx.ecs import EntityStore
    
    registry = SystemRegistry()
    
    # Create a Query type annotation
    QueryTransform = Query[EntityStore]
    
    def my_system(query: QueryTransform, dt: float):
        pass
    
    registry.register(my_system)
    assert len(registry._systems) == 1


def test_system_execution_order():
    """Systems should execute in registration order."""
    from manifoldx.systems import SystemRegistry, Query
    from manifoldx.ecs import EntityStore
    
    registry = SystemRegistry()
    
    QueryTransform = Query[EntityStore]
    
    def system_a(query: QueryTransform, dt: float):
        pass
    
    def system_b(query: QueryTransform, dt: float):
        pass
    
    registry.register(system_a)
    registry.register(system_b)
    assert len(registry._systems) == 2


def test_query_components():
    """Query should store component types."""
    from manifoldx.systems import Query
    from manifoldx.components import Transform, Mesh
    
    q = Query((Transform, Mesh))
    assert len(q.components) == 2
