"""Tests for system command queueing (Phases 1-4)."""
import numpy as np
import pytest


def test_component_view_has_engine_reference():
    """ComponentView should have access to engine for command queueing."""
    from manifoldx.ecs import EntityStore, ComponentView
    
    store = EntityStore(max_entities=10)
    store.register_component('TestComp', np.dtype('f4'), (3,))
    
    # Create a mock engine with command buffer
    class MockEngine:
        def __init__(self):
            from manifoldx.commands import CommandBuffer
            self.commands = CommandBuffer()
    
    engine = MockEngine()
    
    # ComponentView should be able to access engine's commands
    indices = np.array([0, 1, 2])
    view = ComponentView(store, ['TestComp'], indices)
    
    # The view should have access to commands - this will fail initially
    # because ComponentView doesn't store engine reference yet
    assert hasattr(view, '_engine'), "ComponentView should have _engine attribute"


def test_component_accessor_has_commands():
    """ComponentAccessor should be able to queue commands."""
    from manifoldx.ecs import EntityStore, ComponentView
    from manifoldx.commands import CommandBuffer, CommandType
    
    store = EntityStore(max_entities=10)
    store.register_component('TestComp', np.dtype('f4'), (3,))
    
    # Spawn some entities
    indices = store.spawn(n=3, TestComp=np.zeros((3, 3), dtype=np.float32))
    
    # Create mock engine with command buffer
    class MockEngine:
        def __init__(self):
            self.commands = CommandBuffer()
    
    engine = MockEngine()
    
    # Get view and accessor
    view = ComponentView(store, ['TestComp'], indices)
    accessor = view['TestComp']
    
    # Accessor should have access to commands
    assert hasattr(accessor, '_commands'), "ComponentAccessor should have _commands"


def test_field_view_queues_command():
    """_FieldView should queue UPDATE_COMPONENT command instead of writing directly."""
    from manifoldx.ecs import EntityStore, ComponentView
    from manifoldx.commands import CommandBuffer, CommandType
    
    store = EntityStore(max_entities=10)
    store.register_component('TestComp', np.dtype('f4'), (3,))
    
    # Register a component class with fields so we can use field access
    class TestComp:
        _component_start_idx = {'x': (0, 1), 'y': (1, 1), 'z': (2, 1)}
    
    from manifoldx.ecs import _COMPONENT_REGISTRY
    _COMPONENT_REGISTRY['TestComp'] = TestComp
    
    # Spawn entities with initial data
    initial_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    indices = store.spawn(n=2, TestComp=initial_data)
    
    # Create mock engine with command buffer
    class MockEngine:
        def __init__(self):
            self.commands = CommandBuffer()
    
    engine = MockEngine()
    
    # Get view and field accessor - pass engine to ComponentView
    view = ComponentView(store, ['TestComp'], indices, engine)
    accessor = view['TestComp']
    
    # Get field using proper field access (this returns a _FieldView)
    field = accessor.x
    
    # Now do += operation - should queue command, not write directly
    field += 10
    
    # Check that command was queued
    assert len(engine.commands) > 0, "Operation should queue a command"
    
    # Execute commands
    engine.commands.execute(store)
    
    # Verify data was updated - only x field (column 0) should be updated
    updated = store.get_component_data('TestComp', indices)
    assert np.allclose(updated[:, 0], [11, 14]), "X field should be updated after command execution"
    assert np.allclose(updated[:, 1], [2, 5]), "Y field should NOT be updated"
    assert np.allclose(updated[:, 2], [3, 6]), "Z field should NOT be updated"


def test_transform_position_updates_after_system():
    """Transform position should update after system runs and commands execute."""
    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    
    # Create engine
    engine = mx.Engine('Test')
    engine._init_webgpu()
    
    cube_mesh = mx.geometry.cube(1,1,1)
    cube_material = mx.material.phong(mx.colors.RED)
    
    # Spawn cube
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform=Transform(pos=(0,0,0)),
        n=1,
    )
    
    # Get view like system does
    view = engine.store.get_component_view(['Transform'])
    acc = view['Transform']
    
    # Update position
    acc.pos += (0, 0.5, 0)
    
    # Execute commands (simulating what happens in _draw_frame)
    engine.commands.execute(engine.store)
    
    # Check that position was actually updated in store
    position = engine.store._components['Transform'][0, 0:3]
    assert np.allclose(position, [0, 0.5, 0], atol=1e-5), \
        f"Position should be updated, got {position}"
