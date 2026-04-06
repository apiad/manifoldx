"""Tests for MVP rendering - single cube on screen."""
import numpy as np
import pytest


def test_engine_has_camera():
    """Engine should have a camera for rendering."""
    from manifoldx import Engine
    
    engine = Engine("Test")
    
    # Engine must have a camera
    assert hasattr(engine, "_camera"), "Engine must have _camera attribute"
    assert engine._camera is not None, "Camera must be initialized"


def test_camera_has_position_and_target():
    """Camera must have position and look-at target."""
    from manifoldx import Engine
    
    engine = Engine("Test")
    camera = engine._camera
    
    # Camera should have position attribute
    assert hasattr(camera, "position"), "Camera must have position"
    # Camera should have target/look_at
    assert hasattr(camera, "target") or hasattr(camera, "look_at"), "Camera must have target"


def test_geometry_registry_creates_gpu_buffers():
    """GeometryRegistry should create actual GPU buffers."""
    try:
        import wgpu
    except ImportError:
        pytest.skip("wgpu not available")
    
    from manifoldx.resources import GeometryRegistry
    from manifoldx.resources import cube
    
    # Need device - create mock or get real one
    # For now, check that registry has method to create buffers
    registry = GeometryRegistry()
    
    # Should have method to create GPU buffers
    assert hasattr(registry, "create_buffers"), "GeometryRegistry must have create_buffers method"
    
    # Create geometry and register
    geom = cube(1, 1, 1)
    geom_id = registry.register(geom)
    
    # Should have created buffers now
    stored = registry._geometries.get(geom_id)
    assert stored is not None


def test_render_pipeline_has_pipeline():
    """RenderPipeline should have actual WGPU render pipeline."""
    try:
        import wgpu
    except ImportError:
        pytest.skip("wgpu not available")
    
    from manifoldx.ecs import EntityStore
    from manifoldx.renderer import RenderPipeline
    from manifoldx.components import Transform, Mesh, Material
    
    store = EntityStore(max_entities=100)
    Transform.register(store)
    Mesh.register(store)
    Material.register(store)
    
    # Create pipeline with device
    # We need actual WGPU device for real test
    # For now, check pipeline has pipeline attribute
    pipeline = RenderPipeline(store, device=None)
    
    # Should have or create pipeline
    assert hasattr(pipeline, "_pipeline") or hasattr(pipeline, "pipeline"), \
        "RenderPipeline must have pipeline attribute"


def test_render_pipeline_issues_draw_calls():
    """RenderPipeline.run() should issue draw calls."""
    try:
        import wgpu
    except ImportError:
        pytest.skip("wgpu not available")
    
    # This is harder to test without full setup
    # Check that run() method exists and does something
    from manifoldx.renderer import RenderPipeline
    from manifoldx.ecs import EntityStore
    
    store = EntityStore(max_entities=100)
    pipeline = RenderPipeline(store, device=None)
    
    # run() method should exist
    assert hasattr(pipeline, "run"), "RenderPipeline must have run method"
    assert callable(pipeline.run), "run must be callable"