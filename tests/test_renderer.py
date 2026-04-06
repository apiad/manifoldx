"""Tests for manifoldx.renderer module."""
import numpy as np
import pytest


def test_transform_cache_dirty_flag():
    """Test that cache tracks dirty entities."""
    from manifoldx.renderer import TransformCache
    
    cache = TransformCache(max_entities=100)
    
    # Initially all dirty
    assert np.all(cache._dirty == True)
    
    # Mark some as clean
    cache._dirty[5:10] = False
    assert np.all(cache._dirty[:5] == True)
    assert np.all(cache._dirty[5:10] == False)


def test_transform_cache_mark_dirty():
    """Test mark_dirty sets dirty flag."""
    from manifoldx.renderer import TransformCache
    
    cache = TransformCache(max_entities=100)
    
    # Initially all dirty
    assert np.all(cache._dirty == True)
    
    indices = np.array([0, 5, 10])
    cache.mark_dirty(indices)
    
    # Those indices should be dirty
    assert cache._dirty[0] == True
    assert cache._dirty[5] == True
    assert cache._dirty[10] == True


def test_render_pipeline():
    """Test RenderPipeline initialization."""
    from manifoldx.ecs import EntityStore
    from manifoldx.renderer import RenderPipeline
    from manifoldx.components import Transform
    
    store = EntityStore()
    Transform.register(store)
    
    # Spawn some entities
    default_data = Transform.get_default_data(10)
    default_data[:, 0:3] = np.arange(30).reshape(10, 3)  # positions
    indices = store.spawn(n=10, Transform=default_data)
    
    pipeline = RenderPipeline(store)
    
    # Create a mock engine
    class MockEngine:
        pass
    
    engine = MockEngine()
    pipeline.run(engine, 1/60)
