"""Render pipeline that executes after command buffer."""
import numpy as np


# =============================================================================
# Transform Cache (With Dirty Flag Optimization)
# =============================================================================

class TransformCache:
    """
    Caches computed transform matrices.
    
    Stores: _matrix_cache (N, 16) - mat4x4<f32> per entity
            _dirty (N,) bool - needs recompute
    """
    
    def __init__(self, max_entities: int):
        self._matrix_cache = np.zeros((max_entities, 16), dtype=np.float32)
        self._dirty = np.ones(max_entities, dtype=bool)  # All dirty initially
        
    def mark_dirty(self, indices: np.ndarray):
        """Mark entities as needing recompute."""
        self._dirty[indices] = True
        
    def get_transforms(self, store, indices: np.ndarray) -> np.ndarray:
        """
        Get transform matrices for indices, recomputing if dirty.
        Returns (len(indices), 16) array of mat4x4<f32>.
        """
        dirty_mask = self._dirty[indices]
        if dirty_mask.any():
            self._recompute_matrices(store, indices[dirty_mask])
            
        return self._matrix_cache[indices]
    
    def _recompute_matrices(self, store, indices: np.ndarray):
        """Compute mat4x4 from pos/rot/scale using vectorized numpy."""
        if len(indices) == 0:
            return
            
        transform_data = store.get_component_data('Transform', indices)
        
        positions = transform_data[:, 0:3]  # (N, 3)
        # rotations = transform_data[:, 3:7]  # quat (not used in MVP)
        # scales = transform_data[:, 7:10]  # (N, 3) (not used in MVP)
        
        # Simplified: just use position, ignore rotation/scale for MVP
        matrices = np.zeros((len(indices), 16), dtype=np.float32)
        matrices[:, 0] = 1.0  # scale X
        matrices[:, 5] = 1.0  # scale Y
        matrices[:, 10] = 1.0  # scale Z
        matrices[:, 15] = 1.0  # w
        
        # Set positions in matrix (simplified translation)
        matrices[:, 12] = positions[:, 0]  # translate X
        matrices[:, 13] = positions[:, 1]  # translate Y
        matrices[:, 14] = positions[:, 2]  # translate Z
        
        self._matrix_cache[indices] = matrices
        self._dirty[indices] = False


# =============================================================================
# Render Pipeline
# =============================================================================

class RenderPipeline:
    """
    Render pipeline that executes after command buffer.
    
    Flow:
    1. Query all alive entities with Mesh + Material + Transform
    2. Group by (geometry_id, material_id) into batches
    3. For each batch: compute transforms (using cache), upload, draw
    """
    
    def __init__(self, store, device=None):
        self._store = store
        self._device = device
        self._transform_cache = TransformCache(store.max_entities)
        
    def run(self, engine, dt: float):
        """Execute full render pipeline (GPU operations skipped in tests)."""
        # Get all alive entities with required components
        if 'Transform' not in self._store._components:
            return
            
        alive_indices = np.where(self._store._alive)[0]
        if len(alive_indices) == 0:
            return
        
        # Mark transforms as dirty for all alive entities
        self._transform_cache.mark_dirty(alive_indices)
        
        # Get transforms using cache
        transforms = self._transform_cache.get_transforms(self._store, alive_indices)
        
        # Verify transforms were computed
        assert transforms.shape[1] == 16  # mat4x4


__all__ = [
    'TransformCache',
    'RenderPipeline',
]
