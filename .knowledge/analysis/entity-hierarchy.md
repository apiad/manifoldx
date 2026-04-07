# Entity Hierarchy Feature Analysis

## Overview

Add a parent-child relationship system where each entity can have a parent entity ID. Before rendering, compute world matrices by multiplying local matrices by their parent's world matrix, up to a maximum depth (e.g., 8 levels).

## Current State

### Transform System

The renderer already computes local transform matrices from ECS data:

- **Input**: position (vec3), rotation (quaternion vec4), scale (vec3) - 10 floats
- **Output**: mat4x4<f32> world matrix - 16 floats per entity
- **Storage**: `_matrix_cache` (N, 16) in `TransformCache` class

Key code in `renderer.py`:
```python
def get_transforms(self, store, indices: np.ndarray) -> np.ndarray:
    dirty_mask = self._dirty[indices]
    if dirty_mask.any():
        self._recompute_matrices(store, indices[dirty_mask])
    return self._matrix_cache[indices]
```

The `_recompute_matrices` method converts (pos, rot, scale) → mat4 using vectorized numpy.

---

## Hierarchy Design

### Component Addition

Add a new `Parent` component:

```python
class Parent:
    """Parent component storing parent entity ID."""
    
    def __init__(self, parent_id: int | None = None):
        self._parent_id = parent_id
    
    def get_data(self, n: int, registry=None) -> np.ndarray:
        if self._parent_id is None:
            return np.full((n, 1), 0, dtype=np.uint32)  # 0 = no parent
        return np.full((n, 1), self._parent_id, dtype=np.uint32)
    
    @staticmethod
    def register(store):
        store.register_component("Parent", np.dtype("u4"), shape=(1,))
```

**Storage**: Single uint32 per entity (0 = root/no parent)

---

### Matrix Computation Strategy

#### Option A: Iterative Depth-First (Naive but Correct)

```python
def compute_world_matrices(store, indices, max_depth=8):
    """Compute world matrices with hierarchy."""
    n = len(indices)
    local_matrices = compute_local_matrices(store, indices)  # (N, 16)
    world_matrices = local_matrices.copy()
    
    # Get parent component data
    parent_data = store.get_component_data("Parent", indices)  # (N, 1)
    
    # Build index mapping: entity_idx -> local index in our arrays
    index_map = {idx: i for i, idx in enumerate(indices)}
    
    # Iteratively apply parent transforms (up to max_depth iterations)
    for depth in range(max_depth):
        changed = False
        for i, idx in enumerate(indices):
            parent_id = parent_data[i, 0]
            if parent_id == 0:  # No parent
                continue
            if parent_id in index_map:
                parent_local_idx = index_map[parent_id]
                # world = parent_world @ local
                world_matrices[i] = matmul(world_matrices[parent_local_idx], local_matrices[i])
                changed = True
        if not changed:
            break
    
    return world_matrices
```

**Issues**: 
- O(N * depth) - but each entity processed multiple times
- Not easily vectorizable due to irregular parent relationships

---

#### Option B: Topological Sort (Optimal)

Sort entities so parents come before children, then single-pass computation:

```python
def compute_world_matrices_hierarchical(store, indices, max_depth=8):
    """Compute world matrices with hierarchy using topological sort."""
    n = len(indices)
    local_matrices = compute_local_matrices(store, indices)
    world_matrices = np.zeros((n, 16), dtype=np.float32)
    
    parent_data = store.get_component_data("Parent", indices)
    
    # Build adjacency for topological sort
    # For each entity, track its children
    index_to_local = {idx: i for i, idx in enumerate(indices)}
    children = [[] for _ in range(n)]
    has_parent = np.zeros(n, dtype=bool)
    
    for i, idx in enumerate(indices):
        parent_id = parent_data[i, 0]
        if parent_id != 0 and parent_id in index_to_local:
            parent_local = index_to_local[parent_id]
            children[parent_local].append(i)
            has_parent[i] = True
    
    # DFS from roots to compute in correct order
    def compute_subtree(local_idx):
        local_mat = local_matrices[local_idx]
        if has_parent[local_idx]:
            # Find parent - need to look up parent entity ID
            parent_id = parent_data[local_idx, 0]
            parent_local = index_to_local[parent_id]
            parent_world = compute_subtree(parent_local)
            world = matmul(parent_world, local_mat)
        else:
            world = local_mat
        world_matrices[local_idx] = world
        # Process children
        for child_idx in children[local_idx]:
            compute_subtree(child_idx)
    
    # Start from roots (entities without parents in our set)
    roots = [i for i in range(n) if not has_parent[i]]
    for root in roots:
        compute_subtree(root)
    
    return world_matrices
```

**Issues**: 
- Recursive - stack depth issues for deep hierarchies
- Python function call overhead

---

#### Option C: Vectorized Depth Iteration (Recommended)

Use numpy operations with explicit depth tracking:

```python
def compute_world_matrices_vectorized(store, indices, max_depth=8):
    """
    Compute world matrices with hierarchy using vectorized depth iteration.
    
    Strategy:
    1. Compute all local matrices
    2. For each depth level, apply parent transforms for entities whose 
       parent has already been processed at this depth or shallower
    3. Iterate until no changes or max_depth reached
    """
    n = len(indices)
    local_matrices = compute_local_matrices(store, indices)
    world_matrices = local_matrices.copy()  # Start with local (roots get correct world)
    
    parent_data = store.get_component_data("Parent", indices)  # (N, 1)
    
    # Build entity_id -> local_index mapping
    index_map = {idx: i for i, idx in enumerate(indices)}
    
    # For efficient lookup, create array: entity_id -> local_index
    # Use max_entities to size it
    max_ent = store.max_entities
    local_index_of = np.full(max_ent, -1, dtype=np.int32)
    local_index_of[indices] = np.arange(n)
    
    # Track which entities need processing
    needs_update = np.ones(n, dtype=bool)
    
    for depth in range(max_depth):
        if not needs_update.any():
            break
        
        # Get parent local indices for all entities that need update
        parent_ids = parent_data[needs_update, 0]  # Parent entity IDs
        valid_parents = parent_ids > 0
        
        # Convert parent IDs to local indices
        parent_local = local_index_of[parent_ids]
        parent_valid = (parent_local >= 0) & valid_parents
        
        # For entities with valid parents, multiply parent_world @ local
        # This is the core operation we need to vectorize
        update_mask = needs_update & parent_valid
        if not update_mask.any():
            break
        
        # Get indices to update
        update_indices = np.where(update_mask)[0]
        parent_indices = parent_local[update_mask]
        
        # Batch matrix multiply: world[child] = world[parent] @ local[child]
        for i, (child_idx, parent_idx) in enumerate(zip(update_indices, parent_indices)):
            world_matrices[child_idx] = matmul(world_matrices[parent_idx], local_matrices[child_idx])
        
        # These entities now have correct world matrices
        needs_update[update_indices] = False
    
    return world_matrices
```

**Issues**: Still has Python loop - can we do better?

---

#### Option D: Batched Parent Application (Most Vectorizable)

```python
def compute_world_matrices_batched(store, indices, max_depth=8):
    """
    Compute world matrices with fully vectorized parent application.
    
    Uses broadcasting to apply parent transforms at each depth level.
    """
    n = len(indices)
    local_matrices = compute_local_matrices(store, indices)
    world_matrices = local_matrices.copy()
    
    parent_data = store.get_component_data("Parent", indices)
    
    # Create mapping from entity ID to local index
    max_ent = store.max_entities
    local_index_of = np.full(max_ent, -1, dtype=np.int32)
    local_index_of[indices] = np.arange(n)
    
    # Convert parent IDs to local indices (0 = no parent -> -1)
    parent_local = np.where(
        parent_data[:, 0] > 0,
        local_index_of[parent_data[:, 0]],
        -1
    )
    
    # Track which rows need update (those with valid parent)
    needs_update = parent_local >= 0
    
    for depth in range(max_depth):
        if not needs_update.any():
            break
            
        # Get pairs to multiply: (child_idx, parent_idx)
        child_indices = np.where(needs_update)[0]
        parent_indices = parent_local[child_indices]
        
        # Extract parent world matrices and child local matrices
        parent_world = world_matrices[parent_indices]  # (K, 16)
        child_local = local_matrices[child_indices]   # (K, 16)
        
        # Matrix multiply: result[i] = parent_world[i] @ child_local[i]
        # Reshape to (K, 4, 4) for batch multiply
        pw = parent_world.reshape(-1, 4, 4)  # (K, 4, 4)
        cl = child_local.reshape(-1, 4, 4)   # (K, 4, 4)
        
        # Batch matrix multiply using einsum
        # result[i] = pw[i] @ cl[i]
        # (4,4) @ (4,4) -> (4,4) for each i
        result = np.einsum('ijk,ikl->ijl', pw, cl).reshape(-1, 16)
        
        world_matrices[child_indices] = result
        
        # These entities now have correct world, no longer need update
        needs_update[child_indices] = False
    
    return world_matrices
```

**This is the winner!** 
- Fully vectorized within each depth iteration
- Uses numpy einsum for batch matrix multiplication
- Clear termination (no more updates or max depth)

---

### Matrix Multiplication Implementation

```python
def matmul_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Batch multiply A @ B where each is (N, 16) reshaped to (N, 4, 4).
    Returns (N, 16) result.
    """
    A_4x4 = A.reshape(-1, 4, 4)
    B_4x4 = B.reshape(-1, 4, 4)
    # result[i,j,k] = sum_l A[i,j,l] * B[i,l,k]
    return np.einsum('ijk,ikl->ijl', A_4x4, B_4x4).reshape(-1, 16)
```

Note: WGSL uses column-major, but our numpy matrices are row-major. Need to transpose for upload:

```python
# For upload to GPU (column-major for WGSL):
world_matrices_t = world_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
```

---

## Memory Layout

### New Component

| Field | Type | Size |
|-------|------|------|
| parent_id | uint32 | 4 bytes |

### Storage

- `Parent` component array: (max_entities, 1) uint32
- Default: 0 (no parent / root entity)

---

## Edge Cases

1. **Cycle detection**: Parent-child cycles should be prevented
   - Option A: Validate on parent assignment, reject cycles
   - Option B: Let it happen, max_depth limits infinite loops
   
2. **Orphan adoption**: When parent is destroyed, children become roots
   - Need system to clear parent IDs when parent is destroyed

3. **Depth limit**: Entities at depth > max_depth get incorrect transforms
   - Option A: Clamp to max_depth (might be wrong but not broken)
   - Option B: Warn/raise error if hierarchy too deep

4. **Parent outside render set**: Parent might not be in current indices
   - Only compute hierarchy for entities in `indices`
   - If parent not in indices, treat as root

---

## API Design

### Setting Parent

```python
# Create hierarchy
engine.spawn(
    Transform(pos=(0, 0, 0)),
    Parent(parent_id=None),  # Root
)
parent_idx = ...

child_idx = engine.spawn(
    Transform(pos=(1, 0, 0)),  # Local position relative to parent
    Parent(parent_id=parent_idx),
    Mesh(cube),
    Material(standard(color="#ff0000")),
)
```

### Query with Hierarchy

```python
# Get all children of a specific parent
def get_children(store, parent_id):
    parent_data = store.get_component_data("Parent", np.arange(store.max_entities))
    return np.where(parent_data[:, 0] == parent_id)[0]
```

### System Example

```python
@engine.system
def animate_joint(query: mx.Query[Transform, Parent], dt: float):
    # Modify local transforms - world matrices recomputed automatically
    query[Transform].rot += Transform.rotation(y=dt)
```

---

## Files to Modify

1. **`src/manifoldx/components.py`** - Add `Parent` component class
2. **`src/manifoldx/engine.py`** - Register `Parent` component
3. **`src/manifoldx/renderer.py`** - Update `TransformCache` with hierarchy support
4. **`src/manifoldx/resources.py`** - (No changes needed)

---

## Performance Considerations

| Aspect | Complexity | Notes |
|--------|------------|-------|
| Depth 0 (roots only) | O(N) | Same as current |
| Each additional depth | O(N) | Vectorized batch multiply |
| Total worst case | O(N * max_depth) | With max_depth=8, ~8x current |
| Memory | +4 bytes/entity | Parent component |

**Optimization**: Cache invalidation
- If entity's local transform changes → mark dirty
- If parent's world transform changes → children need recompute
- Can propagate dirty flag up to children

---

## Implementation Steps

1. **Add Parent component** - Simple uint32 component
2. **Modify TransformCache** - Add hierarchy-aware matrix computation
3. **Add batch matrix multiply** - Using numpy einsum
4. **Update renderer** - Pass store to TransformCache for Parent data
5. **Add API convenience** - `entity.set_parent(parent_entity)` method
6. **Test** - Verify hierarchy works correctly

---

## References

- Current transform cache: `src/manifoldx/renderer.py` lines 63-149
- Component registration: `src/manifoldx/components.py`
- Matrix upload: `renderer.py` lines 433-438 (transpose for WGSL)
- NumPy einsum: `np.einsum('ijk,ikl->ijl', A, B)` for batch matmul