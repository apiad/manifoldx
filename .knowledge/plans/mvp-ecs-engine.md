# Plan: MVP ECS Rendering Engine

**Status: PLANNING**

## Scope

Minimal viable implementation of the ECS architecture from `examples/cubes.py`:

- Single-threaded systems (thread-local later)
- Commands collected at `engine._commands` (thread-local later)
- GPU resources (Mesh/Material) created on first use
- CPU processing until final GPU upload
- Huge NumPy array: `(max_entities, n_components)` + alive/dead column
- Default implicit camera (refine later)

## Key Classes to Implement

### 1. Entity Store (Data Layer)
- `max_entities`: Pre-allocated capacity (e.g., 100,000)
- `n_components`: Known at registration time
- Single NumPy array `data[n_entities, n_components]` 
- First column = `alive` (bool)
- Each component has a fixed stride in the array

### 2. Component Registration (`@engine.component`)
- Decorator to register component types
- Store: name, dtype, shape per component
- Create column in entity store

### 3. Spawn System (`engine.spawn`)
- Accept component data arrays
- Find first dead slot (or expand)
- Write data to entity store
- Mark as alive

### 4. Query System (`mx.Query[...]`)
- Type-based filter for alive entities with certain components
- Return view with slicing/masking

### 5. System Decorator (`@engine.system`)
- Decorator to register update functions
- Decorator receives Query type annotation
- Systems execute in registration order

### 6. Command Buffer
- Simple list of command tuples
- Commands: `("set_transform", entity_id, matrix)`, `("draw", batch_key, count)`, etc.
- Cleared after each frame execution

### 7. GPU Resources
- Lazy initialization: mesh/material created on first spawn using them
- Cache: reuse existing GPU resources by ID
- Store device reference in Engine

### 8. Render System (Implicit)
- Runs after all user systems
- Groups alive entities by (mesh_id, material_id)
- Issues draw commands to command buffer
- At frame end: upload transforms, execute commands

## File Structure

```
src/manifoldx/
├── __init__.py
├── engine.py          # Engine class (exists, expand)
├── ecs.py             # EntityStore, Query, component decorators
├── commands.py       # Command buffer
├── resources.py      # GPU resource cache (Mesh, Material)
├── components.py     # Built-in components (Transform, Mesh, Material)
├── systems.py        # System runner
├── renderer.py      # Render system (implicit)
└── types.py         # Type definitions
```

## Implementation Order

### Phase 1: ECS Foundation
1. Define component types in `types.py`
2. Implement `EntityStore` in `ecs.py`
3. Implement `@engine.component` decorator
4. Implement `engine.spawn()` method

### Phase 2: Query & Systems  
5. Implement `Query` class
6. Implement `@engine.system` decorator
7. Implement system runner

### Phase 3: Commands & Rendering
8. Implement `CommandBuffer`
9. Implement built-in components (Transform, Mesh, Material)
10. Implement implicit RenderSystem
11. Hook into Engine.run()

### Phase 4: Integration
12. Wire up GPU resource creation on first spawn
13. Implement draw command execution
14. Test with `examples/cubes.py`

## Success Criteria

1. `examples/cubes.py` runs without modification
2. 1000 cubes spawn and animate
3. Cubes die and respawn
4. Frame renders with instancing (single draw call for all cubes)
5. Clean shutdown on window close

## Notes

- Ignore memoryview/mmap for now (use numpy directly)
- Single-threaded execution only
- No explicit camera (use default)
- Max entities pre-allocated (no dynamic expansion in MVP)