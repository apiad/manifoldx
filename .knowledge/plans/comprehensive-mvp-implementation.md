# Comprehensive Implementation Plan: MVP ECS Rendering Engine

**Reference**: Pygfx architecture (`lib/pygfx/`)

---

## Overview

This plan details every class and method needed to implement the MVP ECS engine that can run `examples/cubes.py`.

---

## File 1: `src/manifoldx/types.py`

### Purpose
Type definitions used across the engine.

### Classes/Functions

```python
# Pygfx reference: pygfx/utils/color.py (Color class), uses sRGB
# Pygfx reference: pygfx/utils/array_from_shadertype for typed arrays

class Vector3(np.ndarray): ...      # 3D vector, used for positions
class Vector4(np.ndarray): ...      # 4D vector (for colors)
class Float(float): ...              # Scalar float type marker
class Color: ...                    # RGBA color wrapper

# Type registry for components
COMPONENT_TYPES: dict[str, type] = {}

def register_component(name: str, dtype: np.dtype):
    """Register component type for ECS storage."""
    pass
```

### Implementation Notes
- Use `np.ndarray` subclass for vector types to enable vectorized ops
- `Color` wraps RGBA values, converts sRGB ↔ linear
- Registry maps component name → numpy dtype

---

## File 2: `src/manifoldx/ecs.py`

### Purpose
Core Entity Component System: EntityStore, Query, component decorators.

### Classes/Functions

#### 2.1 `EntityStore`

```python
class EntityStore:
    """
    Central storage for all entity data.
    
    Data layout: [alive:bool, component_0, component_1, ...]
    Each component has fixed stride in the array.
    """
    
    def __init__(self, max_entities: int = 100_000):
        self.max_entities = max_entities
        self._components: dict[str, tuple[int, np.dtype]] = {}  # name → (stride, dtype)
        self._data: np.ndarray = None  # Allocated on first component registration
        self._alive: np.ndarray = None  # Alias to first column
        
    def register_component(self, name: str, dtype: np.dtype):
        """Register a component type, allocate column."""
        # Add column to data array
        # Pygfx reference: resources/_buffer.py uses Buffer with structured dtypes
        pass
    
    def spawn(self, n: int, **component_data) -> np.ndarray:
        """
        Spawn n entities, return indices.
        
        component_data: {component_name: np.ndarray}
        Each array must be (n, component_size) or broadcastable.
        """
        # Find first n dead slots (alive == False)
        # Write component data to respective columns
        # Mark as alive
        # Return entity indices
        pass
    
    def destroy(self, indices: np.ndarray):
        """Mark entities as dead (alive = False)."""
        self._alive[indices] = False
    
    def get_component_view(self, component_names: list[str]) -> ComponentView:
        """Return view for querying entities."""
        pass
```

**Pygfx Reference**: `pygfx/objects/_base.py` uses `IdProvider` for object IDs. We use index-based.

#### 2.2 `ComponentView`

```python
class ComponentView:
    """
    View into entity store for specific components.
    
    IMPORTANT: Read operations return CURRENT data.
    Write operations NEVER modify component data directly.
    Instead, they emit Update commands with the computed NEW values.
    
    All modifications are accumulated and executed at end of frame.
    """
    
    def __init__(self, store: EntityStore, command_buffer: CommandBuffer,
                 component_names: list[str], indices: np.ndarray):
        self._store = store
        self._commands = command_buffer  # Append updates here
        self._components = component_names
        self._indices = indices  # Which entities this view represents
        self._len = len(indices)
        
    def __getitem__(self, component_cls) -> ComponentAccessor:
        """Get accessor for reading component data (READ-ONLY, returns current values)."""
        return ComponentAccessor(self._store, self._commands, component_cls, self._indices)
    
    # NO __setitem__ - updates are done via operator overload
    
    def __len__(self) -> int:
        return self._len
    
    def __iter__(self):
        """Iterate over entity indices."""
        for idx in self._indices:
            yield idx


class ComponentAccessor:
    """
    Accessor for a specific component within a view.
    
    Supports:
    - Reading: returns current component data
    - Augmented assignment (+=, -=, *=, /=): computes new values and EMITS Update command
    """
    
    def __init__(self, store: EntityStore, command_buffer: CommandBuffer,
                 component_name: str, indices: np.ndarray):
        self._store = store
        self._commands = command_buffer
        self._component_name = component_name
        self._indices = indices
        
    def __getitem__(self, key):
        """Read: returns current data for entities at indices."""
        return self._store.get_component_data(self._component_name, self._indices, key)
    
    def __setitem__(self, key, value):
        """Write: NOT ALLOWED - use augmented assignment instead."""
        raise NotImplementedError("Use +=, -=, *=, /= for updates")
    
    def __iadd__(self, other):
        """Augmented assignment: computes new_value = current + other, emits Update command."""
        # 1. Read current data
        current = self._store.get_component_data(self._component_name, self._indices)
        
        # 2. Compute new data (including broadcast if other is scalar)
        if np.isscalar(other):
            new_data = current + other
        else:
            new_data = current + other  # Vectorized
            
        # 3. Emit Update command with new data
        self._commands.append(Command(
            CommandType.UPDATE_COMPONENT,
            (self._component_name, self._indices.copy(), new_data.copy())
        ))
        return self
    
    def __isub__(self, other):  # -= same as += with -other
        return self.__iadd__(-other if np.isscalar(other) else -other)
    
    def __imul__(self, other):  # *= same pattern
        current = self._store.get_component_data(self._component_name, self._indices)
        new_data = current * other
        self._commands.append(Command(
            CommandType.UPDATE_COMPONENT,
            (self._component_name, self._indices.copy(), new_data.copy())
        ))
        return self
    
    def __itruediv__(self, other):  # /= same pattern
        current = self._store.get_component_data(self._component_name, self._indices)
        new_data = current / other
        self._commands.append(Command(
            CommandType.UPDATE_COMPONENT,
            (self._component_name, self._indices.copy(), new_data.copy())
        ))
        return self
```

**Key Design**:
- `__getitem__` reads current data (does NOT modify store)
- Augmented operators (+=, -=, etc.) compute new value and emit Update command
- Component data is NEVER modified during system execution
- Updates are deferred to command execution phase at end of frame

**Example flow**:
```python
# In system: query['Transform'].position += velocity * dt
# 
# 1. __iadd__ is called
# 2. Reads current position array: shape (N, 3)
# 3. Computes: position + velocity * dt = new_position
# 4. Emits Command: UPDATE_COMPONENT(Transform, indices, new_position)
# 5. At end of frame: CommandExecutor processes update, modifies store._data
```

**Pygfx Reference**: `pygfx/utils/transform.py` uses numpy arrays for transforms, applies via ufuncs.

#### 2.3 Query System

```python
@dataclass
class Query:
    """Type annotation for systems. Specifies required components."""
    components: tuple[type, ...]
    
    def __class_getitem__(cls, components: tuple[type, ...]):
        return Query(components=components)

class QueryEngine:
    """Executes queries against EntityStore."""
    
    def execute(self, query: Query) -> ComponentView:
        # Find entities with ALL specified components
        # Return ComponentView for those entities
        pass
```

**Pygfx Reference**: No direct equivalent—pygfx uses scene graph traversal (`WorldObject.children`).

#### 2.4 Global Component Registry (No Engine Dependency)

```python
# Global registry - no Engine instance needed
_COMPONENT_REGISTRY: dict[str, 'ComponentDef'] = {}

class ComponentDef:
    """Definition of a registered component."""
    def __init__(self, name: str, dtype: np.dtype, shape: tuple[int, ...], 
                 default_value: np.ndarray | None = None):
        self.name = name
        self.dtype = dtype
        self.shape = shape  # e.g., (3,) for Vector3, () for scalar
        self.default_value = default_value

def component(cls: type) -> type:
    """
    Decorator to register a component class.
    
    Usage:
        @component
        class Cube:
            velocity: Vector3  # shape (3,)
            life: Float       # shape ()
    
    Registers component in global _COMPONENT_REGISTRY.
    """
    annotations = cls.__annotations__
    
    for field_name, field_type in annotations.items():
        # Map type hints to numpy dtype and shape
        shape = _infer_shape_from_type(field_type)  # e.g., Vector3 → (3,)
        dtype = _infer_dtype_from_type(field_type)  # f4 for float32
        
        # Get default from class if present
        default = getattr(cls, field_name, None)
        
        comp_def = ComponentDef(
            name=field_name,
            dtype=dtype,
            shape=shape,
            default_value=default
        )
        _COMPONENT_REGISTRY[field_name] = comp_def
        
    # Attach registry info to class for later reference
    cls._component_defs = _COMPONENT_REGISTRY
    return cls

def _infer_shape_from_type(tp) -> tuple[int, ...]:
    """Infer array shape from type hint."""
    if tp == Float or tp == float:
        return ()  # Scalar
    elif tp == Vector3:
        return (3,)
    elif tp == Vector4:
        return (4,)
    else:
        return ()  # Default to scalar

def _infer_dtype_from_type(tp) -> np.dtype:
    """Infer numpy dtype from type hint."""
    return np.dtype('f4')  # Default to float32
```

**Pygfx Reference**: Uses `typing.Annotated` for type hints, but we use custom decorator.

---

## File 3: `src/manifoldx/components.py`

### Purpose
Built-in components: Transform, Mesh, Material.

### Classes

#### 3.1 Transform Component

```python
@engine.component
class Transform:
    position: Vector3      # shape (3,), stored as 3 floats
    rotation: Vector4      # shape (4,), quaternion  
    scale: Vector3         # shape (3,), defaults to (1,1,1)

# Storage: 10 floats per entity (position 3 + rotation 4 + scale 3)
# On GPU: mat4x4<f32> (64 bytes, 16-byte aligned)
```

**Pygfx Reference**: `pygfx/utils/transform.py` uses `AffineTransform` with position, rotation, scale properties.

#### 3.2 Mesh Component

```python
@engine.component  
class Mesh:
    """Reference to a geometry resource."""
    geometry_id: int       # Index into GeometryRegistry
```

**Pygfx Reference**: `pygfx/objects/_base.py` stores `geometry` and `material` as attributes.

#### 3.3 Material Component

```python
@engine.component
class Material:
    """Reference to a material resource."""
    material_id: int       # Index into MaterialRegistry
```

---

## File 4: `src/manifoldx/resources.py`

### Purpose
GPU resource management: Geometry, Material caches.

### Classes

#### 4.1 GeometryRegistry

```python
class GeometryRegistry:
    """
    Cache of GPU geometry resources.
    
    Lazily creates WebGPU buffers on first use.
    """
    
    def __init__(self, device):
        self._device = device
        self._geometries: dict[int, GpuGeometry] = {}  # id → GPU resource
        self._next_id = 1
        
    def register(self, geometry_obj) -> int:
        """Register a geometry, return ID."""
        # Create GPU buffers (vertex, index) lazily
        pass
        
    def get(self, geometry_id: int) -> GpuGeometry:
        """Get GPU geometry by ID."""
        pass

class GpuGeometry:
    """GPU resources for a geometry."""
    vertex_buffer: wgpu.Buffer
    index_buffer: wgpu.Buffer | None
    vertex_count: int
    index_count: int
```

**Pygfx Reference**: `pygfx/geometries/_base.py` Geometry class. `pygfx/resources/_buffer.py` Buffer class wraps wgpu buffers.

#### 4.2 MaterialRegistry

```python
class MaterialRegistry:
    """Cache of GPU material pipelines."""
    
    def __init__(self, device):
        self._device = device
        self._materials: dict[int, GpuMaterial] = {}
        self._next_id = 1
        
    def register(self, material_obj) -> int:
        """Register material, create pipeline lazily."""
        pass
    
    def get(self, material_id: int) -> GpuMaterial:
        pass

class GpuMaterial:
    """GPU pipeline and bind groups for a material."""
    pipeline: wgpu.RenderPipeline
    bind_group: wgpu.BindGroup
    uniform_buffer: wgpu.Buffer
```

**Pygfx Reference**: `pygfx/materials/_base.py` Material class. `pygfx/renderers/` contains pipeline creation.

#### 4.3 Built-in Geometry Factories

```python
def cube(width: float, height: float, depth: float) -> Geometry:
    """Create cube geometry."""
    # Generate positions, normals, indices
    # Returns Geometry object (not GPU resource)
    pass

def sphere(radius: float, segments: int = 32) -> Geometry:
    pass

def plane(width: float, height: float) -> Geometry:
    pass
```

**Pygfx Reference**: `pygfx/geometries/_box.py`, `_sphere.py`, `_plane.py`.

#### 4.4 Built-in Material Factories

```python
def basic(color: Color) -> Material:
    """Unlit material."""
    pass

def phong(color: Color) -> Material:
    """Phong shading material."""
    pass

def standard(color: Color, roughness: float = 0.5, metallic: float = 0.0) -> Material:
    """PBR material."""
    pass
```

**Pygfx Reference**: `pygfx/materials/_mesh.py` has MeshBasicMaterial, MeshPhongMaterial, MeshStandardMaterial.

---

## File 5: `src/manifoldx/commands.py`

### Purpose
Command buffer for deferred execution.

**Key Design**: Commands are ONLY for system-side modifications:
- `SPAWN`: Create new entities
- `DESTROY`: Mark entities as dead
- `UPDATE_COMPONENT`: Apply computed values to component data

**Render Pipeline** (NOT commands) runs AFTER commands:
- Batch entities by (geometry, material)
- Compute transform matrices (with dirty-cache optimization)
- Upload to GPU
- Issue draw_instanced calls

### Classes

#### 5.1 CommandBuffer

```python
class CommandType:
    NOP = 0
    SPAWN = 1              # Create new entities with component data
    DESTROY = 2            # Mark entities as dead
    UPDATE_COMPONENT = 3   # Apply computed values to component arrays

@dataclass
class Command:
    type: int
    data: tuple  # Variable-length payload

@dataclass  
class SpawnCommand:
    """Command to spawn new entities."""
    n: int
    component_data: dict[str, np.ndarray]  # {name: data}

@dataclass
class DestroyCommand:
    """Command to destroy entities."""
    indices: np.ndarray

@dataclass
class UpdateCommand:
    """Command to update component data."""
    component_name: str
    indices: np.ndarray
    new_data: np.ndarray


class CommandBuffer:
    """Accumulator for frame commands."""
    
    def __init__(self, capacity: int = 10_000):
        self._commands: list[Command] = []
        self._capacity = capacity
        
    def append(self, cmd: Command):
        self._commands.append(cmd)
        
    def clear(self):
        self._commands.clear()
        
    def __len__(self) -> int:
        return len(self._commands)
        
    def execute(self, store: EntityStore):
        """Execute all commands in order, modifying entity store."""
        for cmd in self._commands:
            self._execute_command(cmd, store)
            
    def _execute_command(self, cmd: Command, store: EntityStore):
        match cmd.type:
            case CommandType.SPAWN:
                # Extract SpawnCommand data
                n = cmd.data['n']
                component_data = cmd.data['components']
                store._spawn_immediate(n, **component_data)
                
            case CommandType.DESTROY:
                # Extract DestroyCommand data
                indices = cmd.data['indices']
                store._alive[indices] = False
                # Optionally add to free list
                
            case CommandType.UPDATE_COMPONENT:
                # Extract UpdateCommand data
                component_name = cmd.data['component_name']
                indices = cmd.data['indices']
                new_data = cmd.data['new_data']
                store._components[component_name][indices] = new_data
                
            case CommandType.NOP:
                pass
```

**Execution Order**: Commands are executed in the order they were received.
- Spawn commands add new alive entities
- Update commands modify component arrays
- Destroy commands mark entities as dead

**Pygfx Reference**: No command buffer—pygfx renders immediately in `Renderer.render()`.

---

## File 6: `src/manifoldx/systems.py`

### Purpose
System registration and execution.

### Classes

#### 6.1 System Registry

```python
class System:
    def __init__(self, func, query: Query):
        self.func = func
        self.query = query
        
    def run(self, engine, dt: float):
        view = engine.store.get_component_view(self.query.components)
        self.func(view, dt)

class SystemRegistry:
    def __init__(self):
        self._systems: list[System] = []
        
    def register(self, func, query: Query) -> System:
        system = System(func, query)
        self._systems.append(system)
        return func  # Allow @decorator pattern
        
    def run_all(self, engine, dt: float):
        for system in self._systems:
            system.run(engine, dt)
```

**Pygfx Reference**: No ECS systems—pygfx uses animation loop callbacks (`gfx.show(..., before_render=...)`).

---

## File 7: `src/manifoldx/renderer.py`

### Purpose
Render pipeline that executes AFTER command buffer is processed.

**IMPORTANT**: This is NOT part of the command system. It runs after all commands
(spawn/destroy/update) have been applied to the entity store.

### Classes

#### 7.1 TransformCache (With Dirty Flag Optimization)

```python
class TransformCache:
    """
    Caches computed transform matrices.
    
    Stores: _matrix_cache (N, 16) - mat4x4<f32> per entity
            _dirty (N,) bool - needs recompute
    
    Only recomputes matrices when marked dirty (after Update command).
    """
    
    def __init__(self, max_entities: int):
        self._matrix_cache = np.zeros((max_entities, 16), dtype=np.float32)
        self._dirty = np.ones(max_entities, dtype=bool)  # All dirty initially
        
    def mark_dirty(self, indices: np.ndarray):
        """Mark entities as needing recompute."""
        self._dirty[indices] = True
        
    def get_transforms(self, store: EntityStore, indices: np.ndarray) -> np.ndarray:
        """
        Get transform matrices for indices, recomputing if dirty.
        Returns (len(indices), 16) array of mat4x4<f32>.
        """
        dirty_mask = self._dirty[indices]
        if dirty_mask.any():
            self._recompute_matrices(store, indices[dirty_mask])
            
        return self._matrix_cache[indices]
    
    def _recompute_matrices(self, store: EntityStore, indices: np.ndarray):
        """Compute mat4x4 from pos/rot/scale using vectorized numpy."""
        transform_data = store.get_component_data('Transform', indices)
        
        positions = transform_data['position']  # (N, 3)
        rotations = transform_data['rotation']  # (N, 4) quat
        scales = transform_data['scale']        # (N, 3)
        
        # Vectorized computation (use Numba for speed if needed)
        matrices = self._compute_batch_matrices(positions, rotations, scales)
        
        self._matrix_cache[indices] = matrices
        self._dirty[indices] = False


class RenderPipeline:
    """
    Render pipeline that executes after command buffer.
    
    Flow:
    1. Query all alive entities with Mesh + Material + Transform
    2. Group by (geometry_id, material_id) into batches
    3. For each batch: compute transforms (using cache), upload, draw
    """
    
    def __init__(self, store: EntityStore, device):
        self._store = store
        self._device = device
        self._transform_cache = TransformCache(store.max_entities)
        
    def run(self, engine, dt: float):
        """Execute full render pipeline."""
        
        # 1. Get all alive entities with required components
        all_indices = np.where(self._store._alive)[0]
        
        # 2. Get component data for each entity
        mesh_ids = self._store.get_component_data('Mesh', all_indices)
        material_ids = self._store.get_component_data('Material', all_indices)
        transform_indices = self._store.get_component_data('Transform', all_indices)
        
        # 3. Group by (geometry_id, material_id) → list of entity indices
        batches = self._group_by_material(mesh_ids, material_ids, all_indices)
        
        # 4. For each batch: compute transforms, upload, draw
        for (geo_id, mat_id), batch_indices in batches.items():
            self._render_batch(geo_id, mat_id, batch_indices)
            
    def _group_by_material(self, mesh_ids, material_ids, indices):
        """Group entity indices by (geometry_id, material_id)."""
        # Use numpy groupby or pandas for efficiency
        batch_key = np.column_stack([mesh_ids, material_ids])
        unique_keys, inverse = np.unique(batch_key, axis=0, return_inverse=True)
        
        batches = {}
        for i, key in enumerate(unique_keys):
            mask = inverse == i
            batches[tuple(key)] = indices[mask]
        return batches
        
    def _render_batch(self, geo_id, mat_id, indices):
        """Render a single batch with instancing."""
        
        # 1. Get transforms (uses cache, auto-recomputes if dirty)
        transforms = self._transform_cache.get_transforms(self._store, indices)
        
        # 2. Upload instance buffer to GPU
        instance_buffer = self._device.create_buffer(
            size=transforms.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
        )
        self._device.queue.write_buffer(instance_buffer, 0, transforms)
        
        # 3. Get geometry and material GPU resources
        gpu_geometry = self._geometry_registry.get(geo_id)
        gpu_material = self._material_registry.get(mat_id)
        
        # 4. Issue draw_instanced call
        render_pass.draw_instanced(
            gpu_geometry.vertex_count,
            len(indices),  # instance count
            0,
            0
        )
```

**Key Design Points**:
- Transform cache with dirty flags - only recomputes changed matrices
- Groups entities by (geometry, material) for instancing
- Runs AFTER command execution - uses updated component data
- NO commands emitted - direct GPU operations

**Pygfx Reference**: `pygfx/renderers/_base.py` Renderer class, iterates scene graph.

#### 7.2 Default Camera

```python
def create_default_camera(aspect_ratio: float = 16/9) -> Camera:
    """Create default perspective camera."""
    pass
```

**Pygfx Reference**: `pygfx/cameras/_base.py` PerspectiveCamera.

---

## File 8: `src/manifoldx/engine.py`

### Purpose
Main Engine class, wires everything together.

### Expanded Class

```python
class Engine:
    def __init__(self, name: str, h: int = 600, w: int = 800, max_entities: int = 100_000):
        # Existing window/WebGPU init...
        
        # NEW: ECS infrastructure
        self.store = EntityStore(max_entities)
        self.commands = CommandBuffer()
        self.systems = SystemRegistry()
        self.query_engine = QueryEngine()
        
        # NEW: Configurable timestep
        self._use_fixed_dt = False
        self._fixed_dt_value = 1/60  # 60 FPS default
        self._last_time = None
        
        # NEW: Resource registries
        self._geometry_registry = GeometryRegistry(self._device)
        self._material_registry = MaterialRegistry(self._device)
        
        # NEW: Built-in components (registered in global registry)
        self._register_builtin_components()
        
        # NEW: Default camera
        self._camera = create_default_camera(w/h)
        
    def _register_builtin_components(self):
        # Register Transform, Mesh, Material in global registry
        from manifoldx.components import Transform, Mesh, Material
        # Components are registered via @component decorator globally
        
    # NEW: Timestep configuration
    def set_fixed_timestep(self, dt: float):
        """Use fixed timestep for deterministic simulations."""
        self._use_fixed_dt = True
        self._fixed_dt_value = dt
        
    def use_wall_clock(self):
        """Use actual elapsed time (default for animations)."""
        self._use_fixed_dt = False
        
    def _compute_dt(self):
        """Compute delta time based on configuration."""
        if self._use_fixed_dt:
            return self._fixed_dt_value
        
        # Wall-clock timing
        current_time = perf_counter_ns()
        if self._last_time is None:
            self._last_time = current_time
            return self._fixed_dt_value  # First frame
            
        dt = (current_time - self._last_time) / 1_000_000_000  # ns → seconds
        self._last_time = current_time
        return dt
        
    # EXISTING: Decorators (global component system)
    # Note: @component is global decorator, not Engine method
    
    def system(self, func):
        """Decorator to register update function."""
        return self.systems.register(func, query_from_annotations(func))
        
    def spawn(self, *args, n: int, **kwargs):
        """Spawn entities by emitting SPAWN command."""
        # Parse args/kwargs into component data
        # Broadcast scalars to arrays
        for name, value in kwargs.items():
            if np.isscalar(value):
                kwargs[name] = np.full((n,), value, dtype=np.float32)
                
        # Emit SPAWN command (not executed immediately)
        self.commands.append(Command(
            CommandType.SPAWN,
            {'n': n, 'components': kwargs}
        ))
        
        # For Mesh/Material components: lazy GPU resource creation
        for component_name, value in kwargs.items():
            if component_name in ('Mesh', 'Material'):
                self._ensure_gpu_resource(component_name, value)
                
    def destroy(self, condition):
        """Destroy entities matching condition by emitting DESTROY command."""
        # Find indices matching condition (boolean mask)
        indices = self._evaluate_condition(condition)
        
        # Emit DESTROY command (not executed immediately)
        self.commands.append(Command(
            CommandType.DESTROY,
            {'indices': indices}
        ))
        
    def _evaluate_condition(self, condition):
        """Evaluate destroy condition to get entity indices."""
        # Could be: array of bools, or lambda function
        # Returns np.ndarray of indices to destroy
        pass
        
    # MODIFIED: run() method
    def run(self):
        self._init_webgpu()
        
        self._running = True
        self._last_time = perf_counter_ns()  # Initialize timing
        
        for callback in self._startup_callbacks:
            callback()
            
        while self._running:
            dt = self._compute_dt()
            
            # 1. Clear command buffer for this frame
            self.commands.clear()
            
            # 2. Run all user systems (they emit commands)
            #    Pass command buffer so systems can append updates
            self.systems.run_all(self, dt)
            
            # 3. Execute command buffer (apply all spawn/destroy/update)
            self.commands.execute(self.store)
            
            # 4. RENDER PIPELINE (NOT commands - separate execution)
            #    Runs AFTER commands to use updated data
            self._render_pipeline.run(self, dt)
            
            # 5. Render frame to screen
            self._render_frame()
            
            if self._render_canvas.is_closing:
                self._running = False
                
        for callback in self._shutdown_callbacks:
            callback()
```

**Key Changes**:
1. `@component` is global decorator, not Engine method
2. `spawn()` emits SPAWN command (not executed immediately)
3. `destroy()` emits DESTROY command
4. `set_fixed_timestep()` / `use_wall_clock()` for dt configuration
5. Render pipeline runs AFTER command execution

---

## File 9: `src/manifoldx/__init__.py`

### Purpose
Public API exports.

```python
from manifoldx.types import Vector3, Vector4, Float, Color
from manifoldx.ecs import EntityStore, Query, component
from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import cube, sphere, plane, basic, phong, standard
from manifoldx.engine import Engine

__all__ = [
    'Engine',
    'component',
    'Query',
    'types',
    'components',
    'geometry',
    'material',
    'colors',
]

# Module shortcuts
class _ModuleProxy:
    types = types_module
    components = components_module
    geometry = _GeometryModule()
    material = _MaterialModule()
    colors = _ColorsModule()

import manifoldx.types as types
import manifoldx.components as components
import manifoldx.resources as geometry
import manifoldx.resources as material
```

---

## Execution Flow (Complete Frame)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRAME N                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. CLEAR COMMAND BUFFER                                            │
│     └─> self.commands.clear()                                       │
│                                                                      │
│  2. RUN USER SYSTEMS                                                │
│     ├─ System A: query['Transform'].position += velocity * dt      │
│     │   └─> Computes new_position                                   │
│     │   └─> EMITS: Update(Transform, indices, new_position)        │
│     │                                                               │
│     ├─ System B: query['Cube'].life -= dt                          │
│     │   └─> EMITS: Update(Cube, indices, new_life)                 │
│     │                                                               │
│     └─ System C: engine.destroy(life <= 0)                         │
│         └─> EMITS: Destroy(indices)                                 │
│                                                                      │
│  3. EXECUTE COMMANDS                                                │
│     ├─ Spawn: Add new entities to _alive                            │
│     ├─ Update: Apply new values to _components[name]               │
│     └─ Destroy: Set _alive[indices] = False                        │
│                                                                      │
│  4. RENDER PIPELINE (AFTER commands)                                │
│     ├─ Query all alive entities with Mesh+Material+Transform        │
│     ├─ Group by (geometry_id, material_id) → batches                │
│     ├─ For each batch:                                              │
│     │   ├─ Get transforms (from cache, recompute if dirty)         │
│     │   ├─ Upload instance buffer to GPU                            │
│     │   └─ draw_instanced(vertex_count, instance_count)            │
│     │                                                               │
│  5. RENDER FRAME                                                    │
│     └─> _render_frame() → present to screen                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

| Priority | File | What to implement |
|----------|------|-------------------|
| 1 | `types.py` | Type definitions, Color class |
| 2 | `ecs.py` | EntityStore, ComponentView, Query, decorators |
| 3 | `components.py` | Built-in Transform, Mesh, Material |
| 4 | `resources.py` | Geometry/Material registries, GPU creation |
| 5 | `commands.py` | Command buffer |
| 6 | `systems.py` | System runner |
| 7 | `renderer.py` | Render system, default camera |
| 8 | `engine.py` | Wire everything together |
| 9 | `__init__.py` | Public API exports |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Fixed component stride | Fast numpy slicing |
| Lazy GPU creation | Don't create unused resources |
| Command buffer | Enable batching, defer execution |
| Index-based entities | Simple, cache-friendly |
| Built-in render system | Automatic rendering, no user code needed |

---

## Unit Testing Strategy

### Test File Structure

```
tests/
├── test_types.py           # Type tests
├── test_ecs.py             # EntityStore, Query, ComponentView
├── test_components.py      # Built-in components
├── test_resources.py       # Geometry/Material registries (mocked GPU)
├── test_commands.py        # Command buffer
├── test_systems.py         # System runner
├── test_renderer.py        # Render system (mocked)
├── test_engine.py          # Integration tests
└── conftest.py             # Shared fixtures
```

### Test Philosophy

- **No GPU required** for most tests - mock or skip GPU calls
- Test numpy operations, strides, data layout
- Test component registration, spawn, destroy
- Test command buffer accumulation
- Test system execution order

### Test Coverage Per File

#### `tests/test_types.py`

```python
def test_vector3_creation():
    v = Vector3([1, 2, 3])
    assert v.shape == (3,)

def test_vector4_creation():
    v = Vector4([1, 2, 3, 4])
    assert v.shape == (4,)

def test_vector3_addition():
    v1 = Vector3([1, 2, 3])
    v2 = Vector3([4, 5, 6])
    result = v1 + v2
    np.testing.assert_array_equal(result, [5, 7, 9])

def test_color_from_hex():
    c = Color("#ff0000")
    assert c.r == 1.0 and c.g == 0.0 and c.b == 0.0

def test_color_from_rgb():
    c = Color(1.0, 0.0, 0.0)
    assert c.r == 1.0

def test_color_to_linear():
    c = Color("#ff0000")  # sRGB
    linear = c.to_linear()
    # Check conversion (approximate)
    assert linear.r < 1.0  # Gamma correction

def test_color_to_srgb():
    c = Color(linear_r=0.2, linear_g=0.0, linear_b=0.0)
    srgb = c.to_srgb()
    assert srgb.r > 0.2
```

#### `tests/test_ecs.py`

```python
def test_entity_store_creation():
    store = EntityStore(max_entities=1000)
    assert store.max_entities == 1000

def test_register_component():
    store = EntityStore()
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    assert "velocity" in store._components
    # Check data shape
    assert store._components["velocity"].shape == (100000, 3)

def test_register_multiple_components():
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    # Check separate arrays per component (SoA)
    assert store._components["position"].shape == (100000, 3)
    assert store._components["velocity"].shape == (100000, 3)

def test_spawn_single():
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=1, position=[[1, 2, 3]])
    assert len(indices) == 1
    assert store._alive[indices[0]] == True

def test_spawn_multiple():
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=100, position=np.random.rand(100, 3))
    assert len(indices) == 100

def test_spawn_reuses_dead():
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    # Spawn and kill
    idx1 = store.spawn(n=1, position=[[1, 2, 3]])
    store.destroy(idx1)
    # Spawn again - should reuse slot
    idx2 = store.spawn(n=1, position=[[4, 5, 6]])
    assert idx2[0] == idx1[0]

def test_destroy():
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=10, position=np.random.rand(10, 3))
    store.destroy(indices[:5])
    assert np.all(store._alive[indices[:5]] == False)
    assert np.all(store._alive[indices[5:]] == True)

def test_query_basic():
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    
    # Spawn with both components
    idx = store.spawn(n=5, position=np.zeros((5, 3)), velocity=np.ones((5, 3)))
    
    view = store.get_component_view(['position', 'velocity'])
    assert len(view) == 5

def test_component_view_getitem():
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=10, position=np.random.rand(10, 3))
    
    view = store.get_component_view(['position'])
    pos_data = view[Transform]  # or store components
    assert pos_data.shape == (10, 3)

def test_component_view_iadd_emits_command():
    """Test that += operation emits Update command."""
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    commands = CommandBuffer()
    
    indices = store.spawn(n=10, position=np.zeros((10, 3)))
    view = ComponentView(store, commands, ['position'], indices)
    
    # Do += operation
    view['position'].__iadd__(np.ones((10, 3)) * 5)
    
    # Should emit Update command
    assert len(commands) == 1
    cmd = commands._commands[0]
    assert cmd.type == CommandType.UPDATE_COMPONENT
    assert cmd.data['component_name'] == 'position'
    np.testing.assert_array_equal(cmd.data['new_data'], np.ones((10, 3)) * 5)

def test_component_view_iadd_broadcast():
    """Test scalar broadcast during +=."""
    store = EntityStore()
    store.register_component("scale", np.dtype("f4"), shape=(3,))
    commands = CommandBuffer()
    
    indices = store.spawn(n=100, scale=np.zeros((100, 3)))
    view = ComponentView(store, commands, ['scale'], indices)
    
    # Scalar += should broadcast
    view['scale'].__iadd__(2.0)
    
    cmd = commands._commands[0]
    # New data should be broadcast to all 100 entities
    assert cmd.data['new_data'].shape == (100, 3)
    np.testing.assert_array_equal(cmd.data['new_data'], np.full((100, 3), 2.0))

def test_command_execution_updates_store():
    """Test that executed Update commands modify store data."""
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=5, position=np.zeros((5, 3)))
    
    # Manually add update command
    commands = CommandBuffer()
    commands.append(Command(
        CommandType.UPDATE_COMPONENT,
        {'component_name': 'position', 'indices': indices, 'new_data': np.ones((5, 3))}
    ))
    
    # Execute commands
    commands.execute(store)
    
    # Verify store was updated
    np.testing.assert_array_equal(store._components['position'][indices], np.ones((5, 3)))

def test_spawn_broadcast_scalar():
    store = EntityStore()
    store.register_component("scale", np.dtype("f4"), shape=(3,))
    
    # Spawn with scalar broadcast
    indices = store.spawn(n=100, scale=2.0)  # Broadcast to all
    view = store.get_component_view(['scale'])
    assert np.all(view['scale'] == 2.0)
```

#### `tests/test_components.py`

```python
def test_transform_dtype():
    store = EntityStore()
    Transform.register(store)
    
    # SoA: each component stored separately
    assert 'Transform' in store._components
    data = store._components['Transform']
    assert data.shape[1] == 10  # pos(3) + rot(4) + scale(3)

def test_transform_default_values():
    store = EntityStore()
    Transform.register(store)
    
    indices = store.spawn(n=10, Transform=None)  # Use defaults
    view = store.get_component_view(['Transform'])
    
    # Check default position is (0, 0, 0)
    pos_data = view['Transform']['position']
    assert np.all(pos_data == 0)

def test_mesh_reference():
    store = EntityStore()
    Mesh.register(store)
    
    # Verify it stores just an ID
    data = store._components['Mesh']
    assert data.shape[1] == 1  # Single uint32

def test_material_reference():
    store = EntityStore()
    Material.register(store)
    data = store._components['Material']
    assert data.shape[1] == 1  # Single uint32


#### `tests/test_transform_cache.py`

```python
def test_transform_cache_dirty_flag():
    """Test that cache tracks dirty entities."""
    cache = TransformCache(max_entities=100)
    
    # Initially all dirty
    assert np.all(cache._dirty == True)
    
    # Mark some as clean
    cache._dirty[5:10] = False
    assert np.all(cache._dirty[:5] == True)
    assert np.all(cache._dirty[5:10] == False)
    
def test_transform_cache_mark_dirty():
    """Test mark_dirty sets dirty flag."""
    cache = TransformCache(max_entities=100)
    indices = np.array([0, 5, 10])
    
    cache.mark_dirty(indices)
    
    assert cache._dirty[0] == True
    assert cache._dirty[5] == True
    assert cache._dirty[10] == True
    assert cache._dirty[1] == False  # Unchanged
    
def test_transform_cache_recompute():
    """Test matrix recomputation from pos/rot/scale."""
    store = EntityStore()
    store.register_component('Transform', ...)
    indices = store.spawn(n=3, Transform=...)
    
    cache = TransformCache(max_entities=100)
    cache._dirty[indices] = True
    
    # Get transforms - should recompute
    result = cache.get_transforms(store, indices)
    
    assert result.shape == (3, 16)  # 3 matrices, 16 floats each
    # First entity at origin should be identity
    np.testing.assert_array_almost_equal(result[0], np.eye(4).flatten())
```

#### `tests/test_resources.py` (GPU-mocked)

```python
def test_geometry_registry_id_allocation():
    class MockDevice:
        pass
    
    registry = GeometryRegistry(MockDevice())
    geo1 = object()
    geo2 = object()
    
    id1 = registry.register(geo1)
    id2 = registry.register(geo2)
    
    assert id1 != id2
    assert id1 == 1
    assert id2 == 2

def test_geometry_cache():
    registry = GeometryRegistry(MockDevice())
    geo = object()
    
    id1 = registry.register(geo)
    id2 = registry.register(geo)  # Same object
    
    assert id1 == id2  # Should return cached ID

def test_material_registry_pipeline_creation():
    # Mock device doesn't create real pipelines
    pass
```

#### `tests/test_commands.py`

```python
def test_command_buffer_append():
    buf = CommandBuffer()
    cmd = Command(CommandType.UPDATE_COMPONENT, 
                  {'component_name': 'pos', 'indices': [0,1], 'new_data': np.ones((2,3))})
    buf.append(cmd)
    assert len(buf) == 1

def test_command_buffer_clear():
    buf = CommandBuffer()
    buf.append(Command(CommandType.NOP, {}))
    buf.append(Command(CommandType.NOP, {}))
    buf.clear()
    assert len(buf) == 0

def test_command_types():
    """Verify command types are correctly numbered."""
    assert CommandType.NOP == 0
    assert CommandType.SPAWN == 1
    assert CommandType.DESTROY == 2
    assert CommandType.UPDATE_COMPONENT == 3

def test_command_execution_order():
    """Test that commands execute in order: spawn → update → destroy."""
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    buf = CommandBuffer()
    
    # Spawn entities
    buf.append(Command(CommandType.SPAWN, 
                      {'n': 3, 'components': {'position': np.ones((3,3))}}))
    
    # Update position
    indices = np.array([0, 1, 2])
    buf.append(Command(CommandType.UPDATE_COMPONENT,
                      {'component_name': 'position', 'indices': indices, 
                       'new_data': np.full((3,3), 5.0)}))
    
    # Destroy
    buf.append(Command(CommandType.DESTROY, {'indices': indices}))
    
    # Execute
    buf.execute(store)
    
    # Final state: entities should be dead
    assert np.all(store._alive[indices] == False)
    
    # Position should have been updated BEFORE destroy
    # (destroy happens last, but we verify data was updated)
    # Note: after destroy, data is irrelevant, but intermediate 
    # execution order is: spawn → update → destroy
```

#### `tests/test_systems.py`

```python
def test_system_registration():
    registry = SystemRegistry()
    
    @system_func
    def my_system(view, dt):
        pass
    
    assert len(registry._systems) == 1

def test_system_execution_order():
    execution_order = []
    
    registry = SystemRegistry()
    
    @registry.register
    def system_a(view, dt):
        execution_order.append('a')
    
    @registry.register
    def system_b(view, dt):
        execution_order.append('b')
    
    # Mock engine with store and commands
    engine = MockEngine()
    registry.run_all(engine, 1/60)
    
    assert execution_order == ['a', 'b']

def test_system_receives_correct_query():
    # Test that query annotation is properly extracted
    pass
```

#### `tests/test_renderer.py` (GPU-mocked)

```python
def test_batch_grouping():
    """Test grouping entities by (geometry_id, material_id)."""
    store = EntityStore()
    store.register_component("Mesh", np.dtype("u4"), shape=(1,))
    store.register_component("Material", np.dtype("u4"), shape=(1,))
    
    # Spawn entities with different mesh/material combinations
    # 5 red cubes, 3 blue cubes, 2 red spheres
    entities = store.spawn(n=10, Mesh=[1,1,1,1,1,2,2,2,1,1], Material=[1,1,1,1,1,1,1,1,1,2])
    # entities 0-4: mesh=1, mat=1 (red cube)
    # entities 5-7: mesh=2, mat=1 (blue cube)  
    # entities 8: mesh=1, mat=1 (red cube)
    # entity 9: mesh=1, mat=2 (red sphere)
    
    # Manual test: group by material
    mesh_ids = store._components['Mesh']
    mat_ids = store._components['Material']
    
    # Should form 3 batches:
    # (mesh=1, mat=1): indices [0,1,2,3,4,8] = 6 entities
    # (mesh=2, mat=1): indices [5,6,7] = 3 entities
    # (mesh=1, mat=2): index [9] = 1 entity
    
    # Get unique combinations
    batch_keys = np.column_stack([mesh_ids[:10], mat_ids[:10]])
    unique, inverse = np.unique(batch_keys, axis=0, return_inverse=True)
    
    # Should have 3 unique keys
    assert len(unique) == 3
    
    # Count each group
    for i in range(3):
        count = np.sum(inverse == i)
        if i == 0:  # mesh=1, mat=1
            assert count == 6
        elif i == 1:  # mesh=2, mat=1
            assert count == 3
        else:  # mesh=1, mat=2
            assert count == 1
```

---

## Real-Time Framerate Requirements

### Timing Strategy

The engine must use actual wall-clock time for animations, not fixed dt:

```python
class Engine:
    def run(self):
        self._init_webgpu()
        
        self._running = True
        self._last_time = perf_counter_ns()
        
        for callback in self._startup_callbacks:
            callback()
            
        while self._running:
            # Calculate actual elapsed time
            current_time = perf_counter_ns()
            dt = (current_time - self._last_time) / 1_000_000_000  # ns → seconds
            self._last_time = current_time
            
            # ... rest of frame
```

### Frame Timing Tests

```python
# tests/test_engine.py

def test_frame_timing_accuracy():
    """Verify dt matches actual elapsed time within 1ms."""
    engine = TestEngine()  # Mock, no GPU
    
    frame_times = []
    
    @engine.system
    def dummy_system(view, dt):
        frame_times.append(dt)
    
    # Run for a few frames with artificial delays
    # NOTE: This is tricky to test accurately without real timing
    pass

def test_spawn_timing():
    """Spawning 10k entities should be fast."""
    import time
    store = EntityStore(max_entities=100_000)
    store.register_component("position", np.dtype("f4"), shape=(3,))
    
    start = time.perf_counter()
    for _ in range(100):
        store.spawn(n=100, position=np.random.rand(100, 3))
    elapsed = time.perf_counter() - start
    
    # Should complete in under 1 second for 10k spawns
    assert elapsed < 1.0

def test_destroy_timing():
    """Destroying entities should also be fast."""
    import time
    store = EntityStore(max_entities=100_000)
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=1000, position=np.random.rand(1000, 3))
    
    start = time.perf_counter()
    store.destroy(indices)
    elapsed = time.perf_counter() - start
    
    assert elapsed < 0.01  # 10ms for 1k destroys
```

### Animation Accuracy Tests

```python
def test_animation_consistency():
    """Same dt should produce same results regardless of frame count."""
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    
    # Spawn with velocity
    pos = np.array([[0, 0, 0]], dtype=np.float32)
    vel = np.array([[1, 0, 0]], dtype=np.float32)  # 1 unit/second
    
    indices = store.spawn(n=1, position=pos, velocity=vel)
    
    # Method 1: Single large step
    store1 = EntityStore()  # Fresh copy
    store1.register_components(...)
    store1.spawn(n=1, position=pos, velocity=vel)
    # Simulate 1 second with dt=1
    store1._data['position'] += store1._data['velocity'] * 1.0
    
    # Method 2: Many small steps
    store2 = EntityStore()
    store2.register_components(...)
    store2.spawn(n=1, position=pos, velocity=vel)
    for _ in range(100):
        store2._data['position'] += store2._data['velocity'] * 0.01
    
    # Results should be very close
    np.testing.assert_allclose(
        store1._data['position'],
        store2._data['position'],
        rtol=1e-5
    )
```

---

## Performance Benchmarks (Unit Tests)

```python
# tests/benchmarks/test_ecs_performance.py

def test_spawn_10k_entities_performance():
    """10k entities should spawn in under 100ms."""
    store = EntityStore(max_entities=100_000)
    store.register_component("position", np.dtype("f4"), shape=(3,))
    
    import time
    start = time.perf_counter()
    store.spawn(n=10_000, position=np.random.rand(10_000, 3))
    elapsed = time.perf_counter() - start
    
    assert elapsed < 0.1, f"Spawn took {elapsed:.3f}s, expected <0.1s"

def test_query_performance():
    """Query over 100k entities should complete in under 10ms."""
    store = EntityStore(max_entities=100_000)
    store.register_component("position", np.dtype("f4"), shape=(3,))
    
    # Spawn half entities
    store.spawn(n=50_000, position=np.random.rand(50_000, 3))
    
    import time
    start = time.perf_counter()
    view = store.get_component_view(['position'])
    elapsed = time.perf_counter() - start
    
    assert elapsed < 0.01, f"Query took {elapsed:.3f}s, expected <0.01s"

def test_numpy_vectorized_operations():
    """Vectorized position += velocity * dt should be fast."""
    n = 100_000
    position = np.random.rand(n, 3).astype(np.float32)
    velocity = np.random.rand(n, 3).astype(np.float32)
    dt = 1/60
    
    import time
    start = time.perf_counter()
    position += velocity * dt
    elapsed = time.perf_counter() - start
    
    assert elapsed < 0.01  # Should be ~1-2ms for 100k entities
```

---

## Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ecs.py -v

# Run with coverage
pytest tests/ --cov=manifoldx --cov-report=html

# Run benchmarks only
pytest tests/benchmarks/ -v

# Run fast tests only (skip slow ones)
pytest tests/ -v -m "not slow"
```

---

## Success Criteria (Per File)

1. **types.py**: Color can be created from hex/rgb, converts to linear. Tests pass.
2. **ecs.py**: `spawn(n=1000, Transform=..., Cube=...)` creates 1000 entities. Tests pass.
3. **components.py**: Transform has position/rotation/scale accessible via view. Tests pass.
4. **resources.py**: First spawn with Mesh creates GPU vertex buffer (mocked). Tests pass.
5. **commands.py**: Commands execute in order, GPU state updated. Tests pass.
6. **systems.py**: `@system` decorated functions run each frame. Tests pass.
7. **renderer.py**: All cubes render in 1 draw call (instancing). Tests pass.
8. **engine.py**: `examples/cubes.py` runs without modification. Tests pass.
9. **__init__.py**: All public APIs accessible. Tests pass.

### Additional Success Criteria

| Criterion | Target |
|-----------|--------|
| Spawn 10k entities | < 100ms |
| Query 100k entities | < 10ms |
| Vectorized 100k updates | < 10ms |
| Destroy 10k entities | < 10ms |
| Frame timing accuracy | ±1ms |
| Animation consistency | rtol < 1e-5 |

---

## Final Design Summary (After Corrections)

### Key Architectural Decisions

| Decision | Final Choice | Rationale |
|----------|--------------|-----------|
| **Component storage** | SoA (Structure of Arrays) | Separate arrays per component for cache efficiency |
| **GPU batching** | RenderPipeline reformatts for GPU | Collects SoA data, converts to instance buffers |
| **Transform caching** | Dirty flag pattern | Only recompute matrices when marked dirty |
| **Component updates** | NEVER write during systems | Compute delta, emit Update command, apply at end of frame |
| **@component decorator** | Global (no Engine dependency) | Simpler, works anywhere |
| **Timestep** | Configurable | `set_fixed_timestep()` or `use_wall_clock()` |
| **Commands** | spawn/destroy/update ONLY | Draw is part of render pipeline |

### Data Flow

```
System execution:
  view['component'] += value
    → computes new_data
    → emits Update(component, indices, new_data)
    → NO direct write to store

After all systems:
  command_buffer.execute(store)
    → applies updates to _components[name][indices]
    
Render pipeline (after commands):
  1. Query all alive entities
  2. Group by (geometry, material)
  3. Get transforms from cache (recompute if dirty)
  4. Upload instance buffer
  5. draw_instanced()
```

### Files to Implement

| File | Purpose |
|------|---------|
| `types.py` | Vector3, Vector4, Float, Color |
| `ecs.py` | EntityStore (SoA), ComponentView, Query |
| `components.py` | Transform (with cache), Mesh, Material, @component |
| `resources.py` | Geometry/Material registries |
| `commands.py` | CommandBuffer, SPAWN/DESTROY/UPDATE commands |
| `systems.py` | SystemRegistry, @system decorator |
| `renderer.py` | RenderPipeline, TransformCache |
| `engine.py` | Wire everything, configurable dt |