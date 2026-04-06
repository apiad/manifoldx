---
id: project-analysis-manifoldx
created: 2026-04-06
type: analysis
status: active
---

# Project Analysis: ManifoldX

## Executive Summary

ManifoldX is a real-time 3D rendering engine built on pure wgpu (WebGPU) with an Entity Component System (ECS) architecture. It provides a Pythonic API for GPU-accelerated 3D graphics with PBR (Physically Based Rendering) materials and lighting.

**Key Stats:**
- 150+ tests passing
- ~2500 lines of core code (src/manifoldx/)
- 4 example programs
- Pure wgpu backend (no PyGfx dependency for core)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Code                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐│
│   │   engine.spawn()    │  @engine.system   │  engine.camera        ││
│   │   - Transform       │  - Query[Transform, Mesh]  │  - orbit(), zoom()   ││
│   │   - Mesh            │  - view[Transform].pos +=  │  - fit()            ││
│   │   - Material        │                            │                     ││
│   └─────────────┘       └─────────────┘       └─────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Engine Core                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  EntityStore│  │CommandBuffer│  │SystemRegistry│  │ Camera       │  │
│  │  (ECS)     │  │ (queue)    │  │ (systems)   │  │ (view/proj)  │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Render Pipeline                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ GeometryRegistry │  │MaterialRegistry │  │  RenderPipeline    │ │
│  │ - cube()        │  │ - BasicMaterial  │  │  - WGSL shaders    │ │
│  │ - sphere()      │  │ - StandardMaterial│ │  - instanced draw │ │
│  │ - plane()      │  │                   │  │  - lights         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          wgpu (WebGPU)                               │
│  - Device, Queue, RenderPipeline, BindGroup                        │
│  - WGSL shaders (vertex + fragment)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. `engine.py` - Application Entry Point

**Responsibilities:**
- Initialize wgpu device, adapter, canvas
- Manage ECS store, command buffer, system registry
- Run game loop (update systems → execute commands → render)

**Key Classes:**
- `Engine` - Main application class

**Key Methods:**
```python
engine = mx.Engine("MyApp")           # Create engine
engine.spawn(n=100, ...)              # Spawn entities
@engine.system                        # Register system
def physics(query, dt): ...           # System function
engine.camera.orbit(45, 30)           # Control camera
engine.set_lights([PointLight, ...])  # Set external lights
engine.run()                          # Start game loop
```

---

### 2. `ecs.py` - Entity Component System

**Architecture:** Structure of Arrays (SoA) - each component is a contiguous numpy array, not interleaved per-entity.

**Key Classes:**
- `EntityStore` - Central storage for all entity data
- `_FieldView` - Writable view into component sub-fields (pos, rot, scale)
- `ComponentView` - Query result with operator overloads

**Data Layout:**
```python
# EntityStore internal:
self._components = {
    "Transform": np.zeros((100_000, 10), dtype=np.float32),  # pos(3) + rot(4) + scale(3)
    "Mesh":       np.zeros((100_000, 1), dtype=np.int32),    # geometry_id
    "Material":   np.zeros((100_000, 1), dtype=np.int32),    # material_id
}
self._alive = np.zeros(100_000, dtype=bool)  # Which entities exist
```

**Key Operations:**
- `spawn(n, **components)` - Create n entities, return indices
- `destroy(indices)` - Mark entities as dead
- `get_component_data(name, indices)` - Get numpy array slice
- `get_component_view(names)` - Get view for querying

---

### 3. `components.py` - Built-in Components

**Standard Components:**
- `Transform` - position (vec3), rotation (quaternion), scale (vec3)
- `Mesh` - reference to geometry (via GeometryRegistry ID)
- `Material` - reference to material (via MaterialRegistry ID)
- `Colors` - helper for component factories

**Component Interface:**
```python
class Transform:
    @staticmethod
    def register(store): ...  # Register with EntityStore
    
    @staticmethod
    def get_data(n, registry): ...  # Get default data for n entities
```

---

### 4. `systems.py` - System Execution

**Architecture:** Functional systems with component views

**Key Classes:**
- `System` - Wraps a system function with its component dependencies
- `Query` - Type annotation for specifying required components
- `SystemRegistry` - Executes all systems each frame

**System Signature:**
```python
@engine.system
def my_system(query: Query[Transform, Mesh], dt: float):
    # query[Transform] returns _FieldView with vectorized operations
    query[Transform].pos += velocity * dt
    query[Transform].rot += Transform.rotation(euler=angular * dt)
```

---

### 5. `renderer.py` - WebGPU Rendering

**Architecture:** Instanced rendering with material-specific pipelines

**Key Classes:**
- `TransformCache` - Optimized transform matrix computation (dirty flag pattern)
- `RenderPipeline` - Main rendering coordinator

**Pipeline Flow:**
```
1. Get all alive entities with Transform + Mesh + Material
2. Group by (geometry_id, material_type) → batch per (geo, mat)
3. Compute all transform matrices, upload to storage buffer once
4. For each batch:
   - Get/create pipeline for (geometry, material_type) shader
   - Upload material uniforms (per batch)
   - Upload lights once (shared)
   - draw_indexed(index_count, instance_count, first_instance=offset)
```

**Shaders:**
- `BasicMaterial` - Unlit flat color with simple diffuse
- `StandardMaterial` - Full PBR with GGX BRDF, multiple point lights

---

### 6. `resources.py` - GPU Resources

**Geometry:**
- `cube(w, h, d)` - Box geometry with normals
- `sphere(radius, segments)` - UV sphere with normals (CCW winding)
- `plane(w, h)` - Quad with normals

**Materials:**
- `Material` (abstract) - Base class with `_compile()` classmethod
- `BasicMaterial` - Unlit shader (binding slot 0)
- `StandardMaterial` - PBR shader (binding slot 1)
- `PhongMaterial` - Alias for BasicMaterial (backward compat)

**Lights (External, not in ECS):**
- `PointLight` - position, color, intensity, distance, decay
- `SpotLight` - position, direction, inner/outer cone angles
- `DirectionalLight` - direction, color, intensity

**Registries:**
- `GeometryRegistry` - Cache GPU buffers for geometries
- `MaterialRegistry` - Cache compiled material pipelines

---

### 7. `camera.py` - View/Projection

**Features:**
- Spherical coordinates (azimuth, elevation, distance)
- Perspective projection (WebGPU NDC)
- Orbit, pan, zoom, dolly controls
- Fit/fit_bounds for automatic framing

**Key Methods:**
```python
camera.orbit(d_azimuth, d_elevation)  # Rotate around target
camera.zoom(factor)                    # Divide distance by factor
camera.pan(dx, dy)                     # Move target perpendicular to view
camera.fit(radius, center, azimuth, elevation)  # Auto-frame sphere
```

---

## Data Flow

### Spawn Flow
```python
engine.spawn(
    Mesh(cube_geo),           # → geometry ID via registry
    Material(red_shiny),     # → material ID via registry  
    Transform(pos=(x,y,z)),  # → numpy array
    n=10,
)
```

1. Convert geometry/material objects to IDs via registries
2. Convert Transform to (n, 10) numpy array
3. `EntityStore.spawn()` writes to SoA arrays
4. Emit SPAWN command to command buffer

### Render Flow
```python
# Each frame:
1. systems.run_all(engine, dt)     # Execute all systems
2. commands.execute(store)         # Apply spawn/destroy/update
3. render_pipeline.run(engine, dt) # CPU-side prep
4. render_pipeline.render(engine, render_pass) # GPU commands
```

---

## Design Decisions

### 1. External Lights (not in ECS)

Lights are passed directly to the engine, not stored in the ECS. Rationale:
- Scene usually has few lights (<10), entity overhead not worth it
- Lights don't need per-frame component updates
- Easier to manage as a simple list

### 2. Material-Type Specific Pipelines

Each material type (BasicMaterial, StandardMaterial) compiles to its own WGSL shader. The render pipeline caches by `(geometry_id, material_type)`.

Rationale:
- Different materials need different shaders
- Can share pipeline if same (geo, mat) combo
- Per-instance material properties via uniform buffer

### 3. Instanced Rendering with Shared Transform Buffer

All transforms uploaded to a single storage buffer, each batch uses `first_instance` offset.

Rationale:
- `queue.write_buffer()` is async; per-batch writes would race
- Single upload + offsets = correct + efficient

### 4. PBR with GGX BRDF

StandardMaterial uses physically-based shading:
- GGX normal distribution
- Smith geometry function
- Fresnel-Schlick approximation
- Multiple point lights with inverse-square attenuation
- Tone mapping (Reinhard) + gamma correction

---

## Testing

- 150+ unit tests covering:
  - ECS operations (spawn, destroy, queries)
  - Component field views (vectorized ops)
  - Geometry generation (cube, sphere, plane)
  - Material compilation (WGSL generation)
  - Renderer pipeline (batching, transforms)
  - Light system (PointLight, SpotLight, DirectionalLight)
  - Camera (view/projection matrices)

---

## File Structure

```
src/manifoldx/
├── __init__.py      # Public API, module proxies
├── engine.py        # Application entry, game loop
├── ecs.py           # EntityStore, _FieldView, ComponentView
├── components.py    # Transform, Mesh, Material, Colors
├── systems.py       # System, Query, SystemRegistry
├── commands.py      # Command buffer, command types
├── renderer.py      # RenderPipeline, TransformCache, WGSL shaders
├── resources.py     # Geometries, Materials, Lights, Registries
├── camera.py        # Camera with orbit controls
├── types.py         # Vector3, Vector4, Float, Color

examples/
├── hello_world.py   # Minimal example
├── cube.py          # Single rotating cube
├── spheres.py      # Many spheres with physics
└── pbr_demo.py     # 3x2 grid with PBR + orbiting lights
```

---

## Dependencies

- `wgpu` - Pure WebGPU binding (no backend abstraction)
- `numpy` - All ECS data storage, vectorized operations
- `rendercanvas` - GLFW/canvas integration for window management

---

## Current Limitations

1. **Single draw call per (geometry, material_type)** - Can't render entities with different material instances in same draw (material params are uniform, not per-instance)
2. **No shadow mapping** - Planned for future
3. **No texture support** - Just solid colors + PBR properties
4. **No HDR/environment maps** - Basic tonemapping only
5. **Limited light types** - Point lights only in PBR shader

---

## Future Directions

1. **Per-instance material data** - Storage buffer for varying roughness/metallic per instance
2. **Shadow mapping** - Shadow pass + PCF sampling
3. **Texture maps** - Diffuse, normal, roughness textures
4. **Spot/Directional lights** - Currently only PointLight implemented in PBR
5. **Environment mapping** - IBL with prefiltered radiance
6. **Skinned animation** - Bone transforms in vertex shader
7. **Post-processing** - Bloom, DOF, TAA