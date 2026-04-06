# Plan: ManifoldX Rendering Engine (Pygfx-Inspired)

**Status: PLANNED**

## Vision

Build a modern 3D rendering engine in Python using WebGPU (via wgpu-py) as the backend. The engine will be inspired by Pygfx's architecture but will deviate where necessary to suit our specific needs and design preferences.

## Key Architectural Decisions (Deviations from Pygfx)

| Aspect | Pygfx Approach | Our Approach |
|--------|----------------|---------------|
| Object hierarchy | Single WorldObject base | Lightweight protocol-based, minimal inheritance |
| Material system | Complex class hierarchy | Composable material descriptors |
| Transform system | Dual local/world with property wrappers | Direct matrix access, explicit updates |
| Event system | Built-in picking, comprehensive events | Minimal, opt-in |
| API style | Verbose, explicit | Concise, ergonomic |
| Rendering | Immediate-mode render passes | Command buffer abstraction |
| Geometry | Named attribute dict | Typed struct arrays |

---

## Phases

### Phase 1: Core Infrastructure

#### 1.1 Scene Graph
- Implement `Scene` class as container for renderable objects
- Implement `Object3D` class with transform matrix (4x4)
- Support parent-child relationships with world matrix propagation
- Add `add()`, `remove()`, `children` access

#### 1.2 Camera System
- Implement `Camera` base class
- Implement `PerspectiveCamera` (fov, aspect, near, far)
- Implement `OrthographicCamera` (left, right, top, bottom)
- Add `projection_matrix` property
- Add `view_matrix` (inverse of world transform)

#### 1.3 Light System
- Implement `Light` base class
- Implement `AmbientLight`, `DirectionalLight`, `PointLight`
- Add light properties: color, intensity

#### 1.4 Basic Render Loop
- Replace simple clear-pass with proper scene rendering
- Implement renderable object protocol (mesh, light, camera)
- Add basic frame timing (vsync)

**Goal**: Render a cube with a directional light and perspective camera

---

### Phase 2: Geometry & Meshes

#### 2.1 Geometry System
- Create `Geometry` class for vertex data
- Implement standard geometries: Box, Sphere, Plane, Cylinder, Cone
- Support attributes: position, normal, texcoord, color, indices

#### 2.2 Mesh System
- Implement `Mesh` class combining geometry + material
- Implement `Material` protocol
- Basic `BasicMaterial` (unlit), `PhongMaterial` (per-vertex lighting)

**Goal**: Render multiple meshes with different materials

---

### Phase 3: Materials & Shading

#### 3.1 Shader System
- Define shader pipeline in WGSL
- Basic vertex shader with MVP transform
- Basic fragment shader with simple lighting

#### 3.2 PBR Material (Deviation)
- Implement simplified PBR (Physically Based Rendering)
- Albedo, roughness, metallic properties
- Basic environment lighting

#### 3.3 Material Variants
- Wireframe material
- Points/particle material
- Line material

**Goal**: Support realistic-looking materials with proper lighting

---

### Phase 4: Resources & Textures

#### 4.1 Buffer Abstraction
- Create `Buffer` class for GPU data
- Support vertex and index buffers
- Dynamic buffer updates

#### 4.2 Texture System
- Create `Texture` class (1D, 2D, 3D)
- Support texture loading (images)
- Implement sampler (filter modes, wrap modes)

#### 4.3 Texture Mapping
- Connect textures to materials
- UV coordinate support in geometry
- Multiple texture channels (albedo, normal, etc.)

**Goal**: Load and display textured meshes

---

### Phase 5: Rendering Pipeline

#### 5.1 Render Pass Organization
- Opaque pass (depth-write, depth-test)
- Transparent pass (depth-test, no depth-write)
- Post-processing pass

#### 5.2 Transparency Handling
- Alpha blending modes (opaque, blend, mask)
- Sort transparent objects back-to-front
- Order-independent transparency (optional, future)

#### 5.3 Render Pipeline State
- Depth stencil state
- Blend state
- Primitive topology

**Goal**: Handle complex scenes with transparency correctly

---

### Phase 6: Advanced Features

#### 6.1 Shadow Mapping
- Directional light shadow maps
- Shadow bias and PCF filtering

#### 6.2 Instancing
- InstancedMesh for many identical objects
- Instance attributes (position, rotation, scale, color)

#### 6.3 Camera Controllers
- OrbitController
- PanZoomController

#### 6.4 Picking/Events (Deviation - Minimal)
- Basic raycasting for mouse picking (optional)

#### 6.5 Text Rendering (Future)
- SDF-based text
- Font loading

**Goal**: Full-featured visualization engine

---

### Phase 7: Optimization

#### 7.1 Batching
- Combine similar draw calls
- Reduce bind group switches

#### 7.2 Frustum Culling
- Skip rendering off-screen objects

#### 7.3 LOD (Level of Detail)
- Multiple detail levels for geometry

**Goal**: Handle large scenes efficiently

---

## File Structure (Proposed)

```
src/manifoldx/
├── __init__.py
├── engine.py          # Engine class (exists)
├── scene.py           # Scene, Object3D
├── camera.py          # Camera classes
├── light.py           # Light classes
├── geometry.py        # Geometry, primitive builders
├── mesh.py            # Mesh class
├── material.py        # Material protocol and implementations
├── buffer.py          # Buffer, Texture resources
├── shader.py          # WGSL shader definitions
├── renderer.py        # Renderer implementation
├── pipeline.py       # Render pipeline state
└── controllers.py    # Camera controllers
```

---

## Dependencies

- `wgpu-py` - WebGPU bindings
- `rendercanvas` - Window management (or glfw directly)
- `numpy` - Array operations

---

## Success Criteria

1. Render 3D scene with multiple objects
2. Support basic materials (unlit, Phong, PBR)
3. Proper lighting with shadows
4. Camera controls (orbit, pan/zoom)
5. Transparent objects render correctly
6. Texture mapping works
7. Reasonable performance for 1000+ objects
8. Clean, extensible API

---

## Risks & Considerations

- **Performance**: Python overhead for scene traversal
- **Shader complexity**: WGSL learning curve
- **Compatibility**: WebGPU support on target platforms
- **Maintenance**: Keep API simple to avoid Pygfx's complexity

---

## Future Considerations

- Compute shaders for GPU-based processing
- Multi-view rendering (VR, split-screen)
- Custom post-processing effects
- Export to GLTF/other formats
- Integration with data science tools