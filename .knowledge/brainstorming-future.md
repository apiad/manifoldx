# Brainstorming: Future Directions for ManifoldX

This document captures brainstorming ideas for future development, organized by category with creative suggestions and practical considerations.

---

## 1. Rendering & Materials

### High Priority
- [ ] **Per-instance material data** — Current: one material params per draw call. Want: varying roughness/metallic per instance in same draw. Solution: storage buffer instead of uniform.
- [ ] **Shadow mapping** — Basic shadow pass with PCF (Percentage Closer Filtering) for soft shadows. Requires: depth pass + shadow sampler in shader.
- [ ] **Spot/Directional lights in PBR** — Currently only PointLight implemented. Need to add SpotLight (cone attenuation) and DirectionalLight (parallel rays) to shader.

### Medium Priority
- [ ] **Environment mapping / IBL** — Image-Based Lighting with prefiltered radiance lookups. Huge impact on material realism.
- [ ] **Texture support** — Diffuse, normal, roughness, AO maps. Need: storage buffer for textures, sampler declarations in WGSL.
- [ ] **More material types:**
  - `ToonMaterial` — Cel shading with stepped gradients
  - `UnlitMaterial` — Pure flat color without any lighting
  - `EmissiveMaterial` — Self-illuminating surfaces
  - `GlassMaterial` — Refraction with Fresnel

### Lower Priority
- [ ] **Deferred rendering** — Forward+ / clustered lighting for 100+ lights
- [ ] **Post-processing pipeline** — Bloom, DOF, TAA, FXAA
- [ ] **Skybox/Environment cubemap** — Background rendering
- [ ] **Multi-pass rendering** — Render to texture for effects

---

## 2. Physics & Simulation

### High Priority
- [ ] **Basic collision detection** — AABB or sphere-sphere, simple ECS query
- [ ] **Rigid body dynamics** — Position/velocity integration with simple gravity
- [ ] **Particle system** — Instanced point sprites with lifetime, velocity

### Medium Priority  
- [ ] **Physics engine integration** — Wrap Rapier or Cannon-es? Heavy dependency though...
- [ ] **Soft body simulation** — Spring-mass system for cloth/jelly
- [ ] **Raycasting** — Picking entities via screen-to-world ray

### Lower Priority
- [ ] **Spatial hashing** — Octree or BVH for broad-phase collision
- [ ] **Verlet integration** — Better stability for constraints

---

## 3. Performance & GPU

### High Priority
- [ ] **Worker thread rendering** — Move GPU submission to separate thread
- [ ] **Batch merge optimization** — Group entities across geometries into single draw
- [ ] **LOD (Level of Detail)** — Auto-switch geometry based on distance

### Medium Priority
- [ ] **Compute shaders** — Particle updates, skinning in compute
- [ ] **WGSL compilation caching** — Avoid recompiling shaders every run
- [ ] **Instanced indirect draw** — GPU-driven draw call count

### Lower Priority
- [ ] **GPU particles** — All particle logic in compute shader
- [ ] **Portals/render to texture** — Mirror rooms, security cameras

---

## 4. Asset Pipeline

### High Priority
- [ ] **GLTF loader** — Load meshes/materials from .glb files. Game changer for usability.
- [ ] **Image loader** — PNG/JPG textures via stb_image equivalent
- [ ] **Procedural geometry** — Torus, cylinder, cone, capsule, rounded cube

### Medium Priority
- [ ] **Animation clips** — Keyframe interpolation, skeletal animation
- [ ] **Font rendering** — SDF text rendering via signed distance fields
- [ ] **Shader material include system** — Share WGSL functions between materials

### Lower Priority
- [ ] **FBX/OBJ loader** — Legacy format support
- [ ] **Audio integration** — Positional audio with OpenAL wrap
- [ ] **Scene file format** — JSON/YAML scene description

---

## 5. Developer Experience

### High Priority
- [ ] **Scene editor** — Dear ImGui immediate mode GUI for placement
- [ ] **Hot reload** — Shader recompilation without restart
- [ ] **Debug visuals** — Wireframe mode, axis gizmo, bounding box overlay

### Medium Priority
- [ ] **Profiler integration** — GPU/frame timing with tracytol
- [ ] **Entity inspector** — ImGui panel showing selected entity's components
- [ ] **Logging framework** — Structured logging with levels

### Lower Priority
- [ ] **Visual scripting** — Node-based logic editor
- [ ] **Timeline animation** — Keyframe editor for transforms
- [ ] **Scripting sandbox** — Python exec for rapid prototyping

---

## 6. Async & Events

### High Priority
- [ ] **Async asset loading** — Load GLTF/textures in background, callback when ready
- [ ] **Event system** — Entity spawn/destroy events, collision callbacks
- [ ] **Animation callbacks** — On-animation-complete triggers

### Medium Priority
- [ ] **Coroutine systems** — Async/await in system definitions
- [ ] **Network sync** — Basic replication for multiplayer (snapshot interpolation)
- [ ] **Job system** — Parallel task scheduling for heavy computation

### Lower Priority
- [ ] **Transactional ECS** — Batch entity changes per frame with rollback
- [ ] **Undo/redo system** — Command pattern for entity edits

---

## 7. Platform & WebGPU Features

### High Priority
- [ ] **Web export** — Compile to WASM + WebGPU for browser
- [ ] **VR support** — WebXR integration (requires WebGPU XR extension)
- [ ] **HDR rendering** — Float16 textures, adaptive exposure

### Medium Priority
- [ ] **Ray tracing** — WebGPU ray tracing extension when stable
- [ ] **Variable rate shading** — Foveated rendering for VR
- [ ] **Multi-GPU** — Linked adapters for SLI/Crossfire

### Lower Priority
- [ ] **Compute shaders** — GPU-driven culling, occlusion
- [ ] **Mesh shaders** — Task/mesh shaders for geometry amplification

---

## 8. Creative Ideas (Blue Sky)

### Wild Ideas
- **[VR sandbox]** — Room-scale VR with hand tracking using ManifoldX as renderer
- **[Procedural worlds]** — Infinite terrain via compute shader + chunked loading
- **[Shader playground]** — Live WGSL editor with instant preview
- **[AI agents]** — Simple entity AI with behavior trees, rendered in-engine
- **[Physics puzzle]** — Block-based physics game demo
- **[Ray traced portal]** — Render the scene multiple times for real-time portal effect
- **[Volumetric fog]** — Ray marching for atmospheric effects
- **[Fluid sim]** — SPH (Smoothed Particle Hydrodynamics) in compute shader

### Community Requests to Explore
- glTF animation support (skeletal)
- PhysX/Rapier integration
- Unity/Unreal-like editor
- Mobile support (WebGPU on mobile)
- Vulkan/Metal/D3D12 backend alternative

---

## Suggested Priority Ranking

Based on impact vs effort:

| Priority | Items |
|----------|-------|
| **Do Next** | Per-instance materials, Shadow mapping, GLTF loader |
| **Do Soon** | Spot/Directional lights, Hot reload, Debug visuals |
| **Do Later** | IBL, Deferred rendering, Physics integration |
| **Maybe** | VR, Web export, Compute shaders |

---

## Notes

- **Core philosophy**: Keep the ECS fast, make rendering flexible
- **Dependency philosophy**: Heavy deps (physics engines) = optional plugins, not core
- **Performance target**: 60fps with 10k entities on mid-range GPU
- **Python limitation**: Keep per-frame Python overhead minimal; heavy lifting in WGSL/compute