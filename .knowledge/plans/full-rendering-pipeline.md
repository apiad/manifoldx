---
id: full-rendering-pipeline
created: 2026-04-06
modified: 2026-04-06
type: plan
status: active
expires: 2026-04-13
phases:
  - name: MVP - Single Cube
    done: false
    goal: Get one cube rendering on screen with default camera
  - name: GPU Resources
    done: false
    goal: Create GPU buffers and resources from geometry data
  - name: Render Pipeline Setup
    done: false
    goal: Define vertex formats, create WGSL shaders, build render pipelines
  - name: Transform Uploads
    done: false
    goal: Set up uniform buffers to upload transform matrices to GPU
  - name: Draw Calls
    done: false
    goal: Issue actual draw calls in render pass with bound pipelines and buffers
---

# Plan: Full Rendering Pipeline Implementation

## Context
ManifoldX has the ECS architecture in place but the render pipeline is incomplete - only clears the screen. Need to implement full WGPU rendering from scratch.

## MVP Target
**Goal:** Run `examples/cube.py` and see one RED cube on screen with a default camera looking at it.

**Current cube.py spawns:**
- 1 cube mesh (from `mx.geometry.cube(1,1,1)`)
- Red Phong material
- Transform at position (0,0,0), scale (1,1,1)

**What's needed:**
1. Camera at some position (e.g., z=5) looking at origin
2. View matrix + projection matrix (perspective)
3. GPU buffers for cube vertices/indices
4. Simple shader with MVP transform
5. Draw call

## Phases

### Phase 1: MVP - Single Cube
**Goal:** Get one cube rendering on screen with default camera

**Deliverable:** Running `python examples/cube.py` shows a cube

**Done when:**
- [ ] Camera created in Engine with position (0, 0, 5), look_at (0, 0, 0), FOV 60°
- [ ] View matrix computed from camera
- [ ] Projection matrix (perspective) computed  
- [ ] Combined MVP uniform buffer created
- [ ] GPU buffers created for cube geometry (vertex + index)
- [ ] Simple WGSL shader: vertex transforms by MVP, fragment outputs red color
- [ ] Render pipeline created (triangle-list topology)
- [ ] Draw call issued in render pass
- [ ] Cube visible on screen

**Technical approach:**
```
_engine.py:
- Add self._camera = Camera(...) in __init__
- Pass device to GeometryRegistry in __init__

_renderer.py RenderPipeline.run():
- Create pipeline once (lazy)
- Each frame: upload transforms to uniform, set pipeline, bind buffers, draw
```

### Phase 2: GPU Resources
**Goal:** Convert geometry data (positions, indices) to WGPU buffers

**Deliverable:** GeometryRegistry creates actual GPU buffers on registration

**Done when:**
- [ ] `create_geometry_buffer()` method in GeometryRegistry creates WGPU vertex/index buffers
- [ ] Buffer created with correct usage flags (VERTEX, INDEX)
- [ ] Geometry data uploaded to GPU via queue.write_buffer()
- [ ] Buffer objects stored and accessible by geometry ID

### Phase 3: Render Pipeline Setup  
**Goal:** Define vertex formats and create WGSL shaders + render pipelines

**Done when:**
- [ ] VertexBufferLayout defined with position (vec3) format
- [ ] WGSL vertex shader: receives position, applies MVP transform, outputs position
- [ ] WGSL fragment shader: outputs solid color (from material)
- [ ] RenderPipeline created with topology (triangle-list)
- [ ] Color target state configured

### Phase 4: Transform Uploads
**Goal:** Upload per-entity transform matrices to GPU each frame

**Done when:**
- [ ] Uniform buffer created for transform matrices
- [ ] TransformCache writes updated matrices to uniform buffer
- [ ] Bind group layout for uniform buffer
- [ ] WGSL shader uses uniform for model transform

### Phase 5: Draw Calls
**Goal:** Actually render meshes each frame

**Done when:**
- [ ] RenderPipeline.run() queries entities with Mesh+Material+Transform
- [ ] For each batch: set pipeline, set vertex/index buffers, set bind groups
- [ ] draw_indexed() called per batch
- [ ] Multiple entities/batches handled correctly

## Success Criteria
- [ ] Running cube.py shows one cube on screen
- [ ] Running cubes.py shows many cubes (when we enable the velocity/life system)

## Technical Approach

```
Engine.run() loop:
  1. Systems run (update transforms)
  2. Commands execute (spawn/destroy)
  3. RenderPipeline.run():
     - Mark transforms dirty -> compute matrices -> upload to GPU buffer
     - For each entity with Mesh+Material+Transform:
       - Get geometry buffer, material color
       - Set render pipeline
       - Set vertex/index buffers  
       - Set transform uniform bind group
       - draw_indexed()
  4. _render_frame(): get texture, begin render pass, submit, present
```

## Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| WGPU API complexity | high | Copy from pygfx shaders/pipeline setup |
| Buffer mapping issues | med | Use write_buffer instead of mapped buffers |
| Matrix ordering issues | med | Test with simple translation first |

## Related
- Research: lib/pygfx/pygfx/renderers/wgpu/shaders/meshshader.py - WGSL shader patterns
- Research: lib/pygfx/pygfx/resources/_buffer.py - GPU buffer creation
- Research: lib/pygfx/pygfx/renderers/wgpu/engine/pipeline.py - pipeline setup