---
id: implement-pbr-material-system
created: 2026-04-06
modified: 2026-04-06
type: plan
status: done
expires: 2026-04-13
phases:
  - name: phase-1-material-base-class
    done: false
  - name: phase-2-basic-material-shader
    done: false
  - name: phase-3-standard-material-pbr
    done: false
  - name: phase-4-external-light-system
    done: false
  - name: phase-5-multi-instance-pbr-demo
    done: false
---

# Plan: PBR Material System with Multi-Instance Rendering

## Context

Current renderer uses a single hardcoded diffuse shader. We need:
1. Material types that compile to different shaders
2. Per-instance material properties via uniform buffers
3. External light system (not in ECS)
4. PBR StandardMaterial with GGX BRDF

Final goal: Render multiple spheres/cubes with different PBR properties (color, roughness, metallic) in a single draw call, with animated external lights.

---

## Phases

### Phase 1: Material Base Class with Shader Compilation
**Goal:** Create abstract Material class that defines the interface for compile-time shader generation

**Deliverable:**
- `Material` base class with abstract `_compile()` classmethod
- `uniform_type()` returning buffer field definitions
- `binding_slot` per material type

**Done when:**
- [ ] `Material` base class in `resources.py`
- [ ] `_compile()` returns WGSL string
- [ ] `uniform_type()` returns dict of field definitions
- [ ] `binding_slot` class attribute per material type
- [ ] Tests: material._compile() returns valid WGSL

---

### Phase 2: BasicMaterial Shader (Unlit)
**Goal:** Refactor current shader through BasicMaterial._compile()

**Deliverable:**
- BasicMaterial class returning unlit WGSL shader
- Renderer updated to use material-type specific pipelines
- Shared bind group layout for all material types

**Done when:**
- [ ] BasicMaterial._compile() returns unlit shader
- [ ] Shader renders flat color (no lighting)
- [ ] Renderer creates pipeline per (geometry_id, material_type)
- [ ] Demo: single cube with BasicMaterial renders correctly

---

### Phase 3: StandardMaterial (PBR) Shader
**Goal:** Implement PBR shader with GGX BRDF

**Deliverable:**
- StandardMaterial with albedo, roughness, metallic, ao properties
- PBR fragment shader with:
  - GGX Normal Distribution Function
  - Smith Geometry Function  
  - Fresnel-Schlick approximation
  - Energy conservation
- Tone mapping + gamma correction

**Done when:**
- [ ] StandardMaterial._compile() returns PBR WGSL
- [ ] Shader compiles without errors
- [ ] Demo: single sphere with StandardMaterial renders with PBR shading
- [ ] Properties: roughness makes surface matte/shiny, metallic makes it reflective

---

### Phase 4: External Light System
**Goal:** Lights passed externally (like camera), not in ECS

**Deliverable:**
- Light classes: PointLight, SpotLight, DirectionalLight
- Engine API: `engine.set_lights([...])`
- Light uniform buffer passed to all PBR shaders
- Animated lights (position updates in system)

**Done when:**
- [ ] Light classes with color, intensity, position/direction
- [ ] engine.set_lights() stores lights for renderer access
- [ ] PBR shader receives light array
- [ ] Demo: animated point light orbiting the scene
- [ ] At least 3 lights (different types) supported

---

### Phase 5: Multi-Instance PBR Demo
**Goal:** Multiple instances with different material properties in single draw call

**Deliverable:**
- Storage buffer for per-instance material properties
- Renderer batches by (geometry_id, material_type)
- Instance properties uploaded per batch:
  - albedo (from material color)
  - roughness
  - metallic
  - ao
- Demo scene:
  - 3 spheres + 3 cubes
  - Different roughness/metallic per instance
  - Single draw call per geometry-type
  - Animated lights illuminating scene

**Done when:**
- [ ] Per-instance material data via storage buffer
- [ ] Instanced draw with varying material properties
- [ ] Demo renders 6+ objects with different PBR params
- [ ] Single draw call per (geometry, material_type) confirmed via debug
- [ ] Lights visibly affect materials (roughness/metalness visible)

---

## Success Criteria

1. **Materials compile to shaders**: `Material._compile()` returns valid WGSL
2. **BasicMaterial works**: Flat-colored unlit rendering
3. **StandardMaterial PBR works**: Roughness/metallic visibly affects surface appearance
4. **Lights external**: `engine.set_lights()` makes lights available to shader
5. **Multi-instance**: Multiple objects with different material properties rendered in single draw call
6. **Performance**: Each (geometry, material_type) batch = 1 draw call

---

## Technical Design

### Material Interface

```python
class Material(ABC):
    binding_slot: int = 0  # Override in subclass
    
    @classmethod
    @abstractmethod
    def _compile(cls) -> str:
        """Return WGSL shader string"""
        
    @classmethod
    @abstractmethod
    def uniform_type(cls) -> dict:
        """Return {field_name: wgsl_type}"""
    
    def get_data(self, n: int, registry) -> np.ndarray:
        """Get uniform buffer data for n instances"""
```

### Shader Binding Layout

```
Binding 0: Globals (viewProj, cameraPos, light_count)
Binding 1: Transforms (storage buffer)
Binding 2: Material uniforms (varies by type)
  - BasicMaterial: color: vec4
  - StandardMaterial: albedo: vec3, roughness: f32, metallic: f32, ao: f32
Binding 3: Lights (storage buffer, for PBR shaders)
```

### Pipeline Cache Key

```python
pipeline_key = (geometry_id, material_type)  # Not material_id
```

### Renderer Flow

```
For each (geometry_id, material_type) batch:
  1. Get/create pipeline for (geometry_id, material_type)
  2. Upload material uniforms per instance to binding 2
  3. Upload light data to binding 3 (if PBR)
  4. draw_indexed(index_count, instance_count)
```

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| WGSL shader compilation errors | High | Test _compile() in isolation first |
| Binding slot conflicts | Medium | Document binding assignments, validate at runtime |
| Per-instance data bandwidth | Low | Start small (6-10 objects), optimize later |
| Light attenuation bugs | Medium | Verify against known-good implementation (pygfx) |

---

## Related

- Research: `.knowledge/notes/pbr-multiple-lights.md`
- Reference: `lib/pygfx/pygfx/materials/_mesh.py`
- Reference: `lib/pygfx/pygfx/renderers/wgpu/wgsl/light_pbr.wgsl`