# Lighting and Materials Analysis

## Current State (Manifoldx)

The manifoldx project has basic rendering infrastructure:

### Renderer (`src/manifoldx/renderer.py`)
- Simple instanced draw pipeline
- Single hardcoded directional light with ambient term
- Basic diffuse shading: `brightness = ambient + (1.0 - ambient) * diffuse`
- Shader code embedded inline (SHADER_SOURCE)

### Materials (`src/manifoldx/resources.py`)
- `BasicMaterial` - unlit material
- `PhongMaterial` - has color + shininess
- `StandardMaterial` - PBR (color, roughness, metallic) - **NOT IMPLEMENTED IN SHADER**

### What's Missing
1. No light objects in the ECS - light is hardcoded in shader
2. No material variant switching in shader
3. No PBR/complex shading implementation
4. No shadows
5. No texture support for materials

---

## Pygfx Reference Architecture

### Light Objects (`lib/pygfx/pygfx/objects/_lights.py`)

```python
class Light(WorldObject):
    # Uniform data: color, intensity, cast_shadow
    
class AmbientLight(Light):
    # Omnidirectional, no shadows
    
class PointLight(Light):
    # distance, decay (inverse square law)
    # Can cast shadows
    
class DirectionalLight(Light):
    # Has target for direction
    # Can cast shadows
    
class SpotLight(Light):
    # angle, penumbra, distance, decay
    # Cone-shaped light with falloff
```

**Key Properties:**
- `color`: sRGB color
- `intensity`: Physical intensity (candela for point/spot)
- `cast_shadow`: Boolean for shadow casting
- Point/Spot: `distance` (cutoff), `decay` (inverse-square)

### PBR Materials (`lib/pygfx/pygfx/materials/_mesh.py`)

```python
class MeshBasicMaterial:
    # color, env_map, wireframe, flat_shading
    
class MeshStandardMaterial(PBR):
    # Base PBR with:
    # - color (diffuse)
    # - roughness (0-1)
    # - metalness (0-1)
    # - env_map (IBL)
    
    # Optional extensions:
    # - clearcoat: Clear coat layer
    # - sheen: Velvet-like surface
    # - iridescence: Rainbow effect
    # - anisotropy: Brushed metal look
```

### Shader Architecture

**Key WGSL files:**
- `light_common.wgsl` - Light attenuation, BRDF helpers
- `light_pbr.wgsl` - PBR BRDF (GGX), IBL
- `light_phong.wgsl` - Phong shading
- `light_toon.wgsl` - Cel shading
- `mesh.wgsl` - Main mesh rendering

**PBR Physical Material Structure:**
```wgsl
struct PhysicalMaterial {
    diffuse_color: vec3<f32>,
    roughness: f32,
    specular_color: vec3<f32>,
    specular_f90: f32,
    // Optional: clearcoat, iridescence, sheen, anisotropy
};
```

**Key BRDF Functions:**
- `D_GGX()` - Normal distribution (microfacet alignment)
- `V_GGX_SmithCorrelated()` - Visibility (shadowing/masking)
- `F_Schlick()` - Fresnel-Schlick approximation

---

## Implementation Plan

### Phase 1: Light System
1. Create Light component in ECS
2. Add LightRegistry to resources
3. Implement light uniform buffer in shader
4. Support multiple light types

### Phase 2: Material Variants
1. Extend shader to support material variants
2. Implement PBR fragment shader
3. Add texture sampling for maps
4. Support env maps for IBL

### Phase 3: Advanced Features
1. Shadow mapping
2. Clearcoat/sheen/iridescence
3. Anisotropy

---

## Key Technical Details

### Color Space
- Pygfx uses sRGB input, converts to physical (linear) via `srgb2physical()`
- Important for proper PBR: intensity scales in physical space

### Light Attenuation
- Inverse square law: `1 / distance^decay`
- Optional distance cutoff for performance

### PBR Formula
```
F0 = mix(0.04, base_color, metallic)
diffuse = (1 - metallic) * base_color / PI
specular = F * V * D
```

### Uniform Buffer Layout
```wgsl
struct Globals {
    vp: mat4x4<f32>,
    color: vec4<f32>,
    light_dir: vec4<f32>,   // xyz = direction, w = ambient
    // Need to expand for multiple lights
};

struct LightData {
    color: vec4<f32>,       // rgb = color, w = intensity
    position: vec4<f32>,    // xyz = position, w = distance
    direction: vec4<f32>,   // xyz = direction, w = angle
    params: vec4<f32>,      // x = decay, y = penumbra, z = type, w = unused
};
```