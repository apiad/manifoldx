# PBR with Multiple Lights: Technical Implementation Requirements

## Overview

This document details the technical requirements for implementing Physically Based Rendering (PBR) with multiple lights in a real-time rendering pipeline. The focus is on practical implementation details with code examples in WGSL and GLSL.

---

## 1. PBR Material Data (CPU to GPU)

### Minimum Required Data

For a basic PBR implementation, the following data must be passed from CPU to GPU:

| Uniform/Attribute | Type | Description |
|-------------------|------|-------------|
| `albedo` | `vec3` | Base surface color (RGB) |
| `metallic` | `float` | Metalness factor [0.0 - 1.0] |
| `roughness` | `float` | Surface roughness [0.0 - 1.0] |
| `ao` | `float` | Ambient occlusion factor |
| `normal` | `vec3` | Surface normal (interpolated) |
| `worldPos` | `vec3` | Fragment world position |

### Optional Extended Data

For textured PBR materials, additional textures are passed:

```glsl
// GLSL Example - Uniform declarations
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

// Fragment shader sampling
void main()
{
    vec3 albedo = pow(texture(albedoMap, TexCoords).rgb, 2.2);
    vec3 normal = getNormalFromNormalMap();
    float metallic = texture(metallicMap, TexCoords).r;
    float roughness = texture(roughnessMap, TexCoords).r;
    float ao = texture(aoMap, TexCoords).r;
}
```

### Data Structure Packing

For optimized GPU memory usage, consider packing related values:

```wgsl
// WGSL Example - Packed material structure
struct PBRMaterial {
    albedo: vec3<f32>,
    padding0: f32,
    metallic_roughness: vec2<f32>,
    ao_emissive: vec2<f32>,
    normal_scale: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}
```

---

## 2. Light Data Requirements

### Point Lights

```wgsl
struct PointLight {
    position: vec3<f32>,
    padding0: f32,
    color: vec3<f32>,
    padding1: f32,
    intensity: f32,
    padding2: f32,
}
```

### Spot Lights

```wgsl
struct SpotLight {
    position: vec3<f32>,
    padding0: f32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    padding1: f32,
    innerCone: f32,
    outerCone: f32,
    padding2: vec2<f32>,
}
```

### Directional Lights

```wgsl
struct DirectionalLight {
    direction: vec3<f32>,
    padding0: f32,
    color: vec3<f32>,
    intensity: f32,
}
```

### Light Buffer Layout

```wgsl
// Array of lights (max N lights)
struct LightData {
    pointLights: array<PointLight, 4>,
    spotLights: array<SpotLight, 4>,
    directionalLights: array<DirectionalLight, 4>,
    numPointLights: u32,
    numSpotLights: u32,
    numDirectionalLights: u32,
}
```

### Minimum Light Data Per Type

| Light Type | Required Fields |
|------------|-----------------|
| **Point** | position (vec3), color (vec3), intensity (float) |
| **Spot** | position (vec3), direction (vec3), color (vec3), intensity (float), innerCone (float), outerCone (float) |
| **Directional** | direction (vec3), color (vec3), intensity (float) |

---

## 3. Fragment Shader Color Computation

### BRDF Functions Required

The Cook-Torrance BRDF requires three primary functions:

```glsl
// GLSL - Normal Distribution Function (GGX)
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265359 * denom * denom;

    return num / denom;
}

// GLSL - Geometry Function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// GLSL - Fresnel Equation (Schlick approximation)
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
```

### Light Contribution Calculation

For each light source, compute radiance and apply BRDF:

```glsl
// GLSL - Main lighting loop
vec3 calculateLightContribution(vec3 N, vec3 V, vec3 worldPos,
                                  vec3 albedo, float metallic,
                                  float roughness, float ao,
                                  vec3 lightPos, vec3 lightColor, float lightIntensity)
{
    // Light direction and half-vector
    vec3 L = normalize(lightPos - worldPos);
    vec3 H = normalize(V + L);

    // Distance and attenuation
    float distance = length(lightPos - worldPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = lightColor * lightIntensity * attenuation;

    // Calculate F0 (base reflectivity)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    // Final light contribution
    float NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / 3.14159265359 + specular) * radiance * NdotL;
}
```

### Multiple Light Accumulation

```glsl
// GLSL - Full fragment shader with multiple lights
#version 330 core

out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

// Material parameters
uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

// Light arrays
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camPos;

const float PI = 3.14159265359;

void main()
{
    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - WorldPos);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Accumulate light contributions
    vec3 Lo = vec3(0.0);

    for(int i = 0; i < 4; ++i)
    {
        vec3 L = normalize(lightPositions[i] - WorldPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Ambient term (simple approximation)
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;

    // Tone mapping (Reinhard)
    color = color / (color + vec3(1.0));

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
```

---

## 4. WGSL Implementation Reference

### Complete WGSL PBR Fragment Shader

```wgsl
struct Uniforms {
    viewProj: mat4x4<f32>,
    cameraPos: vec3<f32>,
    deltaTime: f32,
}

struct PointLight {
    position: vec3<f32>,
    padding0: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct PBRParams {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> material: PBRParams;
@group(0) @binding(2) var<uniform> lights: array<PointLight, 4>;
@group(0) @binding(3) var textureDiffuse: texture_2d<f32>;
@group(0) @binding(4) var samplerDiffuse: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

// Constants
const PI: f32 = 3.14159265359;

// Normal Distribution Function (GGX)
fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let num = a2;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    let denom = PI * denom * denom;

    return num / denom;
}

// Geometry Function (Schlick-GGX)
fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;

    let num = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = geometrySchlickGGX(NdotV, roughness);
    let ggx2 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel Equation (Schlick approximation)
fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

// Calculate attenuation
fn calculateAttenuation(distance: f32, light: PointLight) -> f32 {
    return light.intensity / (distance * distance);
}

// Calculate point light contribution
fn calculatePointLight(N: vec3<f32>, V: vec3<f32>, worldPos: vec3<f32>,
                        F0: vec3<f32>, albedo: vec3<f32>, metallic: f32,
                        roughness: f32, light: PointLight) -> vec3<f32> {
    let L = normalize(light.position - worldPos);
    let H = normalize(V + L);

    let distance = length(light.position - worldPos);
    let attenuation = calculateAttenuation(distance, light);
    let radiance = light.color * attenuation;

    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = numerator / denominator;

    let NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(input.normal);
    let V = normalize(uniforms.cameraPos - input.worldPos);

    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);

    var Lo = vec3<f32>(0.0);

    // Loop through point lights
    for (var i = 0u; i < 4u; i++) {
        let light = lights[i];
        // Skip lights with zero intensity
        if (light.intensity > 0.0) {
            Lo += calculatePointLight(N, V, input.worldPos, F0,
                                       material.albedo, material.metallic,
                                       material.roughness, light);
        }
    }

    // Ambient term
    let ambient = vec3<f32>(0.03) * material.albedo * material.ao;
    var color = ambient + Lo;

    // Tone mapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
```

### Vertex Shader (WGSL)

```wgsl
struct Uniforms {
    viewProj: mat4x4<f32>,
    model: mat4x4<f32>,
    cameraPos: vec3<f32>,
    deltaTime: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let worldPos = (uniforms.model * vec4<f32>(input.position, 1.0)).xyz;
    output.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
    output.worldPos = worldPos;
    output.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
    output.uv = input.uv;
    return output;
}
```

---

## 5. Key Implementation Notes

### Attenuation Models

For physically accurate attenuation, use inverse square law:

```glsl
float attenuation = 1.0 / (distance * distance);
```

For more artistic control, use constant-linear-quadratic:

```glsl
float attenuation = 1.0 / (constant + linear * distance + quadratic * distance * distance);
```

### F0 (Base Reflectivity)

For dielectric surfaces: F0 = 0.04 (constant)
For metallic surfaces: F0 = albedo (metallic reflects base color)

```glsl
vec3 F0 = vec3(0.04);
F0 = mix(F0, albedo, metallic);
```

### Energy Conservation

Always ensure diffuse plus specular contributions do not exceed 1.0:

```glsl
vec3 kS = F; // Specular reflection
vec3 kD = vec3(1.0) - kS; // Diffuse refraction
kD *= 1.0 - metallic; // Metals do not have diffuse
```

### Gamma Correction

PBR requires linear space calculations. Apply gamma correction at the end:

```glsl
// Tone map first
color = color / (color + vec3(1.0));

// Then gamma correct
color = pow(color, vec3(1.0 / 2.2));
```

### Performance Considerations

1. Light Culling: Use frustum or distance culling to limit active lights
2. Deferred Rendering: For many lights, deferred rendering is preferred
3. Shader Permutations: Compile different shaders for different light counts
4. Uniform Buffer Objects: Pack light data contiguously for cache efficiency

---

## 6. References

- LearnOpenGL - PBR Lighting (https://learnopengl.com/PBR/Lighting)
- LearnOpenGL - PBR Theory (https://learnopengl.com/PBR/Theory)
- Epic Games - Real Shading in Unreal Engine 4
- Brian Karis - Specular BRDF Reference

---

## Summary

Implementing PBR with multiple lights requires:

1. **Material uniforms**: albedo, metallic, roughness, ao, normal, worldPos
2. **Light data**: position or direction, color, intensity (and cone angles for spot lights)
3. **BRDF functions**: GGX (NDF), Schlick-GGX (geometry), Fresnel-Schlick
4. **Per-light loop**: Calculate radiance, apply BRDF, accumulate contribution
5. **Post-processing**: Tone mapping and gamma correction