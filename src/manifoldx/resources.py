"""GPU resource management: Geometry, Material registries and factories."""

import numpy as np
import wgpu
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# _SPRITE_QUAD is imported lazily inside GeometryRegistry.__init__ to avoid
# a circular import: resources -> viz.geometry -> viz -> viz.materials -> resources.


# =============================================================================
# Material Base Class
# =============================================================================


class Material(ABC):
    """Abstract base class for PBR materials."""

    binding_slot: int = -1

    @classmethod
    @abstractmethod
    def _compile(cls) -> str:
        """Return WGSL shader source code."""
        pass

    @classmethod
    @abstractmethod
    def uniform_type(cls) -> Dict[str, str]:
        """Return {field_name: wgsl_type} for uniform buffer fields."""
        pass

    def get_texture_bindings(self) -> Dict[int, Any]:
        """Return {binding_index: TextureHandle} for sampler+texture entries.

        Default: no textures. Override on materials that bind 2D textures.
        Sampler is attached at `binding_index`; texture view at `binding_index + 1`.
        """
        return {}


_BASICMATERIAL_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct Transforms {
    models: array<mat4x4<f32>>,
};

struct BasicMaterialUniforms {
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: BasicMaterialUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.position = globals.vp * vec4<f32>(out.world_pos, 1.0);
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.5773, 0.5773, 0.5773));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let brightness = 0.3 + 0.7 * diffuse;
    return vec4<f32>(material.color.rgb * brightness, material.color.a);
}
"""


class BasicMaterial(Material):
    """Unlit material with flat color."""

    binding_slot = 0

    def __init__(self, color):
        self.color = color

    @classmethod
    def _compile(cls) -> str:
        return _BASICMATERIAL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"color": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        """Return material data as numpy array (color as vec4)."""
        if isinstance(self.color, str):
            color_hex = self.color.lstrip("#")
            r = int(color_hex[0:2], 16) / 255.0
            g = int(color_hex[2:4], 16) / 255.0
            b = int(color_hex[4:6], 16) / 255.0
            color = np.array([r, g, b, 1.0], dtype=np.float32)
        else:
            color = np.array(self.color, dtype=np.float32)
            if len(color) == 3:
                color = np.append(color, 1.0)
        return np.tile(color, (n, 1))


_STANDARDMATERIAL_SHADER = """
struct Globals {
    vp:              mat4x4<f32>,   // offset   0
    view:            mat4x4<f32>,   // offset  64
    proj:            mat4x4<f32>,   // offset 128
    camera_pos:      vec3<f32>,     // offset 192
    _pad0:           f32,           // offset 204
    viewport_size:   vec2<f32>,     // offset 208
    _pad1:           vec2<f32>,     // offset 216
    ibl_intensity:   f32,           // offset 224
    ibl_enabled:     u32,           // offset 228
    _pad_ibl:        vec2<f32>,     // offset 232
    light_view_proj: mat4x4<f32>,   // offset 240
    sun_direction:   vec3<f32>,     // offset 304
    _pad_sun0:       f32,           // offset 316
    sun_color:       vec3<f32>,     // offset 320
    sun_intensity:   f32,           // offset 332
    shadow_enabled:  u32,           // offset 336
    shadow_bias:     f32,           // offset 340
    shadow_map_size: f32,           // offset 344
    shadow_pcf_radius: u32,         // offset 348
    spot_position:   vec3<f32>,     // offset 352
    spot_range:      f32,           // offset 364
    spot_direction:  vec3<f32>,     // offset 368
    spot_cos_inner:  f32,           // offset 380
    spot_color:      vec3<f32>,     // offset 384
    spot_intensity:  f32,           // offset 396
    spot_cos_outer:  f32,           // offset 400
    shadow_caster:   u32,           // offset 404  (0=none, 1=sun, 2=spot)
    _pad_spot0:      f32,           // offset 408
    _pad_spot1:      f32,           // offset 412
};

struct Transforms {
    models: array<mat4x4<f32>>,
};

struct PBRMaterialUniforms {
    albedo:    vec3<f32>,
    roughness: f32,
    metallic:  f32,
    ao:        f32,
    _pad0:     f32,
    _pad1:     f32,
};

struct PointLightData {
    position:  vec3<f32>,
    _pad0:     f32,
    color:     vec3<f32>,
    intensity: f32,
};

struct LightData {
    lights: array<PointLightData, 4>,
};

@group(0) @binding(0) var<uniform> globals:    Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material:  PBRMaterialUniforms;
@group(0) @binding(3) var<uniform> light_data: LightData;

@group(1) @binding(0) var env_sampler:     sampler;
@group(1) @binding(1) var irradiance_map:  texture_cube<f32>;
@group(1) @binding(2) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(3) var brdf_lut:        texture_2d<f32>;

@group(2) @binding(0) var shadow_map:     texture_depth_2d;
@group(2) @binding(1) var shadow_sampler: sampler_comparison;

const PI: f32 = 3.14159265359;

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom_base = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom_base * denom_base);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness)
         * geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnelSchlickRoughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let r1 = vec3<f32>(1.0 - roughness);
    return F0 + (max(r1, F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn calculatePointLight(N: vec3<f32>, V: vec3<f32>, worldPos: vec3<f32>,
                       F0: vec3<f32>, albedo: vec3<f32>, metallic: f32,
                       roughness: f32, light: PointLightData) -> vec3<f32> {
    let L = normalize(light.position - worldPos);
    let H = normalize(V + L);
    let dist = length(light.position - worldPos);
    let radiance = light.color * light.intensity / (dist * dist);
    let NDF = distributionGGX(N, H, roughness);
    let G   = geometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let kD  = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}

fn calculateSun(N: vec3<f32>, V: vec3<f32>, F0: vec3<f32>, albedo: vec3<f32>,
                metallic: f32, roughness: f32) -> vec3<f32> {
    // Directional light: L points toward the light = -direction. No distance falloff.
    let L = normalize(-globals.sun_direction);
    let H = normalize(V + L);
    let radiance = globals.sun_color * globals.sun_intensity;
    let NDF = distributionGGX(N, H, roughness);
    let G   = geometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let kD  = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}

fn calculateSpot(N: vec3<f32>, V: vec3<f32>, worldPos: vec3<f32>, F0: vec3<f32>,
                 albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let toL = globals.spot_position - worldPos;
    let dist = length(toL);
    let L = toL / dist;
    // Flashlight cone: smooth falloff between the inner and outer cone angles.
    let cos_angle = dot(-L, normalize(globals.spot_direction));
    let t = clamp((cos_angle - globals.spot_cos_outer)
                  / max(globals.spot_cos_inner - globals.spot_cos_outer, 0.0001), 0.0, 1.0);
    let cone = t * t;  // squared for a softer edge
    // Distance attenuation with a smooth range cutoff.
    let range_falloff = clamp(1.0 - pow(dist / globals.spot_range, 4.0), 0.0, 1.0);
    let atten = range_falloff * range_falloff / (dist * dist + 0.0001);
    let radiance = globals.spot_color * globals.spot_intensity * cone * atten;
    let H = normalize(V + L);
    let NDF = distributionGGX(N, H, roughness);
    let G   = geometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let kD  = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}

fn shadowFactor(world_pos: vec3<f32>, N: vec3<f32>) -> f32 {
    if globals.shadow_enabled == 0u { return 1.0; }
    let lp = globals.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;  // outside the caster's frustum — unshadowed
    }
    // Slope-scaled bias, using the direction toward the casting light so grazing
    // surfaces don't self-shadow (acne) without peter-panning on lit faces.
    var Lc = normalize(-globals.sun_direction);
    if globals.shadow_caster == 2u {
        Lc = normalize(globals.spot_position - world_pos);
    }
    let ndl = max(dot(N, Lc), 0.0);
    let bias = globals.shadow_bias * (1.0 + 4.0 * (1.0 - ndl));
    // PCF: average an (2r+1)x(2r+1) grid of comparisons for soft edges.
    // r == 0 collapses to a single hard-shadow tap. textureSampleCompareLevel
    // (explicit level, no derivatives) is legal in non-uniform control flow.
    let ref_depth = ndc.z - bias;
    let texel = 1.0 / globals.shadow_map_size;
    let r = i32(globals.shadow_pcf_radius);
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        for (var dx = -r; dx <= r; dx = dx + 1) {
            let off = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + off, ref_depth);
            count += 1.0;
        }
    }
    return sum / count;
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos:    vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.position  = globals.vp * vec4<f32>(out.world_pos, 1.0);
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(globals.camera_pos - in.world_pos);

    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);

    var Lo = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let light = light_data.lights[i];
        if light.intensity > 0.0 {
            Lo += calculatePointLight(N, V, in.world_pos, F0,
                                      material.albedo, material.metallic,
                                      material.roughness, light);
        }
    }

    if globals.sun_intensity > 0.0 {
        var s = 1.0;
        if globals.shadow_caster == 1u { s = shadowFactor(in.world_pos, N); }
        Lo += calculateSun(N, V, F0, material.albedo, material.metallic, material.roughness) * s;
    }

    if globals.spot_intensity > 0.0 {
        var s = 1.0;
        if globals.shadow_caster == 2u { s = shadowFactor(in.world_pos, N); }
        Lo += calculateSpot(N, V, in.world_pos, F0, material.albedo,
                            material.metallic, material.roughness) * s;
    }

    var ambient = vec3<f32>(0.03) * material.albedo * material.ao;

    if globals.ibl_enabled != 0u {
        let NdotV = max(dot(N, V), 0.0001);
        let R     = reflect(-V, N);
        let F     = fresnelSchlickRoughness(NdotV, F0, material.roughness);
        let kD    = (1.0 - F) * (1.0 - material.metallic);

        let irradiance  = textureSample(irradiance_map, env_sampler, N).rgb;
        let diffuse_ibl = kD * irradiance * material.albedo;

        let MAX_MIPS: f32 = 7.0;
        let prefilt   = textureSampleLevel(prefiltered_map, env_sampler, R,
                                           material.roughness * MAX_MIPS).rgb;
        let brdf_samp = textureSample(brdf_lut, env_sampler,
                                      vec2<f32>(NdotV, material.roughness)).rg;
        let spec_ibl  = prefilt * (F * brdf_samp.r + brdf_samp.g);

        ambient = (diffuse_ibl + spec_ibl) * material.ao * globals.ibl_intensity;
    }

    var color = ambient + Lo;
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(color, 1.0);
}
"""


_STANDARDMATERIAL_TEXTURED_SHADER = (
    _STANDARDMATERIAL_SHADER.replace(
        "@group(0) @binding(3) var<uniform> light_data: LightData;",
        "@group(0) @binding(3) var<uniform> light_data: LightData;\n"
        "@group(0) @binding(4) var albedo_sampler: sampler;\n"
        "@group(0) @binding(5) var albedo_tex: texture_2d<f32>;",
    )
    .replace(
        "    @location(1) normal:   vec3<f32>,\n    @builtin(instance_index) instance: u32,",
        "    @location(1) normal:   vec3<f32>,\n"
        "    @location(2) uv:       vec2<f32>,\n"
        "    @builtin(instance_index) instance: u32,",
    )
    .replace(
        "    @location(1) world_pos:    vec3<f32>,\n};",
        "    @location(1) world_pos:    vec3<f32>,\n    @location(2) uv:           vec2<f32>,\n};",
    )
    .replace(
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    return out;",
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    out.uv = in.uv;\n"
        "    return out;",
    )
    .replace(
        "    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);",
        "    let sampled_albedo = textureSample(albedo_tex, albedo_sampler, in.uv).rgb;\n"
        "    let F0 = mix(vec3<f32>(0.04), sampled_albedo, material.metallic);",
    )
    .replace(
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                      material.albedo, material.metallic,\n"
        "                                      material.roughness, light);",
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                      sampled_albedo, material.metallic,\n"
        "                                      material.roughness, light);",
    )
    .replace(
        "        let diffuse_ibl = kD * irradiance * material.albedo;",
        "        let diffuse_ibl = kD * irradiance * sampled_albedo;",
    )
    .replace(
        "    var ambient = vec3<f32>(0.03) * material.albedo * material.ao;",
        "    var ambient = vec3<f32>(0.03) * sampled_albedo * material.ao;",
    )
)


class StandardMaterial(Material):
    """PBR material with GGX BRDF. Optional 2D albedo map."""

    binding_slot = 1

    def __init__(self, color, roughness=0.5, metallic=0.0, ao=1.0, albedo_map=None):
        from manifoldx.textures import TextureHandle

        if albedo_map is not None and not isinstance(albedo_map, TextureHandle):
            raise TypeError(
                f"albedo_map expects a TextureHandle from load_texture(...); "
                f"got {type(albedo_map).__name__}. "
                f"Did you forget to call load_texture(engine, path) first?"
            )
        self.color = color
        self.roughness = roughness
        self.metallic = metallic
        self.ao = ao
        self.albedo_map = albedo_map

    @property
    def pipeline_subtype(self):
        return "textured" if self.albedo_map is not None else None

    def get_texture_bindings(self):
        if self.albedo_map is None:
            return {}
        return {4: self.albedo_map}

    @classmethod
    def _compile(cls, textured: bool = False) -> str:
        return _STANDARDMATERIAL_TEXTURED_SHADER if textured else _STANDARDMATERIAL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "albedo": "vec3<f32>",
            "roughness": "f32",
            "metallic": "f32",
            "ao": "f32",
        }

    def get_data(self, n: int, registry) -> np.ndarray:
        """Return material data as numpy array (albedo, roughness, metallic, ao + padding)."""
        if isinstance(self.color, str):
            color_hex = self.color.lstrip("#")
            r = int(color_hex[0:2], 16) / 255.0
            g = int(color_hex[2:4], 16) / 255.0
            b = int(color_hex[4:6], 16) / 255.0
            albedo = np.array([r, g, b], dtype=np.float32)
        else:
            albedo = np.array(self.color[:3], dtype=np.float32)

        data = np.zeros((n, 8), dtype=np.float32)
        data[:, 0:3] = albedo
        data[:, 3] = self.roughness
        data[:, 4] = self.metallic
        data[:, 5] = self.ao

        return data


# =============================================================================
# Geometry Registry
# =============================================================================


class GeometryRegistry:
    """
    Cache of GPU geometry resources.

    Lazily creates WebGPU buffers on first use.
    """

    def __init__(self, device=None):
        from manifoldx.viz.geometry import (
            AXIS_LINE_X as _axis_line_x,
            AXIS_LINE_Y as _axis_line_y,
            AXIS_LINE_Z as _axis_line_z,
            SPRITE_QUAD as _sprite_quad,
        )

        self._device = device
        self._geometries: Dict[int, Any] = {}  # id -> geometry dict
        self._object_to_id: Dict[int, int] = {}  # object id -> registry id
        self._name_to_id: Dict[str, int] = {}  # name -> registry id
        self._next_id = 1
        self._gpu_buffers: Dict[int, dict] = {}  # id -> {vertex_buffer, index_buffer}

        # Register built-in geometries by name
        self._register_builtin(_sprite_quad["name"], _sprite_quad)
        self._register_builtin(_axis_line_x["name"], _axis_line_x)
        self._register_builtin(_axis_line_y["name"], _axis_line_y)
        self._register_builtin(_axis_line_z["name"], _axis_line_z)

    def create_buffers(self, geometry_id: int, geometry_obj: dict, queue):
        """Create GPU buffers for geometry.

        Creates interleaved vertex buffer: [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z] per vertex.
        Falls back to position-only if no normals.
        """
        if self._device is None or queue is None:
            return None

        buffers = {}

        if "positions" in geometry_obj:
            positions = geometry_obj["positions"].astype(np.float32)
        elif "vertices" in geometry_obj:
            positions = geometry_obj["vertices"].astype(np.float32)
        else:
            positions = None

        if positions is not None:
            has_normals = "normals" in geometry_obj
            has_uvs = "uvs" in geometry_obj

            if has_normals and has_uvs:
                normals = geometry_obj["normals"].astype(np.float32)
                uvs = geometry_obj["uvs"].astype(np.float32)
                n_verts = len(positions)
                interleaved = np.zeros((n_verts, 8), dtype=np.float32)
                interleaved[:, 0:3] = positions
                interleaved[:, 3:6] = normals
                interleaved[:, 6:8] = uvs
                data = interleaved.tobytes()
                buffers["stride"] = 8 * 4  # pos(3) + normal(3) + uv(2)
                buffers["has_normals"] = True
                buffers["has_uvs"] = True
            elif has_normals:
                normals = geometry_obj["normals"].astype(np.float32)
                n_verts = len(positions)
                interleaved = np.zeros((n_verts, 6), dtype=np.float32)
                interleaved[:, 0:3] = positions
                interleaved[:, 3:6] = normals
                data = interleaved.tobytes()
                buffers["stride"] = 6 * 4
                buffers["has_normals"] = True
                buffers["has_uvs"] = False
            else:
                data = positions.tobytes()
                buffers["stride"] = 3 * 4
                buffers["has_normals"] = False
                buffers["has_uvs"] = False

            vertex_buffer = self._device.create_buffer(
                size=len(data),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )
            queue.write_buffer(vertex_buffer, 0, data)
            buffers["vertex_buffer"] = vertex_buffer
            buffers["vertex_count"] = len(positions)

        # Create index buffer
        if "indices" in geometry_obj:
            indices = geometry_obj["indices"]
            data = indices.astype(np.uint32).tobytes()

            index_buffer = self._device.create_buffer(
                size=len(data),
                usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
            )
            queue.write_buffer(index_buffer, 0, data)
            buffers["index_buffer"] = index_buffer
            buffers["index_count"] = len(indices)

        self._gpu_buffers[geometry_id] = buffers
        return buffers

    def get_gpu_buffers(self, geometry_id: int) -> dict:
        """Get GPU buffers for geometry."""
        return self._gpu_buffers.get(geometry_id)

    def register(self, geometry_obj) -> int:
        """Register a geometry, return ID. Same object returns same ID."""
        obj_id = id(geometry_obj)
        if obj_id in self._object_to_id:
            return self._object_to_id[obj_id]

        new_id = self._next_id
        self._next_id += 1
        self._object_to_id[obj_id] = new_id
        self._geometries[new_id] = geometry_obj
        return new_id

    def _register_builtin(self, name: str, geometry_obj: dict) -> int:
        """Register a named built-in geometry. Returns the assigned ID."""
        gid = self.register(geometry_obj)
        self._name_to_id[name] = gid
        return gid

    def register_by_name(self, name: str, geometry_obj: dict) -> int:
        """Register a geometry by name. Returns the assigned ID."""
        return self._register_builtin(name, geometry_obj)

    def get_id(self, name: str) -> int:
        """Get the numeric ID for a named geometry."""
        return self._name_to_id[name]

    def get_by_name(self, name: str) -> Any:
        """Get a geometry dict by name."""
        gid = self._name_to_id[name]
        return self._geometries[gid]

    def __contains__(self, name: str) -> bool:
        """Check if a named geometry is registered."""
        return name in self._name_to_id

    def get(self, geometry_id: int) -> Any:
        """Get geometry by ID."""
        return self._geometries.get(geometry_id)


# =============================================================================
# Material Registry
# =============================================================================


class MaterialRegistry:
    """Cache of GPU material pipelines."""

    def __init__(self, device=None):
        self._device = device
        self._materials: Dict[int, Any] = {}
        self._object_to_id: Dict[int, int] = {}
        self._next_id = 1

    def register(self, material_obj) -> int:
        """Register material, return ID. Same object returns same ID."""
        obj_id = id(material_obj)
        if obj_id in self._object_to_id:
            return self._object_to_id[obj_id]

        new_id = self._next_id
        self._next_id += 1
        self._object_to_id[obj_id] = new_id
        self._materials[new_id] = material_obj
        return new_id

    def get(self, material_id: int) -> Any:
        """Get material by ID."""
        return self._materials.get(material_id)


# =============================================================================
# Geometry Factories
# =============================================================================


def cube(width: float, height: float, depth: float) -> dict:
    """
    Create cube geometry with per-face vertices and normals for flat shading.

    Returns dict with 'positions', 'normals', and 'indices' arrays.
    24 vertices (4 per face), 36 indices.
    """
    w, h, d = width / 2, height / 2, depth / 2

    # 6 faces × 4 vertices each = 24 vertices (unshared for flat normals)
    positions = np.array(
        [
            # Front face (z+)
            [-w, -h, d],
            [w, -h, d],
            [w, h, d],
            [-w, h, d],
            # Back face (z-)
            [w, -h, -d],
            [-w, -h, -d],
            [-w, h, -d],
            [w, h, -d],
            # Right face (x+)
            [w, -h, d],
            [w, -h, -d],
            [w, h, -d],
            [w, h, d],
            # Left face (x-)
            [-w, -h, -d],
            [-w, -h, d],
            [-w, h, d],
            [-w, h, -d],
            # Top face (y+)
            [-w, h, d],
            [w, h, d],
            [w, h, -d],
            [-w, h, -d],
            # Bottom face (y-)
            [-w, -h, -d],
            [w, -h, -d],
            [w, -h, d],
            [-w, -h, d],
        ],
        dtype=np.float32,
    )

    normals = np.array(
        [
            # Front
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            # Back
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            # Right
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            # Left
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            # Top
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            # Bottom
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    # 6 faces × 2 triangles × 3 indices = 36
    indices = np.array(
        [
            0,
            1,
            2,
            0,
            2,
            3,  # front
            4,
            5,
            6,
            4,
            6,
            7,  # back
            8,
            9,
            10,
            8,
            10,
            11,  # right
            12,
            13,
            14,
            12,
            14,
            15,  # left
            16,
            17,
            18,
            16,
            18,
            19,  # top
            20,
            21,
            22,
            20,
            22,
            23,  # bottom
        ],
        dtype=np.uint32,
    )

    return {"positions": positions, "normals": normals, "indices": indices}


def sphere(radius: float, segments: int = 32) -> dict:
    """
    Create UV sphere geometry with normals and UVs.

    Normals point outward (normalized position for unit sphere).
    UVs use standard (u = lon/lon_lines, v = lat/lat_lines) spherical mapping.
    Winding order is CCW when viewed from outside.
    """
    lat_lines = segments
    lon_lines = segments * 2

    positions = []
    normals = []
    uvs = []
    for lat in range(lat_lines + 1):
        theta = lat * np.pi / lat_lines
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        v = lat / lat_lines

        for lon in range(lon_lines + 1):
            phi = lon * 2 * np.pi / lon_lines
            nx = sin_theta * np.cos(phi)
            ny = cos_theta
            nz = sin_theta * np.sin(phi)
            normals.append([nx, ny, nz])
            positions.append([nx * radius, ny * radius, nz * radius])
            uvs.append([lon / lon_lines, v])

    positions = np.array(positions, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    indices = []
    for lat in range(lat_lines):
        for lon in range(lon_lines):
            first = lat * (lon_lines + 1) + lon
            second = first + lon_lines + 1

            indices.extend([first, first + 1, second])
            indices.extend([second, first + 1, second + 1])

    indices = np.array(indices, dtype=np.uint32)

    return {"positions": positions, "normals": normals, "uvs": uvs, "indices": indices}


def plane(width: float, height: float) -> dict:
    """Create plane geometry with normals facing +Z and UVs in [0,1]²."""
    w, h = width / 2, height / 2

    positions = np.array(
        [
            [-w, -h, 0],
            [w, -h, 0],
            [w, h, 0],
            [-w, h, 0],
        ],
        dtype=np.float32,
    )

    normals = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        dtype=np.float32,
    )

    uvs = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    return {"positions": positions, "normals": normals, "uvs": uvs, "indices": indices}


# =============================================================================
# Material Factories
# =============================================================================


def basic(color):
    """Create unlit material."""
    return BasicMaterial(color)


class PhongMaterial(BasicMaterial):
    """Phong shading material (uses BasicMaterial shader for now)."""

    def __init__(self, color, shininess: float = 32.0):
        super().__init__(color)
        self.shininess = shininess


def phong(color, shininess: float = 32.0):
    """Create Phong shading material."""
    return PhongMaterial(color, shininess)


def standard(color, roughness: float = 0.5, metallic: float = 0.0) -> StandardMaterial:
    """Create PBR material."""
    return StandardMaterial(color, roughness, metallic)


# =============================================================================
# Light Classes
# =============================================================================


class PointLight:
    """Point light for GPU."""

    def __init__(self, color, intensity, position, distance=0, decay=2):
        self.color = color
        self.intensity = intensity
        self.position = position
        self.distance = distance
        self.decay = decay

    @classmethod
    def uniform_type(cls) -> dict:
        return {
            "position": "vec3<f32>",
            "padding0": "f32",
            "color": "vec3<f32>",
            "intensity": "f32",
        }

    def get_data(self) -> np.ndarray:
        """Return light data as numpy array for GPU upload."""
        color_hex = self.color.lstrip("#")
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        data = np.zeros(8, dtype=np.float32)
        data[0:3] = self.position
        data[4:7] = [r, g, b]
        data[7] = self.intensity
        return data


class SpotLight:
    """Spot light for GPU."""

    def __init__(
        self,
        color,
        intensity,
        position,
        direction,
        inner_angle,
        outer_angle,
        distance=0,
        decay=2,
    ):
        self.color = color
        self.intensity = intensity
        self.position = position
        self.direction = direction
        self.inner_angle = inner_angle
        self.outer_angle = outer_angle
        self.distance = distance
        self.decay = decay

    @classmethod
    def uniform_type(cls) -> dict:
        return {
            "position": "vec3<f32>",
            "padding0": "f32",
            "color": "vec3<f32>",
            "intensity": "f32",
            "direction": "vec3<f32>",
            "inner_angle": "f32",
        }

    def get_data(self) -> np.ndarray:
        """Return light data as numpy array for GPU upload."""
        color_hex = self.color.lstrip("#")
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        data = np.zeros(12, dtype=np.float32)
        data[0:3] = self.position
        data[4:7] = [r, g, b]
        data[7] = self.intensity
        data[8:11] = self.direction
        data[11] = self.outer_angle
        return data


class DirectionalLight:
    """Directional light for GPU."""

    def __init__(self, color, intensity, direction):
        self.color = color
        self.intensity = intensity
        self.direction = direction

    @classmethod
    def uniform_type(cls) -> dict:
        return {
            "direction": "vec3<f32>",
            "padding0": "f32",
            "color": "vec3<f32>",
            "intensity": "f32",
        }

    def get_data(self) -> np.ndarray:
        """Return light data as numpy array for GPU upload."""
        color_hex = self.color.lstrip("#")
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        data = np.zeros(8, dtype=np.float32)
        data[0:3] = self.direction
        data[4:7] = [r, g, b]
        data[7] = self.intensity
        return data


class _VolumeResource:
    """Internal: one registered volume's CPU + (eventual) GPU state."""

    __slots__ = ("data", "name", "dirty", "texture")

    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name
        self.dirty = True  # set on register/update; cleared post-upload
        self.texture = None  # GPU texture; created lazily on first upload


class VolumeRegistry:
    """Cache of GPU 3D scalar field resources.

    Mirrors the GeometryRegistry / MaterialRegistry shape:
    - `register(numpy_array, name=...) -> int handle`
    - `update(handle, numpy_array)` — same shape only; flips dirty bit.
    - `get(handle) -> _VolumeResource`
    - GPU texture creation is lazy (handled by the renderer at the
      first frame in which the volume is needed).
    """

    def __init__(self, device=None):
        self._device = device
        self._volumes: Dict[int, _VolumeResource] = {}
        self._next_id = 1

    def register(self, data: np.ndarray, *, name: str | None = None) -> int:
        if data.ndim != 3:
            raise ValueError(f"volume data must be 3D (Nz, Ny, Nx); got {data.ndim}D")
        if data.dtype != np.float32:
            raise ValueError(
                f"volume data must be float32; got {data.dtype}. "
                f"Convert with `array.astype(np.float32)`."
            )
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "volume data must be C-contiguous; "
                "call `np.ascontiguousarray(array)` before registering."
            )
        handle = self._next_id
        self._next_id += 1
        self._volumes[handle] = _VolumeResource(data=data, name=name or f"volume_{handle}")
        return handle

    def update(self, handle: int, data: np.ndarray) -> None:
        res = self.get(handle)
        if data.shape != res.data.shape:
            raise ValueError(
                f"update_volume: shape mismatch — registered {res.data.shape}, "
                f"got {data.shape}. Re-register if you need a different size."
            )
        if data.dtype != np.float32:
            raise ValueError(f"volume data must be float32; got {data.dtype}.")
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError("volume data must be C-contiguous.")
        res.data = data
        res.dirty = True

    def get(self, handle: int) -> _VolumeResource:
        if handle not in self._volumes:
            raise KeyError(f"unknown volume handle: {handle}")
        return self._volumes[handle]

    def upload_to_gpu(self, handle: int, queue) -> None:
        """Lazily create a `texture_3d` r32float for this volume and write
        the current numpy data into it. Clears the dirty bit. Re-creates the
        texture only if shape changed (which `update` already disallows).
        """
        if self._device is None or queue is None:
            return
        res = self.get(handle)
        if not res.dirty and res.texture is not None:
            return

        nz, ny, nx = res.data.shape
        if res.texture is None:
            res.texture = self._device.create_texture(
                size=(nx, ny, nz),
                format=wgpu.TextureFormat.r32float,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                dimension=wgpu.TextureDimension.d3,
                mip_level_count=1,
                sample_count=1,
            )

        data_bytes = res.data.tobytes()
        bytes_per_row = nx * 4  # 4 bytes per f32 voxel
        rows_per_image = ny
        queue.write_texture(
            {"texture": res.texture, "mip_level": 0, "origin": (0, 0, 0)},
            data_bytes,
            {
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": rows_per_image,
            },
            (nx, ny, nz),
        )
        res.dirty = False


__all__ = [
    "Material",
    "BasicMaterial",
    "StandardMaterial",
    "GeometryRegistry",
    "MaterialRegistry",
    "VolumeRegistry",
    "cube",
    "sphere",
    "plane",
    "basic",
    "phong",
    "standard",
    "PointLight",
    "SpotLight",
    "DirectionalLight",
]
