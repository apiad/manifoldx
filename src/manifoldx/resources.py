"""GPU resource management: Geometry, Material registries and factories."""

import numpy as np
import wgpu
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


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


_UNLIT_VERTEX_SHADER = """
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

struct Uniforms {
    viewProj: mat4x4<f32>,
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (uniforms.model * vec4<f32>(in.position, 1.0)).xyz;
    out.position = uniforms.viewProj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normalize((uniforms.model * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}
"""

_UNLIT_FRAGMENT_SHADER = """
struct FragmentInput {
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

struct MaterialUniforms {
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> material: MaterialUniforms;

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return material.color;
}
"""

_BASICMATERIAL_SHADER = _UNLIT_VERTEX_SHADER + _UNLIT_FRAGMENT_SHADER


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
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

struct Uniforms {
    viewProj: mat4x4<f32>,
    model: mat4x4<f32>,
    cameraPos: vec3<f32>,
    deltaTime: f32,
}

struct PointLight {
    position: vec3<f32>,
    padding0: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct SpotLight {
    position: vec3<f32>,
    padding0: f32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    outer_angle: f32,
}

struct DirectionalLight {
    direction: vec3<f32>,
    padding0: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct LightData {
    lights: array<PointLight, 4>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<uniform> light_data: LightData;

const PI: f32 = 3.14159265359;

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

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

fn calculateAttenuation(distance: f32, light: PointLight) -> f32 {
    return light.intensity / (distance * distance);
}

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

struct MaterialUniforms {
    albedo: vec3<f32>,
    padding0: f32,
    roughness: f32,
    metallic: f32,
    ao: f32,
    padding1: f32,
}

@group(0) @binding(2) var<uniform> material: MaterialUniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (uniforms.model * vec4<f32>(in.position, 1.0)).xyz;
    out.position = uniforms.viewProj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normalize((uniforms.model * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(uniforms.cameraPos - in.world_pos);

    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);

    var Lo = vec3<f32>(0.0);

    for (var i = 0u; i < 4u; i++) {
        let light = light_data.lights[i];
        if (light.intensity > 0.0) {
            Lo += calculatePointLight(N, V, in.world_pos, F0,
                                       material.albedo, material.metallic,
                                       material.roughness, light);
        }
    }

    let ambient = vec3<f32>(0.03) * material.albedo * material.ao;
    var color = ambient + Lo;

    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
"""


class StandardMaterial(Material):
    """PBR material with GGX BRDF."""

    binding_slot = 1

    def __init__(self, color, roughness=0.5, metallic=0.0, ao=1.0):
        self.color = color
        self.roughness = roughness
        self.metallic = metallic
        self.ao = ao

    @classmethod
    def _compile(cls) -> str:
        return _STANDARDMATERIAL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "albedo": "vec3<f32>",
            "roughness": "f32",
            "metallic": "f32",
            "ao": "f32",
        }

    def get_data(self, n: int, registry) -> np.ndarray:
        """Return material data as numpy array (albedo, roughness, metallic, ao)."""
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
        self._device = device
        self._geometries: Dict[int, Any] = {}  # id -> geometry dict
        self._object_to_id: Dict[int, int] = {}  # object id -> registry id
        self._next_id = 1
        self._gpu_buffers: Dict[int, dict] = {}  # id -> {vertex_buffer, index_buffer}

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
            has_normals = "normals" in geometry_obj

            if has_normals:
                normals = geometry_obj["normals"].astype(np.float32)
                # Interleave: [px, py, pz, nx, ny, nz] per vertex
                n_verts = len(positions)
                interleaved = np.zeros((n_verts, 6), dtype=np.float32)
                interleaved[:, 0:3] = positions
                interleaved[:, 3:6] = normals
                data = interleaved.tobytes()
                buffers["stride"] = 6 * 4  # 6 floats * 4 bytes
                buffers["has_normals"] = True
            else:
                data = positions.tobytes()
                buffers["stride"] = 3 * 4
                buffers["has_normals"] = False

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
    Create sphere geometry.

    Simplified - creates UV sphere.
    """
    lat_lines = segments
    lon_lines = segments * 2

    # Generate vertices
    positions = []
    for lat in range(lat_lines + 1):
        theta = lat * np.pi / lat_lines
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for lon in range(lon_lines + 1):
            phi = lon * 2 * np.pi / lon_lines
            x = sin_theta * np.cos(phi)
            y = cos_theta
            z = sin_theta * np.sin(phi)
            positions.append([x * radius, y * radius, z * radius])

    positions = np.array(positions, dtype=np.float32)

    # Generate indices
    indices = []
    for lat in range(lat_lines):
        for lon in range(lon_lines):
            first = lat * (lon_lines + 1) + lon
            second = first + lon_lines + 1

            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    indices = np.array(indices, dtype=np.uint32)

    return {"positions": positions, "indices": indices}


def plane(width: float, height: float) -> dict:
    """Create plane geometry."""
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

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    return {"positions": positions, "indices": indices}


# =============================================================================
# Material Factories
# =============================================================================


def basic(color):
    """Create unlit material."""
    return BasicMaterial(color)


class PhongMaterial:
    """Phong shading material."""

    def __init__(self, color, shininess: float = 32.0):
        self.color = color
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


__all__ = [
    "Material",
    "BasicMaterial",
    "StandardMaterial",
    "GeometryRegistry",
    "MaterialRegistry",
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
