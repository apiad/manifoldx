"""Render pipeline that executes after command buffer."""

import numpy as np
import wgpu


# =============================================================================
# WGSL Shaders (default fallback - not used when materials provide shaders)
# =============================================================================

SHADER_SOURCE = """
struct Globals {
    vp: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct Transforms {
    models: array<mat4x4<f32>>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    let world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.position = globals.vp * vec4<f32>(world_pos, 1.0);
    // Transform normal by model matrix (ignore translation, assume uniform scale)
    out.world_normal = (model * vec4<f32>(in.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.5773, 0.5773, 0.5773));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let brightness = 0.3 + 0.7 * diffuse;
    return vec4<f32>(vec3<f32>(0.8) * brightness, 1.0);
}
"""


# =============================================================================
# Transform Cache (With Dirty Flag Optimization)
# =============================================================================


class TransformCache:
    """
    Caches computed transform matrices.

    Stores: _matrix_cache (N, 16) - mat4x4<f32> per entity
            _dirty (N,) bool - needs recompute
    """

    def __init__(self, max_entities: int):
        self._matrix_cache = np.zeros((max_entities, 16), dtype=np.float32)
        self._dirty = np.ones(max_entities, dtype=bool)  # All dirty initially

    def mark_dirty(self, indices: np.ndarray):
        """Mark entities as needing recompute."""
        self._dirty[indices] = True

    def get_transforms(self, store, indices: np.ndarray) -> np.ndarray:
        """
        Get transform matrices for indices, recomputing if dirty.
        Returns (len(indices), 16) array of mat4x4<f32>.
        """
        dirty_mask = self._dirty[indices]
        if dirty_mask.any():
            self._recompute_matrices(store, indices[dirty_mask])

        return self._matrix_cache[indices]

    def _recompute_matrices(self, store, indices: np.ndarray):
        """Compute mat4x4 from pos/rot/scale using vectorized numpy."""
        if len(indices) == 0:
            return

        transform_data = store.get_component_data("Transform", indices)

        positions = transform_data[:, 0:3]  # (N, 3)
        quats = transform_data[:, 3:7]  # (N, 4) - quaternion (x, y, z, w)
        scales = transform_data[:, 7:10]  # (N, 3)

        n = len(indices)
        matrices = np.zeros((n, 16), dtype=np.float32)

        # Extract quaternion components
        qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz

        sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]

        # Row 0 (scaled)
        matrices[:, 0] = (1 - 2 * (yy + zz)) * sx
        matrices[:, 1] = 2 * (xy - wz) * sy
        matrices[:, 2] = 2 * (xz + wy) * sz
        # Row 1 (scaled)
        matrices[:, 4] = 2 * (xy + wz) * sx
        matrices[:, 5] = (1 - 2 * (xx + zz)) * sy
        matrices[:, 6] = 2 * (yz - wx) * sz
        # Row 2 (scaled)
        matrices[:, 8] = 2 * (xz - wy) * sx
        matrices[:, 9] = 2 * (yz + wx) * sy
        matrices[:, 10] = (1 - 2 * (xx + yy)) * sz
        # Row 3
        matrices[:, 15] = 1.0
        # Translation stored at ROW 0, COLUMN 3 (indices 3, 7, 11)
        # This way, after transpose to column-major, it ends up at
        # COLUMN 3 (indices 12, 13, 14) where WGSL expects it
        matrices[:, 3] = positions[:, 0]
        matrices[:, 7] = positions[:, 1]
        matrices[:, 11] = positions[:, 2]

        # Validate computed matrices
        from manifoldx.ecs import ENABLE_VALIDATION

        if ENABLE_VALIDATION:
            if np.any(np.isnan(matrices)):
                import warnings

                warnings.warn(
                    f"⚠️ NaN in transform matrices for {len(indices)} entities"
                )
            if np.any(np.isinf(matrices)):
                import warnings

                warnings.warn(
                    f"⚠️ Inf in transform matrices for {len(indices)} entities"
                )

        self._matrix_cache[indices] = matrices
        self._dirty[indices] = False


def _color_hex_to_vec4(color_hex: str) -> np.ndarray:
    """Convert hex color string to vec4 float array."""
    color_hex = color_hex.lstrip("#")
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0
    return np.array([r, g, b, 1.0], dtype=np.float32)


# =============================================================================
# Render Pipeline
# =============================================================================


class RenderPipeline:
    """
    Instanced render pipeline.

    Groups entities by (geometry_id, material_type) into batches.
    Each batch uses a single draw_indexed call with instance_count.
    Per-instance transforms are uploaded via a storage buffer.
    Material-type specific pipelines are cached.
    """

    def __init__(self, store, device=None):
        self._store = store
        self._device = device
        self._transform_cache = TransformCache(store.max_entities)
        self._pipelines = {}  # (geometry_id, material_type) -> pipeline
        self._pipeline_layouts = {}  # material_type -> layout
        self._bind_group_layouts = {}  # material_type -> bind group layout
        self._material_buffers = {}  # (geometry_id, material_type) -> buffer
        self._globals_buffer = None  # VP matrix + camera_pos
        self._transform_buffer = None  # Storage buffer for model matrices
        self._transform_buffer_size = 0
        self._lights_buffer = None  # External lights uniform buffer
        self._initialized = False

    def _ensure_pipeline(self, device, texture_format):
        """Lazily initialize device and shared buffers on first use."""
        if self._initialized:
            return

        self._device = device

        # Create globals uniform buffer: mat4x4 VP (64) + vec3 camera_pos (12) + pad (4) = 80 bytes
        self._globals_buffer = device.create_buffer(
            size=80,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Create initial transform storage buffer (will grow as needed)
        initial_size = max(64, 64 * 1024)  # At least 1 matrix, start with 1024
        self._transform_buffer = device.create_buffer(
            size=initial_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._transform_buffer_size = initial_size

        # Create lights uniform buffer (up to 4 lights, 32 bytes each = 128 bytes max)
        self._lights_buffer = device.create_buffer(
            size=128,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._initialized = True

    def _get_or_create_pipeline(
        self, device, texture_format, geometry_id, material, registry
    ):
        """Get or create a material-type specific pipeline."""
        material_type = type(material).__name__
        key = (geometry_id, material_type)

        if key in self._pipelines:
            return self._pipelines[key], self._bind_group_layouts.get(material_type)

        shader_source = material._compile()
        shader_module = device.create_shader_module(code=shader_source)

        # Determine bind group layout based on shader bindings
        needs_lights = "@binding(3)" in shader_source
        if not needs_lights:
            bind_group_entries = [
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        else:
            # StandardMaterial: 4 bindings (globals, transforms, material, lights)
            bind_group_entries = [
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]

        bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
        self._bind_group_layouts[material_type] = bind_group_layout

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        self._pipeline_layouts[material_type] = pipeline_layout

        # Always use 6-float stride (pos + normal) with both vertex attributes
        pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 6 * 4,  # 6 floats * 4 bytes (pos + normal)
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x3,
                                "offset": 0,
                                "shader_location": 0,  # position
                            },
                            {
                                "format": wgpu.VertexFormat.float32x3,
                                "offset": 3 * 4,
                                "shader_location": 1,  # normal
                            },
                        ],
                    }
                ],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": texture_format}],
            },
        )

        self._pipelines[key] = pipeline

        # Create material uniform buffer for this key
        if not needs_lights:
            buffer_size = 16  # vec4 color
        else:
            buffer_size = (
                32  # 8 floats: albedo(3) + roughness(1) + metallic(1) + ao(1) + pad(2)
            )

        material_buffer = device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._material_buffers[key] = material_buffer

        return pipeline, bind_group_layout

    def _ensure_transform_buffer(self, needed_bytes):
        """Grow transform storage buffer if needed."""
        if needed_bytes <= self._transform_buffer_size:
            return

        # Double until big enough
        new_size = self._transform_buffer_size
        while new_size < needed_bytes:
            new_size *= 2

        self._transform_buffer = self._device.create_buffer(
            size=new_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._transform_buffer_size = new_size

    def render(self, engine, render_pass):
        """Issue instanced draw calls into an active render pass."""
        if not self._initialized or self._device is None:
            return

        # Get all alive entities
        alive_indices = np.where(self._store._alive)[0]
        if len(alive_indices) == 0:
            return

        # Check required components exist
        if "Transform" not in self._store._components:
            return
        if "Mesh" not in self._store._components:
            return

        # Compute view-projection matrix from camera
        camera = engine._camera
        aspect = engine.w / engine.h
        vp = camera.get_view_projection_matrix(aspect)

        # Get transform data
        self._transform_cache.mark_dirty(alive_indices)
        model_matrices = self._transform_cache.get_transforms(
            self._store, alive_indices
        )

        # Get mesh and material IDs
        mesh_data = self._store.get_component_data("Mesh", alive_indices)
        material_data = None
        if "Material" in self._store._components:
            material_data = self._store.get_component_data("Material", alive_indices)

        # Group by (geom_id, material_type) for batched instanced drawing
        batches = {}  # (geom_id, mat_type) -> list of local indices
        for i, entity_idx in enumerate(alive_indices):
            geom_id = int(mesh_data[i, 0])
            mat_id = int(material_data[i, 0]) if material_data is not None else 0
            if geom_id == 0:
                continue

            mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
            mat_type = type(mat_obj).__name__ if mat_obj else "BasicMaterial"

            key = (geom_id, mat_type)
            if key not in batches:
                batches[key] = []
            batches[key].append(i)

        # Get camera position for globals
        camera_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        if hasattr(camera, "position"):
            camera_pos = np.array(camera.position, dtype=np.float32)

        # Upload globals (VP + camera_pos + pad) = 80 bytes
        globals_data = np.zeros(80, dtype=np.uint8)
        globals_data[0:64] = np.frombuffer(
            vp.astype(np.float32).tobytes(), dtype=np.uint8
        )
        globals_data[64:76] = np.frombuffer(
            camera_pos.astype(np.float32).tobytes(), dtype=np.uint8
        )
        # bytes 76-79 are padding (already zero)
        self._device.queue.write_buffer(self._globals_buffer, 0, globals_data.tobytes())

        # Draw each batch
        for (geom_id, mat_type), local_indices in batches.items():
            instance_count = len(local_indices)

            # Get GPU buffers for geometry
            gpu_buffers = engine._geometry_registry.get_gpu_buffers(geom_id)
            if gpu_buffers is None:
                geom_obj = engine._geometry_registry.get(geom_id)
                if geom_obj is None:
                    continue
                gpu_buffers = engine._geometry_registry.create_buffers(
                    geom_id, geom_obj, self._device.queue
                )
                if gpu_buffers is None:
                    continue

            # Get material and create/fetch pipeline
            mat_id = (
                int(material_data[local_indices[0], 0])
                if material_data is not None
                else 0
            )
            mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None

            if mat_obj is None:
                continue  # Skip entities without a material

            pipeline, bind_group_layout = self._get_or_create_pipeline(
                self._device,
                engine._texture_format,
                geom_id,
                mat_obj,
                engine._material_registry,
            )

            # Collect model matrices for this batch and transpose each for WGSL
            batch_matrices = model_matrices[local_indices]
            batch_matrices_t = (
                batch_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
            )
            transform_bytes = batch_matrices_t.astype(np.float32).tobytes()

            # Ensure buffer is big enough
            self._ensure_transform_buffer(len(transform_bytes))

            # Upload transforms
            self._device.queue.write_buffer(self._transform_buffer, 0, transform_bytes)

            # Get material data and upload to uniform buffer
            mat_data = mat_obj.get_data(instance_count, engine._material_registry)
            key = (geom_id, mat_type)
            mat_buffer = self._material_buffers.get(key)
            if mat_buffer is not None:
                # Upload first instance's material data (shared uniform for the batch)
                first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
                self._device.queue.write_buffer(
                    mat_buffer, 0, first_row.astype(np.float32).tobytes()
                )

            # Build bind group entries
            needs_lights = "@binding(3)" in type(mat_obj)._compile()
            mat_buffer_size = 32 if needs_lights else 16
            bind_group_entries = [
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._globals_buffer,
                        "offset": 0,
                        "size": 80,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._transform_buffer,
                        "offset": 0,
                        "size": self._transform_buffer_size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": mat_buffer,
                        "offset": 0,
                        "size": mat_buffer_size,
                    },
                },
            ]

            # Upload lights and add binding for PBR materials
            if needs_lights:
                lights = engine._lights
                lights_data = np.zeros(32, dtype=np.float32)  # 32 floats = 128 bytes
                for li, light in enumerate(lights[:4]):
                    light_arr = light.get_data()
                    offset = li * 8  # 8 floats per light
                    lights_data[offset : offset + len(light_arr)] = light_arr
                self._device.queue.write_buffer(
                    self._lights_buffer, 0, lights_data.tobytes()
                )

                bind_group_entries.append(
                    {
                        "binding": 3,
                        "resource": {
                            "buffer": self._lights_buffer,
                            "offset": 0,
                            "size": 128,
                        },
                    }
                )

            # Create bind group per draw call (transforms change per batch)
            bind_group = self._device.create_bind_group(
                layout=bind_group_layout,
                entries=bind_group_entries,
            )

            render_pass.set_pipeline(pipeline)
            render_pass.set_bind_group(0, bind_group)
            render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
            render_pass.set_index_buffer(
                gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32
            )

            render_pass.draw_indexed(gpu_buffers["index_count"], instance_count)

    def run(self, engine, dt: float):
        """Execute full render pipeline (CPU-side prep)."""
        if "Transform" not in self._store._components:
            return

        alive_indices = np.where(self._store._alive)[0]
        if len(alive_indices) == 0:
            return

        self._transform_cache.mark_dirty(alive_indices)
        self._transform_cache.get_transforms(self._store, alive_indices)


__all__ = [
    "TransformCache",
    "RenderPipeline",
    "SHADER_SOURCE",
]
