"""Render pipeline that executes after command buffer."""

import numpy as np
import wgpu

from manifoldx.viz.materials import ColormapMaterial


# =============================================================================
# Batch Buffers (per-batch GPU buffer management)
# =============================================================================


class _BatchBuffers:
    """Per-batch GPU buffers managed by RenderPipeline.

    Extends transform-only management to include optional per-instance
    scalar_values and radii buffers required by ColormapMaterial.
    """

    def __init__(self, device):
        self._device = device
        self.transforms_buf = None
        self.transforms_capacity = 0
        self.scalar_values_buf = None
        self.scalar_values_capacity = 0
        self.radii_buf = None
        self.radii_capacity = 0

    def upload_transforms(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.transforms_buf is None or self.transforms_capacity < n_bytes:
            self.transforms_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.transforms_capacity = n_bytes
        self._device.queue.write_buffer(self.transforms_buf, 0, data.tobytes())

    def upload_scalar_values(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.scalar_values_buf is None or self.scalar_values_capacity < n_bytes:
            self.scalar_values_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.scalar_values_capacity = n_bytes
        self._device.queue.write_buffer(self.scalar_values_buf, 0, data.tobytes())

    def upload_radii(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.radii_buf is None or self.radii_capacity < n_bytes:
            self.radii_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.radii_capacity = n_bytes
        self._device.queue.write_buffer(self.radii_buf, 0, data.tobytes())


# =============================================================================
# WGSL Shaders (default fallback - not used when materials provide shaders)
# =============================================================================

SHADER_SOURCE = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
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

                warnings.warn(f"⚠️ NaN in transform matrices for {len(indices)} entities")
            if np.any(np.isinf(matrices)):
                import warnings

                warnings.warn(f"⚠️ Inf in transform matrices for {len(indices)} entities")

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
        self._globals_buffer = None  # VP matrix + view matrix + camera_pos
        self._batch_buffers = None  # _BatchBuffers instance for per-batch storage
        self._sprite_batch_buffers = None  # _BatchBuffers instance for sprite per-batch storage
        self._lights_buffer = None  # External lights uniform buffer
        self._lut_textures = {}  # cmap_name -> (texture, sampler) for LUT caching
        self._initialized = False

    def _ensure_pipeline(self, device, texture_format):
        """Lazily initialize device and shared buffers on first use."""
        if self._initialized:
            return

        self._device = device

        # Create globals uniform buffer:
        #   vp(64) + view(64) + proj(64) + camera_pos(12) + pad(4) = 208 bytes
        self._globals_buffer = device.create_buffer(
            size=208,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Create batch buffers for per-batch storage (transforms, etc.)
        self._batch_buffers = _BatchBuffers(device)

        # Create separate batch buffers for sprite path
        self._sprite_batch_buffers = _BatchBuffers(device)

        # Create lights uniform buffer (up to 4 lights, 32 bytes each = 128 bytes max)
        self._lights_buffer = device.create_buffer(
            size=128,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._initialized = True

    def _get_or_create_pipeline(self, device, texture_format, geometry_id, material, registry, sprite=False):
        """Get or create a material-type specific pipeline.

        Pipeline cache key:
            mesh:  (geometry_id, material_type)
            sprite: (geometry_id, material_type, material_subtype, sprite)
        """
        material_type = type(material).__name__
        material_subtype = getattr(material, "pipeline_subtype", None)

        if sprite:
            key = (geometry_id, material_type, material_subtype, True)
        else:
            key = (geometry_id, material_type)

        if key in self._pipelines:
            return self._pipelines[key], self._bind_group_layouts.get(key)

        shader_source = material._compile()
        shader_module = device.create_shader_module(code=shader_source)

        if sprite:
            # Sprite pipeline: 7 bindings (0-6) with texture+sampbler
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
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d1,
                    },
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

            # Sprite vertex layout: position only (3 floats, stride=12)
            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 3 * 4,  # position only
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x3,
                                    "offset": 0,
                                    "shader_location": 0,
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

            # Create material uniform buffer for this sprite key
            buffer_size = 16  # 4 floats: vmin, vmax, lit_flag, _pad
            material_buffer = device.create_buffer(
                size=buffer_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

        else:
            # Existing mesh pipeline logic
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
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

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
                                    "shader_location": 0,
                                },
                                {
                                    "format": wgpu.VertexFormat.float32x3,
                                    "offset": 3 * 4,
                                    "shader_location": 1,
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
                buffer_size = 32  # 8 floats: albedo(3) + roughness(1) + metallic(1) + ao(1) + pad(2)

            material_buffer = device.create_buffer(
                size=buffer_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

        return pipeline, bind_group_layout

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

        has_mesh = "Mesh" in self._store._components
        has_point_cloud = "PointCloud" in self._store._components
        if not has_mesh and not has_point_cloud:
            return

        # Compute view-projection matrix from camera
        camera = engine._camera
        aspect = engine.w / engine.h
        vp = camera.get_view_projection_matrix(aspect)

        # Compute view matrix from camera (column-major for WGSL)
        view_mat = camera.get_view_matrix().T.astype(np.float32)

        # Get transform data
        self._transform_cache.mark_dirty(alive_indices)
        model_matrices = self._transform_cache.get_transforms(self._store, alive_indices)

        # Get mesh and material IDs
        mesh_data = None
        if has_mesh:
            mesh_data = self._store.get_component_data("Mesh", alive_indices)
        material_data = None
        if "Material" in self._store._components:
            material_data = self._store.get_component_data("Material", alive_indices)

        # Group by (geom_id, material_type) for mesh batches
        # Group by mat_id for sprite batches (geometry is always SPRITE_QUAD)
        mesh_batches = {}  # (geom_id, mat_type) -> list of local indices
        sprite_batches = {}  # mat_id -> list of local indices

        for i, entity_idx in enumerate(alive_indices):
            mat_id = int(material_data[i, 0]) if material_data is not None else 0

            # Determine if this entity is a sprite (has PointCloud component)
            is_sprite = has_point_cloud and self._store._alive[entity_idx]

            if is_sprite:
                if mat_id not in sprite_batches:
                    sprite_batches[mat_id] = []
                sprite_batches[mat_id].append(i)
            else:
                if not has_mesh:
                    continue
                geom_id = int(mesh_data[i, 0])
                if geom_id == 0:
                    continue

                mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
                mat_type = type(mat_obj).__name__ if mat_obj else "BasicMaterial"

                key = (geom_id, mat_type)
                if key not in mesh_batches:
                    mesh_batches[key] = []
                mesh_batches[key].append(i)

        # Get camera position for globals
        camera_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        if hasattr(camera, "position"):
            camera_pos = np.array(camera.position, dtype=np.float32)

        # Upload globals (vp + view + proj + camera_pos + pad) = 208 bytes
        globals_data = np.zeros(208, dtype=np.uint8)
        globals_data[0:64] = np.frombuffer(vp.astype(np.float32).tobytes(), dtype=np.uint8)
        globals_data[64:128] = np.frombuffer(view_mat.tobytes(), dtype=np.uint8)
        # Upload projection matrix (column-major) at offset 128. Use the camera's
        # own near/far rather than the helper's defaults so the sprite path
        # matches the depth range the rest of the pipeline uses.
        proj_mat = camera.get_projection_matrix(
            aspect, near=camera.near, far=camera.far
        ).T.astype(np.float32)
        globals_data[128:192] = np.frombuffer(proj_mat.tobytes(), dtype=np.uint8)
        globals_data[192:204] = np.frombuffer(camera_pos.astype(np.float32).tobytes(), dtype=np.uint8)
        # bytes 204-207 are padding (already zero)
        self._device.queue.write_buffer(self._globals_buffer, 0, globals_data.tobytes())

        # Upload lights once (shared across all PBR draws)
        lights = getattr(engine, "_lights", [])
        lights_data = np.zeros(32, dtype=np.float32)  # 32 floats = 128 bytes
        for li, light in enumerate(lights[:4]):
            light_arr = light.get_data()
            offset = li * 8  # 8 floats per light
            lights_data[offset : offset + len(light_arr)] = light_arr
        self._device.queue.write_buffer(self._lights_buffer, 0, lights_data.tobytes())

        # ---------------------------------------------------------------
        # Draw mesh batches (if any)
        # ---------------------------------------------------------------
        if mesh_batches:
            self._render_mesh_batches(
                engine, render_pass, mesh_batches, model_matrices, material_data
            )

        # ---------------------------------------------------------------
        # Draw sprite batches (if any)
        # ---------------------------------------------------------------
        if sprite_batches:
            self._render_sprite_batches(
                engine, render_pass, sprite_batches, model_matrices, material_data
            )

    def _render_mesh_batches(
        self, engine, render_pass, mesh_batches, model_matrices, material_data
    ):
        """Render all mesh batches using instanced draw with shared transform buffer."""
        # ---------------------------------------------------------------
        # Upload ALL transforms at once (queue.write_buffer happens
        # before GPU processes the command buffer, so per-batch writes
        # would be overwritten by the last batch).
        # ---------------------------------------------------------------

        # Flatten batch order and record (first_instance, instance_count) per batch
        all_local_indices = []  # ordered list of local indices across all batches
        batch_draw_info = {}  # key -> (first_instance, instance_count)
        instance_offset = 0

        for key, local_indices in mesh_batches.items():
            count = len(local_indices)
            batch_draw_info[key] = (instance_offset, count)
            all_local_indices.extend(local_indices)
            instance_offset += count

        if not all_local_indices:
            return

        # Transpose all matrices for WGSL column-major layout and upload once
        all_matrices = model_matrices[all_local_indices]  # (total_instances, 16)
        all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        self._batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))

        # ---------------------------------------------------------------
        # Draw each batch using first_instance to index into the
        # shared transform buffer.
        # ---------------------------------------------------------------
        for (geom_id, mat_type), local_indices in mesh_batches.items():
            first_instance, instance_count = batch_draw_info[(geom_id, mat_type)]

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
            mat_id = int(material_data[local_indices[0], 0]) if material_data is not None else 0
            mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None

            if mat_obj is None:
                continue

            pipeline, bind_group_layout = self._get_or_create_pipeline(
                self._device,
                engine._texture_format,
                geom_id,
                mat_obj,
                engine._material_registry,
            )

            # Upload material uniforms for this batch
            mat_data = mat_obj.get_data(instance_count, engine._material_registry)
            bkey = (geom_id, mat_type)
            mat_buffer = self._material_buffers.get(bkey)
            if mat_buffer is not None:
                first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
                self._device.queue.write_buffer(
                    mat_buffer, 0, first_row.astype(np.float32).tobytes()
                )

            # Build bind group
            needs_lights = "@binding(3)" in type(mat_obj)._compile()
            mat_buffer_size = 32 if needs_lights else 16
            bind_group_entries = [
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._globals_buffer,
                        "offset": 0,
                        "size": 208,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._batch_buffers.transforms_buf,
                        "offset": 0,
                        "size": self._batch_buffers.transforms_capacity,
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

            if needs_lights:
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

            bind_group = self._device.create_bind_group(
                layout=bind_group_layout,
                entries=bind_group_entries,
            )

            render_pass.set_pipeline(pipeline)
            render_pass.set_bind_group(0, bind_group)
            render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
            render_pass.set_index_buffer(gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32)

            # first_instance offsets into the shared transform buffer
            render_pass.draw_indexed(
                gpu_buffers["index_count"],
                instance_count,
                first_index=0,
                base_vertex=0,
                first_instance=first_instance,
            )

    def _render_sprite_batches(
        self, engine, render_pass, sprite_batches, model_matrices, material_data
    ):
        """Render all sprite batches (PointCloud entities).

        Each batch is one (mat_id) group; geometry is always SPRITE_QUAD.
        Per-instance buffers: transforms, scalar_values, radii.
        """
        # Flatten all sprite local indices for single buffer upload
        all_local_indices = []  # ordered list across sprite batches
        batch_draw_info = {}  # mat_id -> (first_instance, instance_count)
        instance_offset = 0

        for mat_id, local_indices in sprite_batches.items():
            count = len(local_indices)
            batch_draw_info[mat_id] = (instance_offset, count)
            all_local_indices.extend(local_indices)
            instance_offset += count

        if not all_local_indices:
            return

        # Collect per-instance arrays
        ent_arr = np.asarray(all_local_indices, dtype=np.int64)

        # Transform matrices
        all_matrices = model_matrices[ent_arr]  # (total_instances, 16)
        all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        self._sprite_batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))

        # Scalar values (must be registered if we're rendering sprites)
        if "ScalarValue" in self._store._components:
            # Get entity indices (not local indices) for component lookup
            alive_indices = np.where(self._store._alive)[0]
            entity_indices = alive_indices[ent_arr]
            scalar_data = self._store.get_component_data("ScalarValue", entity_indices)
            self._sprite_batch_buffers.upload_scalar_values(
                scalar_data[:, 0].astype(np.float32)
            )
        else:
            scalar_data = np.zeros((len(ent_arr),), dtype=np.float32)
            self._sprite_batch_buffers.upload_scalar_values(scalar_data)

        # Radii
        if "Radius" in self._store._components:
            alive_indices = np.where(self._store._alive)[0]
            entity_indices = alive_indices[ent_arr]
            radius_data = self._store.get_component_data("Radius", entity_indices)
            self._sprite_batch_buffers.upload_radii(
                radius_data[:, 0].astype(np.float32)
            )
        else:
            radius_data = np.ones((len(ent_arr),), dtype=np.float32)
            self._sprite_batch_buffers.upload_radii(radius_data)

        # Get sprite quad geometry and create buffers if needed
        sprite_geom_id = engine._geometry_registry.get_id("sprite_quad")
        gpu_buffers = engine._geometry_registry.get_gpu_buffers(sprite_geom_id)
        if gpu_buffers is None:
            geom_obj = engine._geometry_registry.get(sprite_geom_id)
            if geom_obj is not None:
                gpu_buffers = engine._geometry_registry.create_buffers(
                    sprite_geom_id, geom_obj, self._device.queue
                )
        if gpu_buffers is None:
            return

        # ---------------------------------------------------------------
        # Draw each sprite batch
        # ---------------------------------------------------------------
        for mat_id, local_indices in sprite_batches.items():
            mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
            if mat_obj is None:
                continue

            if not isinstance(mat_obj, ColormapMaterial):
                raise TypeError(
                    f"sprite batch material must be ColormapMaterial; "
                    f"got {type(mat_obj).__name__}"
                )

            first_instance, instance_count = batch_draw_info[mat_id]

            # Create or fetch sprite pipeline
            pipeline, bind_group_layout = self._get_or_create_pipeline(
                self._device,
                engine._texture_format,
                sprite_geom_id,
                mat_obj,
                engine._material_registry,
                sprite=True,
            )

            # Upload material uniforms
            mat_data = mat_obj.get_data(instance_count, engine._material_registry)
            material_type = type(mat_obj).__name__
            material_subtype = getattr(mat_obj, "pipeline_subtype", None)
            bkey = (sprite_geom_id, material_type, material_subtype, True)
            mat_buffer = self._material_buffers.get(bkey)
            if mat_buffer is not None:
                first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
                self._device.queue.write_buffer(
                    mat_buffer, 0, first_row.astype(np.float32).tobytes()
                )

            # Build sprite bind group
            bind_group = self._make_sprite_bind_group(
                bind_group_layout, mat_obj, mat_buffer
            )

            render_pass.set_pipeline(pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 0)
            render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
            render_pass.set_index_buffer(
                gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32
            )
            render_pass.draw_indexed(
                gpu_buffers["index_count"],
                instance_count,
                first_index=0,
                base_vertex=0,
                first_instance=first_instance,
            )

    def _get_or_create_lut_texture(self, device, material):
        """Create or retrieve a cached 1D LUT texture + sampler for a colormap."""
        cmap_name = material.cmap
        if cmap_name in self._lut_textures:
            return self._lut_textures[cmap_name]

        lut = material.get_lut()  # (256, 4) uint8 — matplotlib-encoded sRGB
        # Use rgba8unorm-srgb so the GPU sRGB-decodes on sample. The framebuffer
        # is also sRGB-encoded on write, so the round trip preserves the
        # author-intended display colors. Without -srgb the gamma curve gets
        # applied twice and colors come out brighter than matplotlib's swatch.
        texture = device.create_texture(
            size=(256, 1, 1),
            format=wgpu.TextureFormat.rgba8unorm_srgb,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d1,
        )
        device.queue.write_texture(
            {"texture": texture},
            lut.tobytes(),
            {"bytes_per_row": 256 * 4, "rows_per_image": 1},
            (256, 1, 1),
        )
        sampler = device.create_sampler(
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
        )
        self._lut_textures[cmap_name] = (texture, sampler)
        return self._lut_textures[cmap_name]

    def _make_sprite_bind_group(self, bind_group_layout, material, mat_buffer):
        """Create bind group with bindings 0-6 for sprite rendering.

        Bindings:
            0: globals uniform (vp, view, camera_pos)
            1: transforms storage
            2: material uniform (vmin, vmax, lit_flag, _pad)
            3: scalar_values storage
            4: radii storage
            5: lut_texture (1D RGBA8)
            6: lut_sampler
        """
        lut_texture, lut_sampler = self._get_or_create_lut_texture(
            self._device, material
        )

        mat_buffer_size = 16  # 4 floats: vmin, vmax, lit_flag, _pad

        bind_group_entries = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._globals_buffer,
                    "offset": 0,
                    "size": 208,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": self._sprite_batch_buffers.transforms_buf,
                    "offset": 0,
                    "size": self._sprite_batch_buffers.transforms_capacity,
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
            {
                "binding": 3,
                "resource": {
                    "buffer": self._sprite_batch_buffers.scalar_values_buf,
                    "offset": 0,
                    "size": self._sprite_batch_buffers.scalar_values_capacity,
                },
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": self._sprite_batch_buffers.radii_buf,
                    "offset": 0,
                    "size": self._sprite_batch_buffers.radii_capacity,
                },
            },
            {
                "binding": 5,
                "resource": lut_texture.create_view(),
            },
            {
                "binding": 6,
                "resource": lut_sampler,
            },
        ]

        return self._device.create_bind_group(
            layout=bind_group_layout,
            entries=bind_group_entries,
        )

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
