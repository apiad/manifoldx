"""Render pipeline that executes after command buffer."""

import numpy as np
import wgpu

from manifoldx.render.passes import axis as _axis_pass
from manifoldx.render.passes import label as _label_pass
from manifoldx.render.passes import mesh as _mesh_pass
from manifoldx.render.passes import skybox as _skybox_pass
from manifoldx.render.passes import sprite as _sprite_pass
from manifoldx.render.passes import volume as _volume_pass
from manifoldx.render.passes.volume import VOLUME_SHADER_SOURCE
from manifoldx.viz.materials import ColormapMaterial


# =============================================================================
# Batch Buffers (per-batch GPU buffer management)
# =============================================================================


class MaterialGeometryMismatchError(ValueError):
    """A material requires geometry attributes (e.g. UVs) the geometry doesn't have."""


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
        self.label_indices_buf = None
        self.label_indices_capacity = 0

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

    def upload_label_indices(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.label_indices_buf is None or self.label_indices_capacity < n_bytes:
            self.label_indices_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.label_indices_capacity = n_bytes
        self._device.queue.write_buffer(self.label_indices_buf, 0, data.tobytes())


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
        # Per-material-id uniform buffer for the mesh pass. Distinct from
        # _material_buffers above, which is keyed by pipeline-cache key —
        # that one shares a buffer across all material instances of the same
        # type, which silently broke per-instance material data. The mesh
        # pass now uses this dict keyed by mat_id, one buffer per material.
        self._material_buffers_by_mat_id = {}  # mat_id -> buffer
        self._globals_buffer = None  # VP matrix + view matrix + camera_pos
        self._batch_buffers = None  # _BatchBuffers instance for per-batch storage
        self._sprite_batch_buffers = None  # _BatchBuffers instance for sprite per-batch storage
        self._label_batch_buffers = None  # _BatchBuffers instance for label per-batch storage
        self._axis_batch_buffers = None  # _BatchBuffers instance for axis per-batch storage
        self._lights_buffer = None  # External lights uniform buffer
        self._lut_textures = {}  # cmap_name -> (texture, sampler) for LUT caching
        self._initialized = False

    def _ensure_pipeline(self, device, texture_format):
        """Lazily initialize device and shared buffers on first use."""
        if getattr(self, "_initialized", False):
            return

        self._device = device

        # Create globals uniform buffer:
        #   vp(64) + view(64) + proj(64) + camera_pos(12) + pad(4)
        #   + viewport_size(8) + pad(8) + ibl_intensity(4) + ibl_enabled(4) + pad(8) = 240 bytes
        #   + shadow block: light_view_proj(64) + sun(32) + shadow params(16) = 352 bytes total
        self._globals_buffer = device.create_buffer(
            size=352,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Create batch buffers for per-batch storage (transforms, etc.)
        self._batch_buffers = _BatchBuffers(device)

        # Create separate batch buffers for sprite path
        self._sprite_batch_buffers = _BatchBuffers(device)

        # Create separate batch buffers for label path
        self._label_batch_buffers = _BatchBuffers(device)

        # Create separate batch buffers for axis path
        self._axis_batch_buffers = _BatchBuffers(device)

        # Create lights uniform buffer (up to 4 lights, 32 bytes each = 128 bytes max)
        self._lights_buffer = device.create_buffer(
            size=128,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # IBL GPU resources — shared sampler + placeholder textures
        self._ibl_env_id = None
        self._ibl_sampler = device.create_sampler(
            min_filter="linear",
            mag_filter="linear",
            mipmap_filter="linear",
        )
        self._ibl_bind_group_layout = self._create_ibl_bind_group_layout(device)
        black_cube_view = self._create_black_cubemap(device)
        self._ibl_neutral_lut_view = self._create_brdf_lut_texture(device)
        self._ibl_placeholder_bind_group = self._create_ibl_bind_group(
            device, black_cube_view, black_cube_view, self._ibl_neutral_lut_view
        )
        self._ibl_active_bind_group = None

        self._initialized = True

    def _create_ibl_bind_group_layout(self, device):
        return device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.cube,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.cube,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
            ]
        )

    def _create_black_cubemap(self, device):
        """1×1×6 rgba16float black cubemap, returns texture view."""
        tex = device.create_texture(
            size=(1, 1, 6),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=1,
            sample_count=1,
        )
        data = np.zeros((6, 1, 1, 4), dtype=np.float16).tobytes()
        device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"bytes_per_row": 8, "rows_per_image": 1},
            (1, 1, 6),
        )
        return tex.create_view(
            format=wgpu.TextureFormat.rgba16float,
            dimension=wgpu.TextureViewDimension.cube,
            base_mip_level=0,
            mip_level_count=1,
            base_array_layer=0,
            array_layer_count=6,
        )

    def _create_brdf_lut_texture(self, device):
        """Upload pre-baked BRDF LUT, returns texture view."""
        from manifoldx.ibl import load_brdf_lut

        lut = load_brdf_lut().astype(np.float16)
        tex = device.create_texture(
            size=(512, 512, 1),
            format=wgpu.TextureFormat.rg16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=1,
            sample_count=1,
        )
        device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            lut.tobytes(),
            {"bytes_per_row": 512 * 4, "rows_per_image": 512},
            (512, 512, 1),
        )
        return tex.create_view(
            format=wgpu.TextureFormat.rg16float,
            dimension=wgpu.TextureViewDimension.d2,
        )

    def _create_ibl_bind_group(self, device, irr_view, pf_view, lut_view):
        return device.create_bind_group(
            layout=self._ibl_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self._ibl_sampler},
                {"binding": 1, "resource": irr_view},
                {"binding": 2, "resource": pf_view},
                {"binding": 3, "resource": lut_view},
            ],
        )

    def _upload_ibl_env(self, device, env):
        """Upload EnvironmentMap precomputed data to GPU textures, update active bind group."""
        env._precompute()
        irr = env._irradiance  # (6, 64, 64, 4) float16

        irr_tex = device.create_texture(
            size=(64, 64, 6),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=1,
            sample_count=1,
        )
        device.queue.write_texture(
            {"texture": irr_tex, "mip_level": 0, "origin": (0, 0, 0)},
            irr.tobytes(),
            {"bytes_per_row": 64 * 8, "rows_per_image": 64},
            (64, 64, 6),
        )
        irr_view = irr_tex.create_view(
            format=wgpu.TextureFormat.rgba16float,
            dimension=wgpu.TextureViewDimension.cube,
            base_mip_level=0,
            mip_level_count=1,
            base_array_layer=0,
            array_layer_count=6,
        )

        pf_tex = device.create_texture(
            size=(128, 128, 6),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=8,
            sample_count=1,
        )
        for mip, pf in enumerate(env._prefiltered):
            s = max(1, 128 >> mip)
            device.queue.write_texture(
                {"texture": pf_tex, "mip_level": mip, "origin": (0, 0, 0)},
                pf.astype(np.float16).tobytes(),
                {"bytes_per_row": s * 8, "rows_per_image": s},
                (s, s, 6),
            )
        pf_view = pf_tex.create_view(
            format=wgpu.TextureFormat.rgba16float,
            dimension=wgpu.TextureViewDimension.cube,
            base_mip_level=0,
            mip_level_count=8,
            base_array_layer=0,
            array_layer_count=6,
        )

        self._ibl_active_bind_group = self._create_ibl_bind_group(
            device, irr_view, pf_view, self._ibl_neutral_lut_view
        )
        self._ibl_env_id = id(env)

    def _get_or_create_pipeline(
        self,
        device,
        texture_format,
        geometry_id,
        material,
        registry,
        sprite=False,
        label=False,
        line=False,
        geometry_buffers=None,
    ):
        """Get or create a material-type specific pipeline.

        Pipeline cache key:
            mesh:   (geometry_id, material_type)
            sprite: (geometry_id, material_type, material_subtype, sprite)
            label:  (geometry_id, material_type, material_subtype, "label")
            line:   (geometry_id, material_type, material_subtype, "line")
        """
        self._ensure_pipeline(device, texture_format)
        material_type = type(material).__name__
        material_subtype = getattr(material, "pipeline_subtype", None)

        if line:
            key = (geometry_id, material_type, material_subtype, "line")
        elif label:
            key = (geometry_id, material_type, material_subtype, "label")
        elif sprite:
            key = (geometry_id, material_type, material_subtype, True)
        else:
            key = (geometry_id, material_type, material_subtype)

        if key in self._pipelines:
            return self._pipelines[key], self._bind_group_layouts.get(key)

        if line:
            # 3 bindings: globals, transforms, material uniform.
            # Native LineList topology — 1px lines on Vulkan/Metal/D3D12;
            # thickness honoring is deferred (Plan 3 Task 6 docstring).
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
            ]

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

            shader_module = device.create_shader_module(code=material._compile())

            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 3 * 4,  # AXIS_LINE_*: position only
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
                    "topology": wgpu.PrimitiveTopology.line_list,
                    "front_face": wgpu.FrontFace.ccw,
                    "cull_mode": wgpu.CullMode.none,
                },
                # Standard depth: lines occlude correctly with the 3D scene.
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

            # AxisMaterial uniform: rgba(16) + anchor_mode + 3 pad = 32 bytes.
            material_buffer = device.create_buffer(
                size=32,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

            return pipeline, bind_group_layout

        if label:
            # 6 bindings: globals, transforms, material uniform, label_indices,
            # atlas texture array, atlas sampler.
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
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2_array,
                    },
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

            shader_module = device.create_shader_module(code=material._compile())

            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 3 * 4,  # SPRITE_QUAD: position only
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
                    # Camera-facing billboard always points at the camera
                    # by construction; back-face culling would just be a
                    # hidden footgun.
                    "cull_mode": wgpu.CullMode.none,
                },
                # Depth-test on (labels behind opaque geometry are occluded)
                # but depth-write off (overlapping labels alpha-blend cleanly).
                depth_stencil={
                    "format": wgpu.TextureFormat.depth24plus,
                    "depth_write_enabled": False,
                    "depth_compare": wgpu.CompareFunction.less_equal,
                },
                fragment={
                    "module": shader_module,
                    "entry_point": "fs_main",
                    "targets": [
                        {
                            "format": texture_format,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.src_alpha,
                                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                    "operation": wgpu.BlendOperation.add,
                                },
                            },
                        }
                    ],
                },
            )

            self._pipelines[key] = pipeline

            material_buffer = device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

            return pipeline, bind_group_layout

        # StandardMaterial._compile accepts an optional textured= kwarg;
        # other materials' _compile() takes no args. Detect and forward.
        is_textured = (
            material_subtype == "textured" and getattr(material, "albedo_map", None) is not None
        )
        if is_textured:
            shader_source = material._compile(textured=True)
        else:
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
                # StandardMaterial: 4 bindings (globals, transforms, material, lights);
                # +2 more (sampler, texture) when subtype == "textured".
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
                if material_subtype == "textured":
                    if not (geometry_buffers or {}).get("has_uvs", False):
                        raise MaterialGeometryMismatchError(
                            f"StandardMaterial(albedo_map=...) requires geometry "
                            f"with UVs; geometry id={geometry_id} has none"
                        )
                    bind_group_entries.extend(
                        [
                            {
                                "binding": 4,
                                "visibility": wgpu.ShaderStage.FRAGMENT,
                                "sampler": {"type": wgpu.SamplerBindingType.filtering},
                            },
                            {
                                "binding": 5,
                                "visibility": wgpu.ShaderStage.FRAGMENT,
                                "texture": {
                                    "sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d2,
                                },
                            },
                        ]
                    )

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            if needs_lights:
                # StandardMaterial uses @group(1) for IBL textures
                all_layouts = [bind_group_layout, self._ibl_bind_group_layout]
            else:
                all_layouts = [bind_group_layout]
            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=all_layouts)
            self._pipeline_layouts[key] = pipeline_layout

            # Use the geometry's actual buffer stride so the pipeline
            # advances correctly over UV bytes the scalar shader doesn't read.
            geom_stride = (geometry_buffers or {}).get("stride", 6 * 4)
            vertex_attributes = [
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
            ]
            if material_subtype == "textured":
                vertex_attributes.append(
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 6 * 4,
                        "shader_location": 2,
                    }
                )
            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": geom_stride,
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": vertex_attributes,
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

    def _sun_light_view_proj(self, engine):
        """Light-space view-projection for the sun (identity until shadows are on)."""
        cfg = getattr(engine, "_shadow_config", None)
        sun = getattr(engine, "_sun", None)
        if cfg is None or sun is None:
            return np.eye(4, dtype=np.float32)
        from manifoldx.shadow import compute_light_view_proj

        return compute_light_view_proj(
            direction=np.asarray(sun.direction, dtype=np.float32),
            target=np.asarray(cfg["target"], dtype=np.float32),
            extent=cfg["extent"],
            near=cfg["near"],
            far=cfg["far"],
        )

    def render(self, engine, render_pass):
        """Issue instanced draw calls into an active render pass."""
        if not self._initialized or self._device is None:
            return

        # GUI pass runs unconditionally (even with no scene entities).
        # All other passes require alive entities and known components.
        self._render_scene_passes(engine, render_pass)

        # ---------------------------------------------------------------
        # Draw GUI pass — last in render order. Reads engine.gui directly;
        # no batches needed. No-op when engine.gui is empty.
        # ---------------------------------------------------------------
        from manifoldx.render.passes import gui as _gui_pass

        _gui_pass.render_gui_pass(self, engine, render_pass)

    def _render_scene_passes(self, engine, render_pass):
        """Issue draw calls for scene entities (mesh / sprite / volume / label / axis)."""
        # Get all alive entities
        alive_indices = np.where(self._store._alive)[0]
        if len(alive_indices) == 0:
            return

        # Check required components exist
        if "Transform" not in self._store._components:
            return

        has_mesh = "Mesh" in self._store._components
        has_point_cloud = "PointCloud" in self._store._components
        has_text_label = "TextLabel" in self._store._components
        has_axis_frame = "AxisFrame" in self._store._components
        has_volume = "Volume" in self._store._components
        if (
            not has_mesh
            and not has_point_cloud
            and not has_text_label
            and not has_axis_frame
            and not has_volume
        ):
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
        volume_data = None
        if has_volume:
            volume_data = self._store.get_component_data("Volume", alive_indices)

        # Group by (geom_id, material_type) for mesh batches
        # Group by mat_id for sprite/label/axis batches
        mesh_batches = {}  # (geom_id, mat_type) -> list of local indices
        sprite_batches = {}  # mat_id -> list of local indices
        label_batches = {}  # mat_id -> list of local indices
        axis_batches = {}  # (geom_id, mat_id) -> list of local indices
        volume_batches = []  # list of (entity_local_idx, mat_id, vol_id)

        for i, entity_idx in enumerate(alive_indices):
            mat_id = int(material_data[i, 0]) if material_data is not None else 0
            geom_id = int(mesh_data[i, 0]) if mesh_data is not None else 0

            # Per-entity routing. Priority: axis > label > sprite > mesh.
            # An entity is a sprite when PointCloud is registered AND it has no
            # Mesh geometry (geom_id == 0). Axis frames carry both AxisFrame
            # *and* an AXIS_LINE_* mesh, so they need to win over the mesh path.
            is_axis = has_axis_frame and self._is_axis_entity(entity_idx, mat_id, engine)
            is_label = (
                (not is_axis)
                and has_text_label
                and self._is_label_entity(entity_idx, mat_id, engine)
            )
            is_volume = (
                (not is_axis)
                and (not is_label)
                and has_volume
                and self._is_volume_entity(entity_idx, mat_id, engine)
            )
            is_sprite = (
                (not is_axis)
                and (not is_label)
                and (not is_volume)
                and has_point_cloud
                and geom_id == 0
            )

            if is_axis:
                key = (geom_id, mat_id)
                if key not in axis_batches:
                    axis_batches[key] = []
                axis_batches[key].append(i)
            elif is_label:
                if mat_id not in label_batches:
                    label_batches[mat_id] = []
                label_batches[mat_id].append(i)
            elif is_volume:
                vol_id = int(volume_data[i, 0]) if volume_data is not None else 0
                if vol_id > 0:
                    volume_batches.append((i, mat_id, vol_id))
            elif is_sprite:
                if mat_id not in sprite_batches:
                    sprite_batches[mat_id] = []
                sprite_batches[mat_id].append(i)
            else:
                if not has_mesh or geom_id == 0:
                    continue

                mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
                mat_type = type(mat_obj).__name__ if mat_obj else "BasicMaterial"
                mat_subtype = getattr(mat_obj, "pipeline_subtype", None) if mat_obj else None

                # mat_id is part of the key so two material *instances* of the
                # same class never share a uniform buffer.
                key = (geom_id, mat_type, mat_subtype, mat_id)
                if key not in mesh_batches:
                    mesh_batches[key] = []
                mesh_batches[key].append(i)

        # Get camera position for globals
        camera_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        if hasattr(camera, "position"):
            camera_pos = np.array(camera.position, dtype=np.float32)

        # Upload globals: vp(64) + view(64) + proj(64) + camera_pos(12) +
        # pad(4) + viewport_size(8) + pad(8) + ibl_intensity(4) + ibl_enabled(4) + pad(8) = 240 bytes,
        # then the shadow block: light_view_proj(64) + sun_direction(12)+pad(4)
        # + sun_color(12)+sun_intensity(4) + shadow_enabled(4)+bias(4)+map_size(4)+pad(4) = 112 bytes,
        # total 352 bytes.
        globals_data = np.zeros(352, dtype=np.uint8)
        globals_data[0:64] = np.frombuffer(vp.astype(np.float32).tobytes(), dtype=np.uint8)
        globals_data[64:128] = np.frombuffer(view_mat.tobytes(), dtype=np.uint8)
        proj_mat = camera.get_projection_matrix(aspect, near=camera.near, far=camera.far).T.astype(
            np.float32
        )
        globals_data[128:192] = np.frombuffer(proj_mat.tobytes(), dtype=np.uint8)
        globals_data[192:204] = np.frombuffer(
            camera_pos.astype(np.float32).tobytes(), dtype=np.uint8
        )
        # bytes 204-207: padding
        viewport_size = np.array([float(engine.w), float(engine.h)], dtype=np.float32)
        globals_data[208:216] = np.frombuffer(viewport_size.tobytes(), dtype=np.uint8)
        # bytes 216-223: padding
        ibl_env = getattr(engine, "_environment", None)
        if ibl_env is not None:
            globals_data[224:228] = np.frombuffer(
                np.float32(ibl_env.intensity).tobytes(), dtype=np.uint8
            )
            globals_data[228:232] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.uint8)
        # bytes 232-239: padding

        # --- Shadow block (offset 240+) ---
        # light_view_proj @240 — identity until a sun + enable_shadows() populate it.
        # Stored transposed (column-major), matching proj_mat / view_mat above.
        lvp = self._sun_light_view_proj(engine)  # (4,4) row-major math matrix
        globals_data[240:304] = np.frombuffer(lvp.T.astype(np.float32).tobytes(), dtype=np.uint8)
        # sun_direction+pad @304, sun_color+intensity @320 — 32 bytes from get_data().
        sun = getattr(engine, "_sun", None)
        if sun is not None:
            globals_data[304:336] = np.frombuffer(
                sun.get_data().astype(np.float32).tobytes(), dtype=np.uint8
            )
        # shadow_enabled/bias/map_size @336 — populated once shadows sample (Task 4).
        cfg = getattr(engine, "_shadow_config", None)
        if cfg is not None and sun is not None:
            globals_data[336:340] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.uint8)
            globals_data[340:344] = np.frombuffer(np.float32(cfg["bias"]).tobytes(), dtype=np.uint8)
            globals_data[344:348] = np.frombuffer(
                np.float32(cfg["resolution"]).tobytes(), dtype=np.uint8
            )

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
        # Skybox (renders at far plane — before mesh geometry)
        # ---------------------------------------------------------------
        env = getattr(engine, "_environment", None)
        if env is not None and env.show_skybox:
            _skybox_pass.render_skybox(self, engine, render_pass)

        # ---------------------------------------------------------------
        # Draw mesh batches (if any)
        # ---------------------------------------------------------------
        if mesh_batches:
            _mesh_pass.render_mesh_batches(
                self, engine, render_pass, mesh_batches, model_matrices, material_data
            )

        # ---------------------------------------------------------------
        # Draw sprite batches (if any)
        # ---------------------------------------------------------------
        if sprite_batches:
            _sprite_pass.render_sprite_batches(
                self, engine, render_pass, sprite_batches, model_matrices, material_data
            )

        # ---------------------------------------------------------------
        # Draw volume batches (depth-test on, depth-write off, alpha-blend on)
        # ---------------------------------------------------------------
        if volume_batches:
            _volume_pass.render_volume_pass(self, engine, render_pass, volume_batches)

        # ---------------------------------------------------------------
        # Draw label batches (depth-write off, alpha-blend on)
        # ---------------------------------------------------------------
        if label_batches:
            _label_pass.render_label_pass(
                self, engine, render_pass, label_batches, model_matrices, material_data
            )

        # ---------------------------------------------------------------
        # Draw axis batches (LineList topology, opaque)
        # ---------------------------------------------------------------
        if axis_batches:
            _axis_pass.render_axis_pass(
                self, engine, render_pass, axis_batches, model_matrices, material_data
            )

    def _is_axis_entity(self, entity_idx, mat_id, engine):
        if mat_id <= 0:
            return False
        mat_obj = engine._material_registry.get(mat_id)
        if mat_obj is None:
            return False
        from manifoldx.viz import AxisMaterial

        return isinstance(mat_obj, AxisMaterial)

    def _is_label_entity(self, entity_idx, mat_id, engine):
        if mat_id <= 0:
            return False
        mat_obj = engine._material_registry.get(mat_id)
        if mat_obj is None:
            return False
        from manifoldx.viz import LabelMaterial

        return isinstance(mat_obj, LabelMaterial)

    def _is_volume_entity(self, entity_idx, mat_id, engine):
        if mat_id <= 0:
            return False
        mat_obj = engine._material_registry.get(mat_id)
        if mat_obj is None:
            return False
        from manifoldx.viz import VolumeMaterial

        return isinstance(mat_obj, VolumeMaterial)

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
    "VOLUME_SHADER_SOURCE",
]
