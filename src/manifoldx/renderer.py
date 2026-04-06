"""Render pipeline that executes after command buffer."""
import numpy as np
import wgpu


# =============================================================================
# WGSL Shaders
# =============================================================================

SHADER_SOURCE = """
struct Globals {
    vp: mat4x4<f32>,
    color: vec4<f32>,
    light_dir: vec4<f32>,   // xyz = normalized direction, w = ambient intensity
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
    let mvp = globals.vp * model;
    out.position = mvp * vec4<f32>(in.position, 1.0);
    // Transform normal by model matrix (ignore translation, assume uniform scale)
    out.world_normal = (model * vec4<f32>(in.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(globals.light_dir.xyz);
    let ambient = globals.light_dir.w;
    let diffuse = max(dot(normal, light_dir), 0.0);
    let brightness = ambient + (1.0 - ambient) * diffuse;
    return vec4<f32>(globals.color.rgb * brightness, globals.color.a);
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
            
        transform_data = store.get_component_data('Transform', indices)
        
        positions = transform_data[:, 0:3]   # (N, 3)
        quats = transform_data[:, 3:7]        # (N, 4) - quaternion (x, y, z, w)
        scales = transform_data[:, 7:10]      # (N, 3)
        
        n = len(indices)
        matrices = np.zeros((n, 16), dtype=np.float32)
        
        # Extract quaternion components
        qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        
        sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]
        
        # Row 0 (scaled)
        matrices[:, 0] = (1 - 2*(yy + zz)) * sx
        matrices[:, 1] = 2*(xy - wz) * sy
        matrices[:, 2] = 2*(xz + wy) * sz
        # Row 1 (scaled)
        matrices[:, 4] = 2*(xy + wz) * sx
        matrices[:, 5] = (1 - 2*(xx + zz)) * sy
        matrices[:, 6] = 2*(yz - wx) * sz
        # Row 2 (scaled)
        matrices[:, 8] = 2*(xz - wy) * sx
        matrices[:, 9] = 2*(yz + wx) * sy
        matrices[:, 10] = (1 - 2*(xx + yy)) * sz
        # Row 3
        matrices[:, 15] = 1.0
        # Translation (column 3)
        matrices[:, 12] = positions[:, 0]
        matrices[:, 13] = positions[:, 1]
        matrices[:, 14] = positions[:, 2]
        
        self._matrix_cache[indices] = matrices
        self._dirty[indices] = False


def _color_hex_to_vec4(color_hex: str) -> np.ndarray:
    """Convert hex color string to vec4 float array."""
    color_hex = color_hex.lstrip('#')
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
    
    Groups entities by (geometry_id, material_id) into batches.
    Each batch uses a single draw_indexed call with instance_count.
    Per-instance transforms are uploaded via a storage buffer.
    """
    
    def __init__(self, store, device=None):
        self._store = store
        self._device = device
        self._transform_cache = TransformCache(store.max_entities)
        self._pipeline = None
        self._bind_group_layout = None
        self._pipeline_layout = None
        self._globals_buffer = None  # VP matrix + color (80 bytes)
        self._transform_buffer = None  # Storage buffer for model matrices
        self._transform_buffer_size = 0
        self._bind_group = None
        self._initialized = False
        
    def _ensure_pipeline(self, device, texture_format):
        """Lazily create the WGPU render pipeline on first use."""
        if self._initialized:
            return
            
        self._device = device
        
        # Create shader module
        shader_module = device.create_shader_module(code=SHADER_SOURCE)
        
        # Create globals uniform buffer (mat4x4 VP + vec4 color + vec4 light_dir = 64 + 16 + 16 = 96 bytes)
        self._globals_buffer = device.create_buffer(
            size=96,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        
        # Create initial transform storage buffer (will grow as needed)
        initial_size = max(64, 64 * 1024)  # At least 1 matrix, start with 1024
        self._transform_buffer = device.create_buffer(
            size=initial_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._transform_buffer_size = initial_size
        
        # Create bind group layout
        self._bind_group_layout = device.create_bind_group_layout(
            entries=[
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
            ]
        )
        
        self._create_bind_group()
        
        # Create pipeline layout
        self._pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        
        # Create render pipeline
        self._pipeline = device.create_render_pipeline(
            layout=self._pipeline_layout,
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
                "targets": [
                    {
                        "format": texture_format,
                    }
                ],
            },
        )
        
        self._initialized = True
    
    def _create_bind_group(self):
        """Create or recreate bind group (needed when transform buffer changes)."""
        self._bind_group = self._device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._globals_buffer,
                        "offset": 0,
                        "size": 96,
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
            ],
        )
    
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
        self._create_bind_group()  # Recreate with new buffer
        
    def render(self, engine, render_pass):
        """Issue instanced draw calls into an active render pass."""
        if not self._initialized or self._device is None:
            return
            
        # Get all alive entities
        alive_indices = np.where(self._store._alive)[0]
        if len(alive_indices) == 0:
            return
            
        # Check required components exist
        if 'Transform' not in self._store._components:
            return
        if 'Mesh' not in self._store._components:
            return
            
        # Compute view-projection matrix from camera
        camera = engine._camera
        aspect = engine.w / engine.h
        vp = camera.get_view_projection_matrix(aspect)
        
        # Get transform data
        self._transform_cache.mark_dirty(alive_indices)
        model_matrices = self._transform_cache.get_transforms(self._store, alive_indices)
        
        # Get mesh and material IDs
        mesh_data = self._store.get_component_data('Mesh', alive_indices)
        material_data = None
        if 'Material' in self._store._components:
            material_data = self._store.get_component_data('Material', alive_indices)
        
        # Group by (geom_id, mat_id) for batched instanced drawing
        batches = {}  # (geom_id, mat_id) -> list of local indices
        for i, entity_idx in enumerate(alive_indices):
            geom_id = int(mesh_data[i, 0])
            mat_id = int(material_data[i, 0]) if material_data is not None else 0
            if geom_id == 0:
                continue
            key = (geom_id, mat_id)
            if key not in batches:
                batches[key] = []
            batches[key].append(i)
        
        # Set pipeline once
        render_pass.set_pipeline(self._pipeline)
        
        # Draw each batch
        for (geom_id, mat_id), local_indices in batches.items():
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
            
            # Get color from material
            color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if mat_id > 0:
                mat_obj = engine._material_registry.get(mat_id)
                if mat_obj is not None and hasattr(mat_obj, 'color'):
                    if isinstance(mat_obj.color, str):
                        color = _color_hex_to_vec4(mat_obj.color)
            
            # Upload globals (VP + color + light_dir)
            # Light direction: from upper-right, w = ambient intensity
            light_dir = np.array([0.5773, 0.5773, 0.5773, 0.3], dtype=np.float32)  # normalized (1,1,1), ambient=0.3
            
            globals_data = np.zeros(96, dtype=np.uint8)
            globals_data[0:64] = np.frombuffer(vp.astype(np.float32).tobytes(), dtype=np.uint8)
            globals_data[64:80] = np.frombuffer(color.tobytes(), dtype=np.uint8)
            globals_data[80:96] = np.frombuffer(light_dir.tobytes(), dtype=np.uint8)
            self._device.queue.write_buffer(self._globals_buffer, 0, globals_data.tobytes())
            
            # Collect model matrices for this batch and transpose each for WGSL
            batch_matrices = model_matrices[local_indices]  # (instance_count, 16)
            # Transpose each 4x4 matrix for column-major WGSL layout
            batch_matrices_t = batch_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
            transform_bytes = batch_matrices_t.astype(np.float32).tobytes()
            
            # Ensure buffer is big enough
            self._ensure_transform_buffer(len(transform_bytes))
            
            # Upload transforms
            self._device.queue.write_buffer(self._transform_buffer, 0, transform_bytes)
            
            # Bind
            render_pass.set_bind_group(0, self._bind_group)
            render_pass.set_vertex_buffer(0, gpu_buffers['vertex_buffer'])
            render_pass.set_index_buffer(gpu_buffers['index_buffer'], wgpu.IndexFormat.uint32)
            
            # Single instanced draw call!
            render_pass.draw_indexed(gpu_buffers['index_count'], instance_count)
    
    def run(self, engine, dt: float):
        """Execute full render pipeline (CPU-side prep)."""
        if 'Transform' not in self._store._components:
            return
            
        alive_indices = np.where(self._store._alive)[0]
        if len(alive_indices) == 0:
            return
        
        self._transform_cache.mark_dirty(alive_indices)
        self._transform_cache.get_transforms(self._store, alive_indices)


__all__ = [
    'TransformCache',
    'RenderPipeline',
    'SHADER_SOURCE',
]
