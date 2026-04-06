"""Render pipeline that executes after command buffer."""
import numpy as np
import wgpu


# =============================================================================
# WGSL Shaders
# =============================================================================

SHADER_SOURCE = """
struct Uniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return uniforms.color;
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
        
        positions = transform_data[:, 0:3]  # (N, 3)
        scales = transform_data[:, 7:10]  # (N, 3)
        
        # Build model matrix: scale + translation (no rotation for now)
        matrices = np.zeros((len(indices), 16), dtype=np.float32)
        matrices[:, 0] = scales[:, 0]   # scale X
        matrices[:, 5] = scales[:, 1]   # scale Y
        matrices[:, 10] = scales[:, 2]  # scale Z
        matrices[:, 15] = 1.0           # w
        
        # Set positions in matrix (column-major translation)
        matrices[:, 12] = positions[:, 0]  # translate X
        matrices[:, 13] = positions[:, 1]  # translate Y
        matrices[:, 14] = positions[:, 2]  # translate Z
        
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
    Render pipeline that executes after command buffer.
    
    Flow:
    1. Query all alive entities with Mesh + Material + Transform
    2. Group by (geometry_id, material_id) into batches
    3. For each batch: compute transforms (using cache), upload, draw
    """
    
    def __init__(self, store, device=None):
        self._store = store
        self._device = device
        self._transform_cache = TransformCache(store.max_entities)
        self._pipeline = None  # WGPU render pipeline (lazy init)
        self._bind_group_layout = None
        self._pipeline_layout = None
        self._uniform_buffer = None
        self._bind_group = None
        self._initialized = False
        
    def _ensure_pipeline(self, device, texture_format):
        """Lazily create the WGPU render pipeline on first use."""
        if self._initialized:
            return
            
        self._device = device
        
        # Create shader module
        shader_module = device.create_shader_module(code=SHADER_SOURCE)
        
        # Create uniform buffer (mat4x4 + vec4 = 64 + 16 = 80 bytes)
        self._uniform_buffer = device.create_buffer(
            size=80,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        
        # Create bind group layout
        self._bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        
        # Create bind group
        self._bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._uniform_buffer,
                        "offset": 0,
                        "size": 80,
                    },
                }
            ],
        )
        
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
                        "array_stride": 3 * 4,  # 3 floats * 4 bytes
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x3,
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                    }
                ],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
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
        
    def render(self, engine, render_pass):
        """Issue draw calls into an active render pass."""
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
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)
        vp = proj @ view  # view-projection matrix
        
        # Get transform data
        self._transform_cache.mark_dirty(alive_indices)
        model_matrices = self._transform_cache.get_transforms(self._store, alive_indices)
        
        # Get mesh component data (geometry IDs)
        mesh_data = self._store.get_component_data('Mesh', alive_indices)
        
        # Get material data if available
        material_data = None
        if 'Material' in self._store._components:
            material_data = self._store.get_component_data('Material', alive_indices)
        
        # Set pipeline
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(0, self._bind_group)
        
        # Draw each entity
        for i, entity_idx in enumerate(alive_indices):
            # Get geometry ID
            geom_id = int(mesh_data[i, 0])
            if geom_id == 0:
                continue
                
            # Get GPU buffers for this geometry
            gpu_buffers = engine._geometry_registry.get_gpu_buffers(geom_id)
            if gpu_buffers is None:
                # Need to create buffers
                geom_obj = engine._geometry_registry.get(geom_id)
                if geom_obj is None:
                    continue
                gpu_buffers = engine._geometry_registry.create_buffers(
                    geom_id, geom_obj, self._device.queue
                )
                if gpu_buffers is None:
                    continue
            
            # Compute MVP for this entity
            model = model_matrices[i].reshape(4, 4)
            mvp = (vp @ model).T  # Transpose for WGSL column-major layout
            
            # Get color from material
            color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # default red
            if material_data is not None:
                mat_id = int(material_data[i, 0])
                if mat_id > 0:
                    mat_obj = engine._material_registry.get(mat_id)
                    if mat_obj is not None and hasattr(mat_obj, 'color'):
                        if isinstance(mat_obj.color, str):
                            color = _color_hex_to_vec4(mat_obj.color)
            
            # Upload uniforms (MVP + color)
            uniform_data = np.zeros(80, dtype=np.uint8)
            uniform_data[0:64] = np.frombuffer(
                mvp.astype(np.float32).tobytes(), dtype=np.uint8
            )
            uniform_data[64:80] = np.frombuffer(
                color.tobytes(), dtype=np.uint8
            )
            self._device.queue.write_buffer(
                self._uniform_buffer, 0, uniform_data.tobytes()
            )
            
            # Set vertex and index buffers
            render_pass.set_vertex_buffer(0, gpu_buffers['vertex_buffer'])
            render_pass.set_index_buffer(
                gpu_buffers['index_buffer'], wgpu.IndexFormat.uint32
            )
            
            # Draw!
            render_pass.draw_indexed(gpu_buffers['index_count'])
    
    def run(self, engine, dt: float):
        """Execute full render pipeline (CPU-side prep, no draw calls here)."""
        # Get all alive entities with required components
        if 'Transform' not in self._store._components:
            return
            
        alive_indices = np.where(self._store._alive)[0]
        if len(alive_indices) == 0:
            return
        
        # Mark transforms as dirty for all alive entities
        self._transform_cache.mark_dirty(alive_indices)
        
        # Get transforms using cache
        transforms = self._transform_cache.get_transforms(self._store, alive_indices)
        
        # Verify transforms were computed
        assert transforms.shape[1] == 16  # mat4x4


__all__ = [
    'TransformCache',
    'RenderPipeline',
    'SHADER_SOURCE',
]
