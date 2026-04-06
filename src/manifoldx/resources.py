"""GPU resource management: Geometry, Material registries and factories."""
import numpy as np
import wgpu
from typing import Any, Dict, Optional


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
        
        if 'positions' in geometry_obj:
            positions = geometry_obj['positions'].astype(np.float32)
            has_normals = 'normals' in geometry_obj
            
            if has_normals:
                normals = geometry_obj['normals'].astype(np.float32)
                # Interleave: [px, py, pz, nx, ny, nz] per vertex
                n_verts = len(positions)
                interleaved = np.zeros((n_verts, 6), dtype=np.float32)
                interleaved[:, 0:3] = positions
                interleaved[:, 3:6] = normals
                data = interleaved.tobytes()
                buffers['stride'] = 6 * 4  # 6 floats * 4 bytes
                buffers['has_normals'] = True
            else:
                data = positions.tobytes()
                buffers['stride'] = 3 * 4
                buffers['has_normals'] = False
            
            vertex_buffer = self._device.create_buffer(
                size=len(data),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )
            queue.write_buffer(vertex_buffer, 0, data)
            buffers['vertex_buffer'] = vertex_buffer
            buffers['vertex_count'] = len(positions)
            
        # Create index buffer
        if 'indices' in geometry_obj:
            indices = geometry_obj['indices']
            data = indices.astype(np.uint32).tobytes()
            
            index_buffer = self._device.create_buffer(
                size=len(data),
                usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
            )
            queue.write_buffer(index_buffer, 0, data)
            buffers['index_buffer'] = index_buffer
            buffers['index_count'] = len(indices)
            
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
    positions = np.array([
        # Front face (z+)
        [-w, -h,  d], [ w, -h,  d], [ w,  h,  d], [-w,  h,  d],
        # Back face (z-)
        [ w, -h, -d], [-w, -h, -d], [-w,  h, -d], [ w,  h, -d],
        # Right face (x+)
        [ w, -h,  d], [ w, -h, -d], [ w,  h, -d], [ w,  h,  d],
        # Left face (x-)
        [-w, -h, -d], [-w, -h,  d], [-w,  h,  d], [-w,  h, -d],
        # Top face (y+)
        [-w,  h,  d], [ w,  h,  d], [ w,  h, -d], [-w,  h, -d],
        # Bottom face (y-)
        [-w, -h, -d], [ w, -h, -d], [ w, -h,  d], [-w, -h,  d],
    ], dtype=np.float32)
    
    normals = np.array([
        # Front
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        # Back
        [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
        # Right
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        # Left
        [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
        # Top
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        # Bottom
        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
    ], dtype=np.float32)
    
    # 6 faces × 2 triangles × 3 indices = 36
    indices = np.array([
        0, 1, 2, 0, 2, 3,       # front
        4, 5, 6, 4, 6, 7,       # back
        8, 9, 10, 8, 10, 11,    # right
        12, 13, 14, 12, 14, 15, # left
        16, 17, 18, 16, 18, 19, # top
        20, 21, 22, 20, 22, 23, # bottom
    ], dtype=np.uint32)
    
    return {'positions': positions, 'normals': normals, 'indices': indices}


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
            x = cos_theta * np.cos(phi)
            y = sin_theta
            z = cos_theta * np.sin(phi)
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
    
    return {'positions': positions, 'indices': indices}


def plane(width: float, height: float) -> dict:
    """Create plane geometry."""
    w, h = width / 2, height / 2
    
    positions = np.array([
        [-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0],
    ], dtype=np.float32)
    
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    
    return {'positions': positions, 'indices': indices}


# =============================================================================
# Material Factories
# =============================================================================

class BasicMaterial:
    """Unlit material."""
    def __init__(self, color):
        self.color = color


class PhongMaterial:
    """Phong shading material."""
    def __init__(self, color, shininess: float = 32.0):
        self.color = color
        self.shininess = shininess


class StandardMaterial:
    """PBR material."""
    def __init__(self, color, roughness: float = 0.5, metallic: float = 0.0):
        self.color = color
        self.roughness = roughness
        self.metallic = metallic


def basic(color) -> BasicMaterial:
    """Create unlit material."""
    return BasicMaterial(color)


def phong(color, shininess: float = 32.0) -> PhongMaterial:
    """Create Phong shading material."""
    return PhongMaterial(color, shininess)


def standard(color, roughness: float = 0.5, metallic: float = 0.0) -> StandardMaterial:
    """Create PBR material."""
    return StandardMaterial(color, roughness, metallic)


__all__ = [
    'GeometryRegistry',
    'MaterialRegistry',
    'cube',
    'sphere', 
    'plane',
    'basic',
    'phong',
    'standard',
]
