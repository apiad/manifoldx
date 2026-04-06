"""GPU resource management: Geometry, Material registries and factories."""
import numpy as np
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
        self._geometries: Dict[int, Any] = {}  # id -> GPU resource
        self._object_to_id: Dict[int, int] = {}  # object id -> registry id
        self._next_id = 1
        
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
    Create cube geometry.
    
    Returns dict with 'positions' and 'indices' arrays.
    """
    w, h, d = width / 2, height / 2, depth / 2
    
    # 8 vertices of cube
    positions = np.array([
        [-w, -h,  d], [ w, -h,  d], [ w,  h,  d], [-w,  h,  d],  # front
        [-w, -h, -d], [ w, -h, -d], [ w,  h, -d], [-w,  h, -d],  # back
    ], dtype=np.float32)
    
    # 12 triangles (36 indices)
    indices = np.array([
        # front
        0, 1, 2, 0, 2, 3,
        # right
        1, 5, 6, 1, 6, 2,
        # back
        5, 4, 7, 5, 7, 6,
        # left
        4, 0, 3, 4, 3, 7,
        # top
        3, 2, 6, 3, 6, 7,
        # bottom
        4, 5, 1, 4, 1, 0,
    ], dtype=np.uint32)
    
    return {'positions': positions, 'indices': indices}


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
