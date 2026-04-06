"""Built-in components: Transform, Mesh, Material."""
import numpy as np


# =============================================================================
# Transform Component
# =============================================================================

class Transform:
    """
    Transform component storing position, rotation, scale.
    
    Storage layout (10 floats per entity):
    - position: vec3 (columns 0-2)
    - rotation: vec4 quaternion (columns 3-6)
    - scale: vec3 (columns 7-9)
    
    Default values:
    - position: (0, 0, 0)
    - rotation: (0, 0, 0, 1) - identity quaternion
    - scale: (1, 1, 1)
    """
    # Combined shape: position(3) + rotation(4) + scale(3) = 10 floats
    
    @staticmethod
    def register(store):
        """Register Transform component in entity store."""
        store.register_component("Transform", np.dtype('f4'), shape=(10,))
        
    @staticmethod
    def get_default_data(n: int) -> np.ndarray:
        """Get default Transform data for n entities."""
        # Default: position=(0,0,0), rotation=(0,0,0,1), scale=(1,1,1)
        data = np.zeros((n, 10), dtype=np.float32)
        data[:, 7:10] = 1.0  # Default scale to (1, 1, 1)
        # Default rotation is (0, 0, 0, 1) - already zeros with last element 1
        data[:, 6] = 1.0  # w component of quaternion
        return data


# =============================================================================
# Mesh Component
# =============================================================================

class Mesh:
    """
    Mesh component storing reference to geometry.
    
    Storage: Single uint32 (geometry_id)
    """
    
    @staticmethod
    def register(store):
        """Register Mesh component in entity store."""
        store.register_component("Mesh", np.dtype('u4'), shape=(1,))
        
    @staticmethod
    def get_default_data(n: int) -> np.ndarray:
        """Get default Mesh data (0 = no geometry)."""
        return np.zeros((n, 1), dtype=np.uint32)


# =============================================================================
# Material Component  
# =============================================================================

class Material:
    """
    Material component storing reference to material.
    
    Storage: Single uint32 (material_id)
    """
    
    @staticmethod
    def register(store):
        """Register Material component in entity store."""
        store.register_component("Material", np.dtype('u4'), shape=(1,))
        
    @staticmethod
    def get_default_data(n: int) -> np.ndarray:
        """Get default Material data (0 = no material)."""
        return np.zeros((n, 1), dtype=np.uint32)


# =============================================================================
# Built-in Colors
# =============================================================================

class Colors:
    """Color constants."""
    RED = "#ff0000"
    GREEN = "#00ff00"
    BLUE = "#0000ff"
    WHITE = "#ffffff"
    BLACK = "#000000"
    YELLOW = "#ffff00"
    CYAN = "#00ffff"
    MAGENTA = "#ff00ff"


__all__ = [
    'Transform',
    'Mesh', 
    'Material',
    'Colors',
]
