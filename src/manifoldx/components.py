"""Built-in components: Transform, Mesh, Material."""

import numpy as np


# =============================================================================
# Transform Component
# =============================================================================


class Transform:
    """
    Transform component storing position, rotation, scale.

    Usage:
        Transform()                    # Default transform
        Transform(pos=(0,0,0))         # With position
        Transform(pos=(0,0,0), scale=(1,1,1))  # With position and scale

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

    def __init__(self, pos=None, rot=None, scale=None):
        """Create Transform with optional position/rotation/scale."""
        self._pos = pos
        self._rot = rot
        self._scale = scale

    def get_data(self, n: int, registry=None) -> np.ndarray:
        """Get Transform data array for n entities."""
        data = np.zeros((n, 10), dtype=np.float32)

        # Set defaults
        data[:, 7:10] = 1.0  # scale = (1, 1, 1)
        data[:, 6] = 1.0  # rotation w = 1 (quaternion identity)

        # Apply user-provided values
        if self._pos is not None:
            pos = np.asarray(self._pos, dtype=np.float32)
            data[:, 0:3] = pos

        if self._scale is not None:
            scale = np.asarray(self._scale, dtype=np.float32)
            data[:, 7:10] = scale

        # Broadcast to n entities if needed
        if n > 1 and (self._pos is not None or self._scale is not None):
            # If single values provided, broadcast them
            if self._pos is not None:
                data[:, 0:3] = np.broadcast_to(self._pos, (n, 3))
            if self._scale is not None:
                data[:, 7:10] = np.broadcast_to(self._scale, (n, 3))

        return data

    @staticmethod
    def rotation(x=0, y=0, z=0, euler=None) -> np.ndarray:
        """Create quaternion(s) (x, y, z, w) from euler angles (radians).

        Uses intrinsic Tait-Bryan angles: rotate around X, then Y, then Z.

        Accepts scalars or arrays:
            Transform.rotation(x=0, y=0.5, z=0)          -> (4,) quaternion
            Transform.rotation(euler=angular_vel * dt)     -> (N, 4) quaternions
            Transform.rotation(x=np.array([...]), y=...) -> (N, 4) quaternions

        The `euler` parameter is an (N, 3) or (3,) array of [x, y, z] angles.
        """
        if euler is not None:
            euler = np.asarray(euler, dtype=np.float32)
            if euler.ndim == 1:
                x, y, z = euler[0], euler[1], euler[2]
            else:
                x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]

        cx, sx = np.cos(x / 2), np.sin(x / 2)
        cy, sy = np.cos(y / 2), np.sin(y / 2)
        cz, sz = np.cos(z / 2), np.sin(z / 2)

        # Quaternion from euler (XYZ order)
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
        qw = cx * cy * cz + sx * sy * sz

        return np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)

    @staticmethod
    def register(store):
        """Register Transform component in entity store."""
        store.register_component("Transform", np.dtype("f4"), shape=(10,))

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

    Usage:
        Mesh(cube_geometry)  # Creates component data with geometry_id

    Storage: Single uint32 (geometry_id)
    """

    def __init__(self, geometry):
        """Create Mesh component with geometry."""
        # Store geometry object for later registration
        self._geometry = geometry
        self._geometry_id = None

    def get_data(self, n: int, registry) -> np.ndarray:
        """Get component data array for n entities."""
        # Register geometry if not already
        if self._geometry_id is None:
            self._geometry_id = registry.register(self._geometry)

        # Create array with geometry_id repeated n times
        return np.full((n, 1), self._geometry_id, dtype=np.uint32)

    @staticmethod
    def register(store):
        """Register Mesh component in entity store."""
        store.register_component("Mesh", np.dtype("u4"), shape=(1,))

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

    Usage:
        Material(phong_material)  # Creates component data with material_id

    Storage: Single uint32 (material_id)
    """

    def __init__(self, material):
        """Create Material component with material."""
        self._material = material
        self._material_id = None

    def get_data(self, n: int, registry) -> np.ndarray:
        """Get component data array for n entities."""
        # Register material if not already
        if self._material_id is None:
            self._material_id = registry.register(self._material)

        return np.full((n, 1), self._material_id, dtype=np.uint32)

    @staticmethod
    def register(store):
        """Register Material component in entity store."""
        store.register_component("Material", np.dtype("u4"), shape=(1,))

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
    "Transform",
    "Mesh",
    "Material",
    "Colors",
]
