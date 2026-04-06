"""Type definitions for ManifoldX ECS engine."""

import numpy as np


# =============================================================================
# Vector Types
# =============================================================================


class Vector3(np.ndarray):
    """3D vector type for positions, velocities, etc.

    Usage:
        v = Vector3([1, 2, 3])
        result = v + other_v  # Vectorized operations
    """

    def __new__(cls, value=None):
        if value is None:
            value = [0, 0, 0]
        value = np.asarray(value, dtype=np.float32)
        if value.shape == ():
            value = np.array([value.item(), 0, 0], dtype=np.float32)
        elif value.shape == (3,):
            value = value.astype(np.float32)
        else:
            raise ValueError(f"Vector3 requires shape (3,), got {value.shape}")
        return value.view(cls)


class Vector4(np.ndarray):
    """4D vector type for quaternions, colors, etc.

    Usage:
        v = Vector4([1, 2, 3, 4])
    """

    def __new__(cls, value=None):
        if value is None:
            value = [0, 0, 0, 0]
        value = np.asarray(value, dtype=np.float32)
        if value.shape == ():
            value = np.array([value.item(), 0, 0, 0], dtype=np.float32)
        elif value.shape == (4,):
            value = value.astype(np.float32)
        else:
            raise ValueError(f"Vector4 requires shape (4,), got {value.shape}")
        return value.view(cls)


# =============================================================================
# Float Type Marker
# =============================================================================


class Float(float):
    """Scalar float type marker for component annotations.

    Usage:
        @component
        class Custom:
            value: Float  # Scalar value
    """

    pass


# =============================================================================
# Color Type
# =============================================================================


class Color:
    """RGBA color wrapper with sRGB <-> linear conversion.

    Usage:
        c = Color("#ff0000")        # From hex
        c = Color(r=1, g=0, b=0)    # From RGB (0-1)
        c = Color(linear_r=0.2)     # From linear RGB

        linear = c.to_linear()      # Convert to linear
        srgb = c.to_srgb()          # Convert to sRGB
    """

    def __init__(
        self,
        hex_str=None,
        r=None,
        g=None,
        b=None,
        a=1.0,
        linear_r=None,
        linear_g=None,
        linear_b=None,
        linear_a=1.0,
    ):
        # Handle different input formats
        if hex_str is not None:
            # Parse hex string like "#ff0000" or "ff0000"
            self._parse_hex(hex_str)
        elif linear_r is not None:
            # Input is linear RGB
            self._linear = True
            self._r = linear_r
            self._g = linear_g if linear_g is not None else linear_r
            self._b = linear_b if linear_b is not None else linear_r
            self._a = linear_a if linear_a is not None else 1.0
        elif r is not None:
            # Input is sRGB (default)
            self._linear = False
            self._r = r
            self._g = g if g is not None else r
            self._b = b if b is not None else r
            self._a = a if a is not None else 1.0
        else:
            raise ValueError("Color requires hex_str, r/g/b, or linear_r")

    def _parse_hex(self, hex_str: str) -> None:
        """Parse hex color string."""
        self._linear = False
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = 1.0
        elif len(hex_str) == 8:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = int(hex_str[6:8], 16) / 255.0
        else:
            raise ValueError(f"Invalid hex color: {hex_str}")
        self._r = r
        self._g = g
        self._b = b
        self._a = a

    @property
    def r(self):
        return self._r

    @property
    def g(self):
        return self._g

    @property
    def b(self):
        return self._b

    @property
    def a(self):
        return self._a

    def to_linear(self):
        """Convert sRGB to linear RGB."""
        if self._linear:
            return Color(linear_r=self._r, linear_g=self._g, linear_b=self._b, linear_a=self._a)

        # Gamma correction: sRGB -> linear
        def gamma_correct(c):
            if c <= 0.04045:
                return c / 12.92
            else:
                return ((c + 0.055) / 1.055) ** 2.4

        return Color(
            linear_r=gamma_correct(self._r),
            linear_g=gamma_correct(self._g),
            linear_b=gamma_correct(self._b),
            linear_a=self._a,
        )

    def to_srgb(self):
        """Convert linear to sRGB."""
        if not self._linear:
            return Color(r=self._r, g=self._g, b=self._b, a=self._a)

        # Inverse gamma: linear -> sRGB
        def inverse_gamma(c):
            if c <= 0.0031308:
                return c * 12.92
            else:
                return 1.055 * (c ** (1 / 2.4)) - 0.055

        return Color(
            r=inverse_gamma(self._r), g=inverse_gamma(self._g), b=inverse_gamma(self._b), a=self._a
        )


# =============================================================================
# Component Registry (for type hints)
# =============================================================================

COMPONENT_REGISTRY = {}


def register_component(name: str, dtype: np.dtype, shape: tuple):
    """Register a component type for ECS storage."""
    COMPONENT_REGISTRY[name] = {"dtype": dtype, "shape": shape}


__all__ = [
    "Vector3",
    "Vector4",
    "Float",
    "Color",
    "COMPONENT_REGISTRY",
    "register_component",
]
