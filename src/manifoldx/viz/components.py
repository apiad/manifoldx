"""Sci-viz ECS components: PointCloud (marker), ScalarValue, Radius."""

import numpy as np


class ScalarValue:
    """Per-entity scalar attribute, mapped through ColormapMaterial's LUT.

    Storage layout: 1 float per entity (column 0).

    Usage:
        ScalarValue()                      # default 0.0 for all
        ScalarValue(value=1.5)             # broadcast scalar
        ScalarValue(value=array_shape_N)   # explicit per-entity
    """

    def __init__(self, value=None):
        self._value = value

    def get_data(self, n: int, registry=None) -> np.ndarray:
        data = np.zeros((n, 1), dtype=np.float32)
        if self._value is None:
            return data
        v = np.asarray(self._value, dtype=np.float32)
        if v.ndim == 0:
            data[:, 0] = float(v)
        elif v.ndim == 1 and v.shape[0] == n:
            data[:, 0] = v
        else:
            raise ValueError(f"ScalarValue: value shape {v.shape} incompatible with n={n}")
        return data


class Radius:
    """Per-entity world-space radius for sprite scaling.

    Storage layout: 1 float per entity (column 0).
    Default: 1.0.

    Usage:
        Radius()                       # all 1.0
        Radius(radius=0.05)            # broadcast scalar
        Radius(radius=array_shape_N)   # explicit per-entity
    """

    def __init__(self, radius=None):
        self._radius = radius

    def get_data(self, n: int, registry=None) -> np.ndarray:
        data = np.ones((n, 1), dtype=np.float32)
        if self._radius is None:
            return data
        v = np.asarray(self._radius, dtype=np.float32)
        if v.ndim == 0:
            data[:, 0] = float(v)
        elif v.ndim == 1 and v.shape[0] == n:
            data[:, 0] = v
        else:
            raise ValueError(f"Radius: radius shape {v.shape} incompatible with n={n}")
        return data


class PointCloud:
    """Marker component — tags entities for the sprite render path.

    Carries no per-entity data. The renderer detects this component
    and substitutes the SPRITE_QUAD geometry, ignoring any Mesh component.
    """

    def __init__(self):
        pass

    def get_data(self, n: int, registry=None) -> np.ndarray:
        return np.zeros((n, 0), dtype=np.float32)
