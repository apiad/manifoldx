import numpy as np

# Export core modules and types
from manifoldx.engine import Engine
from manifoldx.types import Vector3, Vector4, Float, Color
from manifoldx.ecs import EntityStore, ComponentView
from manifoldx.systems import Query
from manifoldx.components import Transform, Mesh, Material, Colors
from manifoldx.resources import cube, sphere, plane, basic, phong, standard


# =============================================================================
# Module Proxies for mx.types, mx.components, mx.geometry, etc.
# =============================================================================

class _TypesProxy:
    """Proxy for mx.types module."""
    Vector3 = Vector3
    Vector4 = Vector4
    Float = Float
    Color = Color


class _ComponentsProxy:
    """Proxy for mx.components module."""
    Transform = Transform
    Mesh = Mesh
    Material = Material
    Colors = Colors


class _GeometryProxy:
    """Proxy for mx.geometry module."""
    def cube(self, w=1, h=1, d=1): return cube(w, h, d)
    def sphere(self, r=1, segments=32): return sphere(r, segments)
    def plane(self, w=1, h=1): return plane(w, h)


class _MaterialProxy:
    """Proxy for mx.material module."""
    def basic(self, color): return basic(color)
    def phong(self, color, shininess=32): return phong(color, shininess)
    def standard(self, color, roughness=0.5, metallic=0): return standard(color, roughness, metallic)


class _ColorsProxy:
    """Proxy for mx.colors module."""
    RED = "#ff0000"
    GREEN = "#00ff00"
    BLUE = "#0000ff"
    WHITE = "#ffffff"
    BLACK = "#000000"
    YELLOW = "#ffff00"
    CYAN = "#00ffff"
    MAGENTA = "#ff00ff"


# Create module instances
types = _TypesProxy()
components = _ComponentsProxy()
geometry = _GeometryProxy()
material = _MaterialProxy()
colors = _ColorsProxy()


__all__ = [
    'Engine',
    'Vector3',
    'Vector4',
    'Float',
    'Color',
    'EntityStore',
    'ComponentView',
    'Query',
    'Transform',
    'Mesh',
    'Material',
    'Colors',
    'cube',
    'sphere',
    'plane',
    'basic',
    'phong',
    'standard',
    'component',
    'types',
    'components',
    'geometry',
    'material',
    'colors',
]