# Lazy import to avoid wgpu dependency for types-only imports
def __getattr__(name):
    if name == 'Engine':
        from manifoldx.engine import Engine
        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def hello() -> str:
    return "Hello from manifoldx!"


# Export core modules and types
from manifoldx.types import Vector3, Vector4, Float, Color
from manifoldx.ecs import EntityStore, ComponentView, component
from manifoldx.systems import Query
from manifoldx.components import Transform, Mesh, Material, Colors
from manifoldx.resources import cube, sphere, plane, basic, phong, standard


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
]
