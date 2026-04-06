import math
import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query

engine = mx.Engine("Cubes")

# These are all static things that are created
# and stored in memory once
cube_mesh = mx.geometry.cube(1,1,1)
cube_material = mx.material.phong(mx.colors.RED)


# This method gets executed at startup
@engine.startup
def create_cubes():
    # Instantiate 1000 cubes in a single call
    # This creates a drawing group bound to Mesh/Material
    engine.spawn(
        # These components just store indices to resources
        Mesh(cube_mesh),
        Material(cube_material),
        # Components like Transform and Cube store data per instance
        # Their parameters are broadcast if scalar
        Transform(pos=(0,0,0), scale=(1,1,1)),
        # Vectorial components must match the instancing size
        # First dimension is n-instance, second dimension matches vector size
        n=1,
    )


@engine.system
def rotate(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=0, y=dt * math.pi, z =0)


engine.run()
