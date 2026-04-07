import math
import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query

engine = mx.Engine("Cubes")

# These are all static things that are created
# and stored in memory once
cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.RED)


# This method gets executed at startup
@engine.startup
def create_cubes():
    # Instantiate a single cube
    engine.spawn(
        # These components just store indices to resources
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(0, 0, 0)),
        n=1,
    )


@engine.system
def rotate(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=0, y=dt * math.pi, z=0)
    query[Transform].pos = (0, math.sin(engine.elapsed), 0)


if __name__ == "__main__":
    engine.cli()
