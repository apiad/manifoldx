import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material

engine = mx.Engine("Cubes")
engine.camera.zoom(0.1)

# These are all static things that are created
# and stored in memory once
cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.RED)


# Custom component, gets registered in engine to keep track
# Only used for reflection on the values
@engine.component
class Cube:
    velocity: mx.Vector3
    angular: mx.Vector3
    life: mx.Float  # alternatively can be just `float`


# This runs every frame
# You receive a view of all entities that have
# a Cube and a Transform component,
# and are alive
@engine.system
def cube_life(query: mx.Query[Cube, Transform], dt: float):
    # Here `query` is an EntitySet which contains a view to
    # all alive entities with the corresponding components

    # Indexing per component type gives us access to the view that
    # contains per-component data

    query[Cube].life -= dt  # Single vectorial operation
    # The previous emits a command that is executed at the end of the frame
    # Scalar operations are broadcast on system execution so this
    # is equivalent to the following but much faster
    # ...
    # query[Data].life -= np.asarray([dt for _ in len(query)])

    # This is a single vectorial operation, where velocity * dt is computed
    # during the system execution, but the actual addition is enqueued
    # as an Update command that will write to the appropriate buffer
    query[Transform].position += query[Cube].velocity * dt
    query[Transform].rotation += Transform.rotation(euler=query[Cube].angular * dt)

    # Scale depending on remaining life
    query[Transform].scale = query[Cube].life / 20.0

    # This issues a single destroy command to all items that pass the filter.
    # Destroyed entities are actually just marked as `alive = False` in some
    # buffer and thus never drawn, but not actually destroyed
    engine.destroy(query[Cube].life <= 0)

    # Now we create lots of cubes
    n_new = int(1000 * dt)

    # This will in principle reuse the buffers for dead entities, but will
    # expand the buffer if necessary
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(0, 0, 0), scale=(1, 1, 1)),
        Cube(
            velocity=np.random.uniform(-5, 5, (n_new, 3)),
            angular=np.random.uniform(-2, 2, (n_new, 3)),
            life=np.random.rand(n_new) * 20,
        ),
        n=n_new,
    )

    # Update camera
    engine.camera.orbit(45 * dt, 0)



engine.run()
