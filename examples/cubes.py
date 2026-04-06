import manifoldx as mx
import numpy as np

engine = mx.Engine("Cubes")

# These are all static things that are created
# and stored in memory once
cube_mesh = mx.geometry.cube(1,1,1)
cube_material = mx.material.phong(mx.colors.RED)


# Custom component, gets registered in engine to keep track
# Only used for reflection on the values
@mx.component
class Cube:
    velocity: mx.types.Vector3
    life: mx.types.Float # alternatively can be just `float`


# This method gets executed at startup
@engine.startup
def create_cubes():
    # Instantiate 1000 cubes in a single call
    # This creates a drawing group bound to Mesh/Material
    engine.spawn(
        # These components just store indices to resources
        mx.components.Mesh(cube_mesh),
        mx.components.Material(cube_material),
        # Components like Transform and Cube store data per instance
        # Their parameters are broadcast if scalar
        mx.components.Transform(pos=(0,0,0), scale=(1,1,1)),
        # Vectorial components must match the instancing size
        # First dimension is n-instance, second dimension matches vector size
        Cube(velocity=np.random.rand(1000, 3), life=np.random.rand(1000) * 20),
        # Total number of instances to create
        n=1000,
    )

# This runs every frame
# You receive a view of all entities that have
# a Cube and a Transform component,
# and are alive
@engine.system
def cube_life(query: mx.Query[Cube, mx.components.Transform], dt: float):
    # Here `query` is an EntitySet which contains a view to
    # all alive entities with the corresponding components

    # Indexing per component type gives us access to the view that
    # contains per-component data

    query[Cube].life -= dt # Single vectorial operation
    # This emits a command that is executed at the end of the frame
    # Scalar operations are broadcast on system execution so this
    # is equivalent to the following but much faster
    # ...
    # query[Data].life -= np.asarray([dt for _ in len(query)])

    query[mx.components.Transform].position += query[Cube].velocity * dt
    # This is a single vectorial operation, where velocity * dt is computed
    # during the system execution, but the actual addition is enqueued
    # as an Update command that will write to the appropriate buffer

    engine.destroy(query[Cube].life <= 0)
    # This issues a single destroy command to all items that pass the filter.
    # Destroyed entities are actually just marked as `alive = False` in some
    # buffer and thus never drawn, but not actually destroyed

    # Now we create extra cubes
    # Since ~100 cubes die every second (there were 1000 with average life = 10 seconds)
    # We need to replace the same amount
    n_new = 100 * dt

    engine.spawn(
        mx.components.Mesh(cube_mesh),
        mx.components.Material(cube_material),
        mx.components.Transform(pos=(0,0,0), scale=(1,1,1)),
        Cube(velocity=np.random.rand(n_new, 3), life=np.random.rand(n_new) * 20),
        n=n_new,
    )

    # This will in principle reuse the buffers for dead entities, but will
    # expand the buffer if necessary

engine.run()
