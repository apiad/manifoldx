"""Ideal gas simulation demo.

Particles bounce inside an invisible box with elastic collisions.
No gravity — pure kinetic theory. All physics is vectorized numpy.
See examples/nbody.py for a gravity-based simulation.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import sphere, PhongMaterial

NUM_PARTICLES = 500
BOX_HALF = 10.0  # half-size of the invisible bounding box
INITIAL_SPEED = 5.0  # Maxwell-Boltzmann-ish initial speed
PARTICLE_RADIUS = 0.2  # collision & visual radius

engine = mx.Engine("Ideal Gas")
engine.camera.fit(BOX_HALF)

# Random positions inside the box, leaving room for the particle radius.
positions = mx.random.positions_in_box(
    NUM_PARTICLES, half_size=BOX_HALF - PARTICLE_RADIUS, rng=7
)

# Maxwell-Boltzmann-ish: random direction × |N(speed, 0.3·speed)|.
directions = mx.random.velocities_on_sphere(NUM_PARTICLES, speed=1.0, rng=8)
speeds = np.abs(
    mx.random.scalars_gaussian(NUM_PARTICLES, sigma=INITIAL_SPEED * 0.3,
                                mean=INITIAL_SPEED, rng=9)
)
velocities = directions * speeds[:, None]

# Spawn all particles at once (instanced rendering — single draw call)
engine.spawn(
    Mesh(sphere(PARTICLE_RADIUS, 8)),
    Material(PhongMaterial("#44aaff")),
    Transform(pos=positions),
    n=NUM_PARTICLES,
)


@engine.system
def gas_physics(query: mx.Query[Transform], dt: float):
    global velocities
    pos = query[Transform].pos.data
    mx.physics.box_boundary(
        pos, velocities, half_size=BOX_HALF - PARTICLE_RADIUS, dt=dt
    )
    mx.physics.elastic_collisions(pos, velocities, radius=PARTICLE_RADIUS)
    query[Transform].pos += velocities * dt


if __name__ == "__main__":
    engine.cli()
