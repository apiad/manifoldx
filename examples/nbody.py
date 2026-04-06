"""N-Body gravitational simulation demo."""

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import sphere, PhongMaterial, PointLight

NUM_BODIES = 100
G = 50.0

engine = mx.Engine("N-Body Simulation")
engine.camera.fit(30)

# Random positions on sphere surface
theta = np.random.uniform(0, 2 * np.pi, NUM_BODIES)
phi = np.arccos(2 * np.random.uniform(0, 1, NUM_BODIES) - 1)
radius = 15.0

positions = np.column_stack(
    [
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi),
    ]
).astype(np.float32)

# Random masses and compute scales (cubic law)
masses = np.random.uniform(0.5, 3.0, NUM_BODIES).astype(np.float32)
scales = (masses ** (1 / 3) * 10.0).reshape(-1, 1)

# Spawn all at once
engine.spawn(
    Mesh(sphere(0.3, 12)),
    Material(PhongMaterial("#ffaa44")),
    Transform(pos=positions, scale=scales),
    n=NUM_BODIES,
)

# Velocities for physics
velocities = np.zeros((NUM_BODIES, 3), dtype=np.float32)


@engine.system
def nbody_physics(query: mx.Query[Transform], dt: float):
    global velocities

    pos = query[Transform].pos.data

    # Vectorized all-pairs gravity
    diff = pos[None, :] - pos[:, None]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, 0.5)

    mass_prod = masses[None, :] * masses[:, None]
    force_mag = G * mass_prod / (dist**3)
    force_mag = force_mag[:, :, np.newaxis]

    forces = force_mag * diff
    net_force = np.sum(forces, axis=1)

    velocities += (net_force / masses[:, None]) * dt
    query[Transform].pos += velocities * dt


light = PointLight(color="#ffffff", intensity=20.0, position=(20, 20, 20))
engine.set_lights([light])

print(f"N-Body: {NUM_BODIES} bodies")
engine.run()
