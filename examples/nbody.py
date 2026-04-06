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

# Radii from scales (scale = mass^(1/3) * 10, so radius = scale / 10)
radii = (scales.squeeze() / 10).astype(np.float32)

# Velocities for physics
velocities = np.zeros((NUM_BODIES, 3), dtype=np.float32)


@engine.system
def nbody_physics(query: mx.Query[Transform], dt: float):
    global velocities

    pos = query[Transform].pos.data

    # Vectorized all-pairs gravity
    diff = pos[None, :] - pos[:, None]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    dist = np.maximum(dist, 0.5)

    mass_prod = masses[None, :] * masses[:, None]
    force_mag = G * mass_prod / (dist**3)
    force_mag = force_mag[:, :, np.newaxis]

    forces = force_mag * diff
    net_force = np.sum(forces, axis=1)

    velocities += (net_force / masses[:, None]) * dt
    query[Transform].pos += velocities * dt

    # Simple collision detection and response
    radii_sum = radii[None, :] + radii[:, None]  # (N, N)
    collision_mask = dist < radii_sum

    # Get collision normals
    normal = diff / dist[:, :, np.newaxis]
    normal = np.nan_to_num(normal, 0)

    # Relative velocity
    rel_vel = velocities[None, :] - velocities[:, :]  # (N, N, 3)

    # Velocity along collision normal
    vel_along_normal = np.sum(rel_vel * normal, axis=2)  # (N, N)

    # Only resolve if velocities are approaching
    approaching = vel_along_normal < 0

    # Combine masks
    resolve_mask = (
        collision_mask & approaching & (np.triu(np.ones((NUM_BODIES, NUM_BODIES), dtype=bool), k=1))
    )

    # For each collision pair, reflect velocities
    damping = 0.8
    for i, j in zip(*np.where(resolve_mask)):
        v1 = velocities[i]
        v2 = velocities[j]
        n = normal[i, j]

        # Reflect velocities
        v1_new = v1 - 2 * np.dot(v1, n) * n
        v2_new = v2 - 2 * np.dot(v2, n) * n

        velocities[i] = v1_new * damping
        velocities[j] = v2_new * damping

        # Separate bodies to avoid overlap
        overlap = (radii[i] + radii[j]) - dist[i, j]
        if overlap > 0:
            sep = n * (overlap / 2 + 0.01)
            pos[i] += sep
            pos[j] -= sep

    query[Transform].pos = pos


light = PointLight(color="#ffffff", intensity=20.0, position=(20, 20, 20))
engine.set_lights([light])

print(f"N-Body: {NUM_BODIES} bodies")
engine.run()
