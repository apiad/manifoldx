"""N-Body gravitational simulation demo.

Pure-numpy vectorized all-pairs gravity. No Python loops in the hot path.
See examples/gas.py for a collision-based simulation.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import sphere, PhongMaterial

NUM_BODIES = 500
G = 20.0  # gravitational constant
SOFTENING = 0.05  # prevents singularities at close range
MAX_SPEED = 20.0  # velocity clamp
SPHERE_RADIUS = 0.5  # base mesh radius
SIZE = 5 * NUM_BODIES ** (1 / 3)

engine = mx.Engine("N-Body Simulation")
engine.camera.fit(SIZE)

# Random initial positions spread uniformly
positions = np.random.uniform(-SIZE, SIZE, size=(NUM_BODIES, 3)).astype(np.float32)

# Random masses → visual scale (cube-root so volume ∝ mass)
masses = np.random.uniform(0.5, 3.0, NUM_BODIES).astype(np.float32)
visual_scales = masses ** (1 / 3)
scales = visual_scales.reshape(-1, 1)  # (N, 1) → broadcasts to (N, 3)

# Spawn all bodies at once (instanced rendering)
engine.spawn(
    Mesh(sphere(SPHERE_RADIUS, 12)),
    Material(PhongMaterial("#ffaa44")),
    Transform(pos=positions, scale=scales),
    n=NUM_BODIES,
)

# Velocities (not part of ECS — pure numpy)
velocities = np.zeros((NUM_BODIES, 3), dtype=np.float32)


@engine.system
def nbody_gravity(query: mx.Query[Transform], dt: float):
    global velocities

    pos = query[Transform].pos.data  # (N, 3) copy

    # ── Vectorised all-pairs gravity ────────────────────────────
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    dist_safe = np.maximum(dist, SOFTENING)

    # Force magnitudes: G * m_i * m_j / r²
    mass_prod = masses[np.newaxis, :] * masses[:, np.newaxis]
    force_mag = G * mass_prod / (dist_safe**2)

    # Direction unit vectors (safe division — zero on diagonal)
    inv_dist = np.zeros_like(dist)
    nz = dist > 1e-6
    inv_dist[nz] = 1.0 / dist[nz]
    direction = diff * inv_dist[:, :, np.newaxis]

    # Net force per body: sum over all others
    forces = force_mag[:, :, np.newaxis] * direction
    net_force = forces.sum(axis=1)

    # Integrate velocity (F = ma → a = F/m)
    velocities += (net_force / masses[:, np.newaxis]) * dt

    # Write back
    query[Transform].pos += velocities * dt


if __name__ == "__main__":
    engine.cli()
