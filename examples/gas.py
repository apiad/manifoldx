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

# Random positions inside the box
positions = np.random.uniform(
    -BOX_HALF + PARTICLE_RADIUS,
    BOX_HALF - PARTICLE_RADIUS,
    size=(NUM_PARTICLES, 3),
).astype(np.float32)

# Random initial velocities (uniform direction, Maxwell-Boltzmann-ish speed)
directions = np.random.randn(NUM_PARTICLES, 3).astype(np.float32)
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
speeds = np.abs(np.random.normal(INITIAL_SPEED, INITIAL_SPEED * 0.3, NUM_PARTICLES))
velocities = (directions * speeds[:, np.newaxis]).astype(np.float32)

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

    pos = query[Transform].pos.data  # (N, 3) copy

    # ── Wall collisions (invisible box) ─────────────────────────
    # Predict next position
    next_pos = pos + velocities * dt
    lo = -BOX_HALF + PARTICLE_RADIUS
    hi = BOX_HALF - PARTICLE_RADIUS

    # Reflect velocity where particles cross walls (vectorized per-axis)
    below = next_pos < lo
    above = next_pos > hi
    velocities[below] = np.abs(velocities[below])
    velocities[above] = -np.abs(velocities[above])

    # ── Vectorised particle-particle collisions ─────────────────
    # Use current positions for overlap detection
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)

    # Find overlapping pairs (upper triangle only — no double-counting)
    collision_dist = 2 * PARTICLE_RADIUS
    overlap = dist < collision_dist
    np.fill_diagonal(overlap, False)
    i_idx, j_idx = np.where(np.triu(overlap))

    if i_idx.size > 0:
        # Collision normals (i → j)
        n_vec = diff[i_idx, j_idx]  # (K, 3)
        n_dist = dist[i_idx, j_idx, np.newaxis]  # (K, 1)
        n_hat = np.where(n_dist > 1e-6, n_vec / n_dist, 0.0)  # (K, 3)

        # Relative velocity
        v_rel = velocities[j_idx] - velocities[i_idx]  # (K, 3)
        v_dot_n = (v_rel * n_hat).sum(axis=1)  # (K,)

        # Only resolve approaching pairs
        approaching = v_dot_n < 0
        if approaching.any():
            idx_i = i_idx[approaching]
            idx_j = j_idx[approaching]
            v_n = v_dot_n[approaching]
            n_h = n_hat[approaching]

            # Equal-mass elastic collision: exchange velocity along normal
            # Impulse = -(1+e) * v_rel·n / (1/m + 1/m) = -(1+e)/2 * v_rel·n
            impulse = (-0.5 * v_n)[:, np.newaxis] * n_h

            # Safe accumulation (handles particle in multiple collisions)
            np.add.at(velocities, idx_i, -impulse)
            np.add.at(velocities, idx_j, impulse)

    # ── Write back ──────────────────────────────────────────────
    query[Transform].pos += velocities * dt


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1 and sys.argv[1] == "--render":
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60
        engine.render(str(Path(__file__).with_suffix(".mp4")), duration=duration)
    else:
        engine.run()