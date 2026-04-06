"""N-Body gravitational simulation demo.

100 bodies with pure-numpy vectorized gravity and elastic collisions.
No Python loops in the hot path.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import sphere, PhongMaterial

NUM_BODIES = 250
G = 10.0  # gravitational constant
RESTITUTION = 0.999  # collision bounciness (0=inelastic, 1=elastic)
SOFTENING = 0.3  # gravity softening to prevent singularities
DRAG = 0.995  # velocity damping per frame (keeps system bounded)
MAX_SPEED = 20.0  # velocity clamp
SPHERE_RADIUS = 0.5  # base mesh radius
SIZE = 0.5 * NUM_BODIES ** 1/3

engine = mx.Engine("N-Body Simulation")
engine.camera.fit(SIZE / 3)

# Random initial positions (normal distribution, ~5 unit spread)
positions = np.random.uniform(-SIZE, SIZE, size=(NUM_BODIES, 3)).astype(np.float32)

# Random masses → visual scale (cube-root so volume ∝ mass)
masses = np.random.uniform(0.5, 3.0, NUM_BODIES).astype(np.float32)
visual_scales = masses ** (1 / 3)
scales = visual_scales.reshape(-1, 1)  # (N, 1) → broadcasts to (N, 3)

# Collision radii = base mesh radius × visual scale
radii = SPHERE_RADIUS * visual_scales  # (N,)

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
def nbody_physics(query: mx.Query[Transform], dt: float):
    global velocities

    pos = query[Transform].pos.data  # (N, 3) copy

    # ── Vectorised all-pairs gravity ────────────────────────────
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    dist_safe = np.maximum(dist, SOFTENING)

    # Force magnitudes: G * m_i * m_j / r^2
    mass_prod = masses[np.newaxis, :] * masses[:, np.newaxis]  # (N, N)
    force_mag = G * mass_prod / (dist_safe**2)  # (N, N)

    # Direction unit vectors (safe division)
    inv_dist = np.zeros_like(dist)
    nz = dist > 1e-6
    inv_dist[nz] = 1.0 / dist[nz]
    direction = diff * inv_dist[:, :, np.newaxis]  # (N, N, 3)

    # Net force per body
    forces = force_mag[:, :, np.newaxis] * direction  # (N, N, 3)
    net_force = forces.sum(axis=1)  # (N, 3)

    # Integrate velocity (F = ma → a = F/m)
    velocities += (net_force / masses[:, np.newaxis]) * dt

    # ── Vectorised elastic collisions ───────────────────────────
    radii_sum = radii[np.newaxis, :] + radii[:, np.newaxis]  # (N, N)

    # Upper-triangle overlapping pairs (no double-counting)
    overlap = dist < radii_sum
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
            m_i = masses[idx_i]
            m_j = masses[idx_j]

            # Impulse (elastic collision with restitution)
            j_mag = -(1 + RESTITUTION) * v_n / (1.0 / m_i + 1.0 / m_j)
            impulse = j_mag[:, np.newaxis] * n_h

            # Safe accumulation (handles duplicate indices)
            np.add.at(velocities, idx_i, -impulse / m_i[:, np.newaxis])
            np.add.at(velocities, idx_j, impulse / m_j[:, np.newaxis])

    # ── Velocity damping & clamping ─────────────────────────────
    velocities *= DRAG
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    too_fast = speeds > MAX_SPEED
    velocities = np.where(too_fast, velocities * MAX_SPEED / speeds, velocities)

    # ── Write back ──────────────────────────────────────────────
    query[Transform].pos += velocities * dt


print(f"N-Body: {NUM_BODIES} bodies with gravity + collisions")
engine.run()
