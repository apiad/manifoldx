"""Boids flocking simulation demo.

300 boids with separation, alignment, cohesion, and a soft boundary.
Pure-numpy vectorized — no Python loops in the hot path.
Boids orbit a central attractor so the flock stays visible.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import sphere, PhongMaterial

# ── Simulation parameters ───────────────────────────────────────
NUM_BOIDS = 300
PERCEPTION_SQ = 4.0**2  # squared perception radius (avoids sqrt)
SEP_W = 1.5  # separation weight (strongest — avoid crowding)
ALI_W = 1.0  # alignment weight (match neighbor heading)
COH_W = 0.8  # cohesion weight (steer toward group center)
MAX_SPEED = 10.0  # speed cap
MIN_SPEED = 4.0  # prevents hovering
BOUND_RADIUS = 12.0  # soft boundary — strong push beyond this
PULL_STRENGTH = 0.02  # gentle always-on central attraction

# ── Engine setup ────────────────────────────────────────────────
engine = mx.Engine("Boids Flocking")
engine.camera.fit(BOUND_RADIUS * 1.2)

# ── Initial conditions ──────────────────────────────────────────
# Spawn in a cloud around the origin
positions = np.random.normal(0, 5, (NUM_BOIDS, 3)).astype(np.float32)

# Tangential initial velocity → swirling start instead of random chaos
axis = np.array([0.2, 1.0, 0.3], dtype=np.float32)
axis /= np.linalg.norm(axis)
tangent = np.cross(positions, axis)
norms = np.linalg.norm(tangent, axis=1, keepdims=True)
tangent = np.where(
    norms > 1e-6,
    tangent / norms,
    np.random.randn(NUM_BOIDS, 3).astype(np.float32),
)
velocities = (tangent * 6.0).astype(np.float32)

# ── Spawn all boids at once (instanced rendering) ──────────────
engine.spawn(
    Mesh(sphere(0.15, 6)),
    Material(PhongMaterial("#33ddaa")),
    Transform(pos=positions),
    n=NUM_BOIDS,
)


@engine.system
def boids_physics(query: mx.Query[Transform], dt: float):
    global velocities

    pos = query[Transform].pos.data  # (N, 3) copy

    # ── All-pairs squared distances (avoids sqrt) ───────────────
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, 3)
    dist_sq = (diff * diff).sum(axis=2)  # (N, N)

    # Neighbor mask (within perception, exclude self)
    neighbors = (dist_sq < PERCEPTION_SQ) & (dist_sq > 1e-6)  # (N, N)
    n_count = neighbors.sum(axis=1, keepdims=True).astype(np.float32)
    safe_count = np.maximum(n_count, 1.0)

    # ── Separation: steer away from close neighbors ─────────────
    # Weight by 1/dist² — closer neighbors repel much more strongly
    inv_dsq = np.zeros_like(dist_sq)
    inv_dsq[neighbors] = 1.0 / dist_sq[neighbors]
    sep = (-diff * (neighbors[:, :, np.newaxis] * inv_dsq[:, :, np.newaxis])).sum(axis=1)

    # ── Alignment: match average neighbor velocity ──────────────
    avg_vel = (velocities[np.newaxis, :, :] * neighbors[:, :, np.newaxis]).sum(axis=1)
    avg_vel /= safe_count
    ali = avg_vel - velocities

    # ── Cohesion: steer toward neighbor center of mass ──────────
    center = (pos[np.newaxis, :, :] * neighbors[:, :, np.newaxis]).sum(axis=1)
    center /= safe_count
    coh = center - pos

    # ── Soft boundary: strong inward push outside BOUND_RADIUS ──
    dist_center = np.linalg.norm(pos, axis=1, keepdims=True)
    overshoot = np.maximum(dist_center - BOUND_RADIUS, 0.0)
    inward_dir = -pos / np.maximum(dist_center, 1e-6)
    bound = inward_dir * overshoot * 3.0

    # ── Gentle central pull (keeps flock orbiting) ──────────────
    pull = -pos * PULL_STRENGTH

    # ── Combine forces and integrate ────────────────────────────
    accel = SEP_W * sep + ALI_W * ali + COH_W * coh + pull + bound
    velocities += accel * dt

    # Speed clamping (min + max keeps boids energetic)
    speed = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = np.where(speed > MAX_SPEED, velocities * MAX_SPEED / speed, velocities)
    velocities = np.where(
        (speed < MIN_SPEED) & (speed > 1e-6),
        velocities * MIN_SPEED / speed,
        velocities,
    )

    # Write back
    query[Transform].pos += velocities * dt


print(f"Boids: {NUM_BOIDS} agents with flocking + soft boundary")
engine.run()
