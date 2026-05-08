"""Ideal gas demo with GPU compute physics.

A side-by-side counterpart to examples/gas.py: same scene of particles
bouncing inside an invisible cube with pairwise elastic collisions —
all on GPU.

Compared to examples/gas.py:
- Box-bounce + elastic collisions + integration all run as a single
  WGSL compute pass.
- The collision step is reformulated to be race-free: instead of
  pair-iterating and writing BOTH partners' velocities (the natural
  CPU formulation, which would race on GPU), each thread `i` surveys
  every other particle `j` and accumulates *its own* impulse from
  approaching overlaps. Thread `j` independently computes the
  equal-and-opposite impulse for itself. No thread ever writes another
  thread's velocity, so the kernel is safe at any workgroup size.

Compared to examples/nbody_compute.py and examples/point_cloud_compute.py:
- First example to use vec3 swizzle (`.x` / `.y` / `.z`) inside a
  kernel — the per-axis bounce condition can't be expressed without
  reading individual vector components.
- First example to use vec3 / f32 broadcast (collision normal).
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform
from manifoldx.compute.shader import abs, dot, sqrt, u32, vec3
from manifoldx.resources import PhongMaterial, sphere


# ── Custom velocity component ────────────────────────────────────────────────
class Velocity(Component):
    """Per-entity velocity vector (3 floats)."""
    vector: vec3


# ── Simulation parameters ────────────────────────────────────────────────────
NUM_PARTICLES = 500
BOX_HALF = 10.0
INITIAL_SPEED = 5.0
PARTICLE_RADIUS = 0.2
RESTITUTION = 1.0  # 1.0 = perfectly elastic; 0.0 = fully inelastic


# ── GPU compute kernel — box-boundary + elastic collisions + integration ────
class GasKernel(Compute):
    """Per-particle physics, one thread per entity:

    1. Reflect off the walls of a [-half, +half]^3 cube.
    2. Survey all other particles; for each approaching overlap,
       accumulate this particle's share of the elastic impulse.
    3. Advance position by velocity*dt.

    Race-free because each thread writes only its own velocity. The
    collision math is symmetric — equal-and-opposite impulses fall out
    automatically when thread `j` runs the same routine.
    """

    transforms: ReadsWrites[Transform]
    velocities: ReadsWrites[Velocity]

    half_size:   Uniform[float] = BOX_HALF - PARTICLE_RADIUS
    collide_d2:  Uniform[float] = (2.0 * PARTICLE_RADIUS) ** 2
    restitution: Uniform[float] = RESTITUTION
    dt:          Uniform[float] = "frame_dt"      # type: ignore[assignment]  # auto-bound
    n:           Uniform[float] = "entity_count"  # type: ignore[assignment]  # auto-bound

    workgroup_size = 64
    dispatch = "entity_count"

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos_i: vec3 = self.transforms[i].pos
        vel_i: vec3 = self.velocities[i].vector

        # ── Box boundary: flip velocity component on any axis that
        # would step past a wall (matches mx.physics.box_boundary
        # `mode="reflect"`).
        next_pos: vec3 = pos_i + vel_i * self.dt
        if next_pos.x < -self.half_size:
            vel_i = vec3(abs(vel_i.x), vel_i.y, vel_i.z)
        elif next_pos.x > self.half_size:
            vel_i = vec3(-abs(vel_i.x), vel_i.y, vel_i.z)
        if next_pos.y < -self.half_size:
            vel_i = vec3(vel_i.x, abs(vel_i.y), vel_i.z)
        elif next_pos.y > self.half_size:
            vel_i = vec3(vel_i.x, -abs(vel_i.y), vel_i.z)
        if next_pos.z < -self.half_size:
            vel_i = vec3(vel_i.x, vel_i.y, abs(vel_i.z))
        elif next_pos.z > self.half_size:
            vel_i = vec3(vel_i.x, vel_i.y, -abs(vel_i.z))

        # ── Elastic collisions: scan all neighbours, accumulate the
        # impulse this particle receives from approaching overlaps.
        # Equal masses → impulse magnitude is 0.5*(1+e)*v_rel·n along n.
        # `vel_j` here is the start-of-frame velocity from the storage
        # buffer (no other thread has written yet); good enough at frame
        # granularity.
        dv: vec3 = vec3(0.0, 0.0, 0.0)
        for j in range(u32(self.n)):
            if i == j:
                continue
            pos_j: vec3 = self.transforms[j].pos
            diff: vec3 = pos_j - pos_i
            d2: float = dot(diff, diff)
            if d2 > self.collide_d2:
                continue
            if d2 < 1e-12:
                continue
            d: float = sqrt(d2)
            n_hat: vec3 = diff / d
            vel_j: vec3 = self.velocities[j].vector
            v_rel: vec3 = vel_j - vel_i
            v_dot_n: float = dot(v_rel, n_hat)
            # n_hat points from i to j. v_rel = vel_j - vel_i. Approaching
            # pairs have v_dot_n < 0 (j moves opposite to n_hat relative
            # to i). Separating pairs already moved past each other —
            # skip them so resolved overlaps don't get re-kicked.
            if v_dot_n >= 0.0:
                continue
            dv += 0.5 * (1.0 + self.restitution) * v_dot_n * n_hat

        vel_i += dv
        self.velocities[i].vector = vel_i
        self.transforms[i].pos = pos_i + vel_i * self.dt


# ── Engine setup ─────────────────────────────────────────────────────────────
engine = mx.Engine("Ideal Gas (compute)")
engine.camera.fit(BOX_HALF)

positions = mx.random.positions_in_box(
    NUM_PARTICLES, half_size=BOX_HALF - PARTICLE_RADIUS, rng=7
)
directions = mx.random.velocities_on_sphere(NUM_PARTICLES, speed=1.0, rng=8)
speeds = np.abs(
    mx.random.scalars_gaussian(
        NUM_PARTICLES, sigma=INITIAL_SPEED * 0.3, mean=INITIAL_SPEED, rng=9
    )
)
initial_velocities = (directions * speeds[:, None]).astype(np.float32)

engine.spawn(
    Mesh(sphere(PARTICLE_RADIUS, 8)),
    Material(PhongMaterial("#44aaff")),
    Transform(pos=positions),
    Velocity(vector=initial_velocities),
    n=NUM_PARTICLES,
)

engine.compute(GasKernel)


if __name__ == "__main__":
    engine.cli()
