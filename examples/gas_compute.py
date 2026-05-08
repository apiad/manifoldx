"""Ideal gas demo with GPU compute physics (box-bounce subset).

A side-by-side counterpart to examples/gas.py: same scene of particles
bouncing inside an invisible cube, but the per-particle box-boundary
reflection + position integration runs as a WGSL compute shader.

Compared to examples/gas.py:
- Box-bounce physics is on GPU.
- The pairwise elastic collisions from gas.py are intentionally
  omitted: that step writes BOTH colliding particles' velocities, which
  invites a GPU-thread race when many threads see the same overlap.
  Resolving that needs either a two-pass approach with atomics or a
  serialized impulse pass — a separate piece of work.

Compared to examples/nbody_compute.py and examples/point_cloud_compute.py:
- First example to use vec3 swizzle (`.x` / `.y` / `.z`) inside a
  kernel — the per-axis bounce condition can't be expressed without
  reading individual vector components.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform
from manifoldx.compute.shader import abs, u32, vec3
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


# ── GPU compute kernel — box-boundary reflection + integration ───────────────
class GasKernel(Compute):
    """Reflect each particle off the walls of a [-half, +half]^3 cube,
    then advance position by velocity*dt. One thread per particle.
    """

    transforms: ReadsWrites[Transform]
    velocities: ReadsWrites[Velocity]

    half_size: Uniform[float] = BOX_HALF - PARTICLE_RADIUS
    dt:        Uniform[float] = "frame_dt"      # type: ignore[assignment]  # auto-bound
    n:         Uniform[float] = "entity_count"  # type: ignore[assignment]  # auto-bound

    workgroup_size = 64
    dispatch = "entity_count"

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos: vec3 = self.transforms[i].pos
        vel: vec3 = self.velocities[i].vector
        # Predict next position; flip the velocity component on any axis
        # that would step past a wall (matches mx.physics.box_boundary
        # `mode="reflect"` exactly).
        next_pos: vec3 = pos + vel * self.dt
        if next_pos.x < -self.half_size:
            vel = vec3(abs(vel.x), vel.y, vel.z)
        elif next_pos.x > self.half_size:
            vel = vec3(-abs(vel.x), vel.y, vel.z)
        if next_pos.y < -self.half_size:
            vel = vec3(vel.x, abs(vel.y), vel.z)
        elif next_pos.y > self.half_size:
            vel = vec3(vel.x, -abs(vel.y), vel.z)
        if next_pos.z < -self.half_size:
            vel = vec3(vel.x, vel.y, abs(vel.z))
        elif next_pos.z > self.half_size:
            vel = vec3(vel.x, vel.y, -abs(vel.z))
        self.velocities[i].vector = vel
        self.transforms[i].pos = pos + vel * self.dt


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
