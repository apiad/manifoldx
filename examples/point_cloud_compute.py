"""Point cloud demo with GPU compute physics.

A side-by-side counterpart to examples/point_cloud_demo.py: same
protoplanetary-disk scene (~10000 dust particles orbiting a central
star, colormapped by orbital speed) but the per-particle Keplerian
gravity + velocity/position integration runs as a WGSL compute shader
instead of pure-numpy on CPU.

Compared to examples/nbody_compute.py:
- Kernel has no inner loop (per-particle physics, no pairwise sum).
- Kernel writes a SCALAR component (ScalarValue.value = current speed),
  so the GPU-driven colormap pipeline picks up speeds without any CPU
  round-trip — sprites recolor each frame purely from compute output.

The HUD-text bookkeeping from point_cloud_demo.py is intentionally
omitted: it's CPU coordination, not physics, and dropping it keeps the
port to its minimal physics core.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform, Writes
from manifoldx.compute.shader import dot, sqrt, u32, vec3
from manifoldx.resources import PointLight, StandardMaterial, sphere
from manifoldx.viz import (
    ColormapMaterial,
    PointCloud,
    Radius,
    ScalarValue,
)


# ── Custom velocity component ────────────────────────────────────────────────
# point_cloud_demo.py keeps velocities in a numpy array outside the ECS;
# moving them to a Component lets the compute kernel bind to them as a
# storage buffer.
class Velocity(Component):
    """Per-entity velocity vector (3 floats)."""
    vector: vec3


# ── Simulation parameters ────────────────────────────────────────────────────
NUM_PARTICLES = 10000
GM = 60.0
SOFTENING = 0.4
DISK_INNER = 2.0
DISK_OUTER = 8.0
DISK_THICK = 0.18
ECCENTRICITY = 0.06
STAR_RADIUS = 1.0
EXTENT = DISK_OUTER * 1.2


# ── GPU compute kernel — plain typed Python ──────────────────────────────────
class OrbitKernel(Compute):
    """Soft Keplerian gravity around the origin + speed-driven colormap.

    One thread per entity. The central star sits at the origin: with
    softening > 0 its self-gravity vanishes (`accel = -GM * 0 * inv_r3`),
    so it stays put without an explicit guard.
    """

    transforms: ReadsWrites[Transform]
    velocities: ReadsWrites[Velocity]
    speeds:     Writes[ScalarValue]

    GM:        Uniform[float] = GM
    softening: Uniform[float] = SOFTENING
    dt:        Uniform[float] = "frame_dt"      # type: ignore[assignment]  # auto-bound
    n:         Uniform[float] = "entity_count"  # type: ignore[assignment]  # auto-bound

    workgroup_size = 64
    dispatch = "entity_count"

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos: vec3 = self.transforms[i].pos
        r2: float = dot(pos, pos) + self.softening * self.softening
        inv_r3: float = 1.0 / (r2 * sqrt(r2))
        accel: vec3 = -self.GM * pos * inv_r3
        self.velocities[i].vector += accel * self.dt
        self.transforms[i].pos    += self.velocities[i].vector * self.dt
        v_new: vec3 = self.velocities[i].vector
        self.speeds[i].value = sqrt(dot(v_new, v_new))


# ── Engine setup ─────────────────────────────────────────────────────────────
engine = mx.Engine("Protoplanetary disk (compute)")
engine.camera.fit(EXTENT)

# Central star — entity index 0. Origin gravity vanishes through softening.
engine.spawn(
    Mesh(sphere(STAR_RADIUS, 32)),
    Material(StandardMaterial("#ffd45a", roughness=0.85, metallic=0.0)),
    Transform(pos=(0.0, 0.0, 0.0)),
    Velocity(vector=np.zeros(3, dtype=np.float32)),
    n=1,
)
engine.set_lights(
    [
        PointLight(color="#ffe9a8", intensity=80.0, position=(0.0, 0.0, 0.0)),
        PointLight(color="#ffffff", intensity=8.0, position=(6.0, 8.0, 6.0)),
    ]
)

# Dust particles on Keplerian circular orbits + small eccentricity.
positions = mx.random.positions_in_disk(
    NUM_PARTICLES, inner=DISK_INNER, outer=DISK_OUTER,
    thickness=DISK_THICK, axis="y", rng=7,
)
particle_velocities = mx.random.velocities_orbit(positions, GM=GM, axis="y")
v_circ_mag = np.linalg.norm(particle_velocities, axis=1, keepdims=True)
particle_velocities += (
    mx.random.velocities_gaussian(NUM_PARTICLES, sigma=ECCENTRICITY, rng=8)
    * v_circ_mag
)
radii = mx.random.scalars_uniform(NUM_PARTICLES, low=0.04, high=0.10, rng=9)
initial_speeds = np.linalg.norm(particle_velocities, axis=1).astype(np.float32)

engine.spawn(
    PointCloud(),
    Material(ColormapMaterial(cmap="inferno", vmin=2.5, vmax=6.5)),
    Transform(pos=positions),
    Velocity(vector=particle_velocities.astype(np.float32)),
    ScalarValue(value=initial_speeds),
    Radius(radius=radii),
    n=NUM_PARTICLES,
)

engine.compute(OrbitKernel)


if __name__ == "__main__":
    engine.cli()
