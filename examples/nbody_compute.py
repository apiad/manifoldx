"""N-Body simulation with GPU compute shader.

A side-by-side counterpart to examples/nbody.py: same physics, same
visuals, but the all-pairs O(N²) gravity loop runs in a WGSL compute
shader instead of pure-numpy on CPU.

Demonstrates the Phase-2 Python-as-shader DSL: GravityKernel.main is
plain typed Python; the engine traces it to WGSL on engine.compute(...)
via manifoldx.compute.transpile.

Phase 1 (raw WGSL via compile() override) still works — see the test
suite for an explicit hand-written counterpart.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
from manifoldx.compute.shader import vec3, dot, sqrt
from manifoldx.resources import PhongMaterial, sphere
from manifoldx.types import Float, Vector3


# ── Custom components ────────────────────────────────────────────────────────
class Velocity(Component):
    """Per-entity velocity vector (3 floats)."""
    vector: Vector3


class Mass(Component):
    """Per-entity gravitational mass (1 float)."""
    value: Float = 1.0


# ── Simulation parameters ────────────────────────────────────────────────────
NUM_BODIES = 500
G = 20.0
SOFTENING = 0.05
SPHERE_RADIUS = 0.5
SIZE = 5 * NUM_BODIES ** (1 / 3)


# ── GPU compute kernel — plain typed Python ──────────────────────────────────
class GravityKernel(Compute):
    """N-body gravity + velocity-and-position integration on the GPU.

    Each thread handles one body, sums gravitational pull from every
    other body, then integrates velocity and position. The kernel body
    is plain typed Python — manifoldx.compute.transpile traces it to
    WGSL when engine.compute(GravityKernel) is called.
    """

    transforms: ReadsWrites[Transform]
    masses:     Reads[Mass]
    velocities: ReadsWrites[Velocity]

    G:         Uniform[float] = G
    softening: Uniform[float] = SOFTENING
    dt:        Uniform[float] = "frame_dt"      # auto-bound, re-uploaded each frame
    n:         Uniform[float] = "entity_count"  # auto-bound, re-uploaded each frame

    workgroup_size = 64
    dispatch = "entity_count"

    def pair_accel(self, pos_i: vec3, pos_j: vec3, m_j: float) -> vec3:
        diff:   vec3  = pos_j - pos_i
        r2:     float = dot(diff, diff) + self.softening * self.softening
        inv_r3: float = 1.0 / (r2 * sqrt(r2))
        return self.G * m_j * diff * inv_r3

    def main(self, i: int):
        if i >= u32(self.n):
            return
        pos_i: vec3 = self.transforms[i].pos
        accel: vec3 = vec3(0.0, 0.0, 0.0)
        for j in range(u32(self.n)):
            if i == j:
                continue
            accel += self.pair_accel(pos_i, self.transforms[j].pos, self.masses[j].value)
        self.velocities[i].vector += accel * self.dt
        self.transforms[i].pos    += self.velocities[i].vector * self.dt


# ── Engine setup ─────────────────────────────────────────────────────────────
engine = mx.Engine("N-Body Compute (transpiled)")
engine.camera.fit(SIZE)

positions = mx.random.positions_in_box(NUM_BODIES, half_size=SIZE, rng=7)
mass_values = mx.random.scalars_uniform(NUM_BODIES, low=0.5, high=3.0, rng=7)
scales = (mass_values ** (1 / 3)).reshape(-1, 1)
initial_velocities = np.zeros((NUM_BODIES, 3), dtype=np.float32)

engine.spawn(
    Mesh(sphere(SPHERE_RADIUS, 12)),
    Material(PhongMaterial("#ffaa44")),
    Transform(pos=positions, scale=scales),
    Velocity(vector=initial_velocities),
    Mass(value=mass_values),
    n=NUM_BODIES,
)

engine.compute(GravityKernel)


if __name__ == "__main__":
    engine.cli()
