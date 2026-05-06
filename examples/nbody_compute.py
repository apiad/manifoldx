"""N-Body simulation with GPU compute shader.

A side-by-side counterpart to examples/nbody.py: same physics, same
visuals, but the all-pairs O(N²) gravity loop runs in a WGSL compute
shader instead of pure-numpy on CPU.

Demonstrates the compute-systems extensibility surface end-to-end:
- Two custom Component subclasses (Velocity, Mass) used as compute
  inputs / outputs.
- A GravityKernel(Compute) subclass declares its bindings via
  Reads/ReadsWrites/Uniform annotations and provides raw WGSL via
  compile().
- The engine auto-registers the components, compiles the pipeline
  on first frame, dispatches every frame between CPU command flush
  and the render pass.

Phase 2 (still to design) will replace `compile()` with a Python
`def main(self, i)` body that gets traced to WGSL by a code generator;
the rest of this file stays identical.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
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


# ── GPU compute kernel ───────────────────────────────────────────────────────
_GRAVITY_WGSL = """
struct Uniforms { G: f32, softening: f32, dt: f32, n: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> masses: array<f32>;
@group(0) @binding(2) var<storage, read_write> transforms: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocities: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(uniforms.n);
    if (i >= n) { return; }

    // Transform layout: pos.xyz (3) + rot.xyzw (4) + scale.xyz (3) = 10 floats.
    let stride_t = 10u;
    // Velocity layout: vector.xyz (3 floats).
    let stride_v = 3u;

    let pos_i = vec3<f32>(
        transforms[i * stride_t + 0u],
        transforms[i * stride_t + 1u],
        transforms[i * stride_t + 2u],
    );

    // All-pairs gravity. One thread per body; sum forces from all others.
    var accel = vec3<f32>(0.0);
    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }
        let pos_j = vec3<f32>(
            transforms[j * stride_t + 0u],
            transforms[j * stride_t + 1u],
            transforms[j * stride_t + 2u],
        );
        let diff = pos_j - pos_i;
        let r2 = dot(diff, diff) + uniforms.softening * uniforms.softening;
        let inv_r3 = 1.0 / (r2 * sqrt(r2));
        accel = accel + uniforms.G * masses[j] * diff * inv_r3;
    }

    // Integrate velocity, then position.
    var vel = vec3<f32>(
        velocities[i * stride_v + 0u],
        velocities[i * stride_v + 1u],
        velocities[i * stride_v + 2u],
    );
    vel = vel + accel * uniforms.dt;

    velocities[i * stride_v + 0u] = vel.x;
    velocities[i * stride_v + 1u] = vel.y;
    velocities[i * stride_v + 2u] = vel.z;

    transforms[i * stride_t + 0u] = pos_i.x + vel.x * uniforms.dt;
    transforms[i * stride_t + 1u] = pos_i.y + vel.y * uniforms.dt;
    transforms[i * stride_t + 2u] = pos_i.z + vel.z * uniforms.dt;
}
""".strip()


class GravityKernel(Compute):
    """N-body gravity + velocity-and-position integration on the GPU."""

    masses:     Reads[Mass]
    transforms: ReadsWrites[Transform]
    velocities: ReadsWrites[Velocity]

    G:         Uniform[float] = G
    softening: Uniform[float] = SOFTENING
    dt:        Uniform[float] = "frame_dt"      # auto-bound, re-uploaded each frame
    n:         Uniform[float] = "entity_count"  # auto-bound, re-uploaded each frame

    workgroup_size = 64
    dispatch = "entity_count"

    def compile(self) -> str:
        return _GRAVITY_WGSL


# ── Engine setup ─────────────────────────────────────────────────────────────
engine = mx.Engine("N-Body Compute")
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
