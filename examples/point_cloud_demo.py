"""Point cloud demo — sci-viz primitives v1 (Plan 1).

A protoplanetary-disk-style demo: ~5000 dust particles orbiting a central
star under Keplerian gravity. Each particle is rendered as a camera-facing
point sprite with sphere-imposter shading and colormapped by orbital speed
(inferno LUT — slow → black, fast → yellow). The central star is a real
PBR sphere lit by a bright point light at the origin.

Demonstrates Plan 1's full surface end-to-end:
- PointCloud + ColormapMaterial + ScalarValue + Radius (sprite path)
- Mesh + StandardMaterial + Transform (PBR mesh path) coexisting in
  the same scene as the sprites
- Per-frame mutation of ScalarValue propagating to GPU
- 5000 sprites + 1 mesh = 2 draw calls per frame
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import PointLight, StandardMaterial, sphere
from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue

# Simulation parameters -------------------------------------------------------
NUM_PARTICLES = 5000
GM = 60.0           # gravitational parameter of the central star
SOFTENING = 0.4     # avoids singularity / wild accelerations near the star
DISK_INNER = 2.0    # inner edge of the disk
DISK_OUTER = 8.0    # outer edge of the disk
DISK_THICK = 0.18   # z-axis half-width of the disk
ECCENTRICITY = 0.06  # fractional perturbation on circular orbits
STAR_RADIUS = 1.0   # central PBR sphere
EXTENT = DISK_OUTER * 1.4

engine = mx.Engine("Protoplanetary disk")
engine.camera.fit(EXTENT)

# Pre-register the viz components — Plan 4's functional shim hides this.
engine.store.register_component("PointCloud", np.dtype("f4"), (0,))
engine.store.register_component("ScalarValue", np.dtype("f4"), (1,))
engine.store.register_component("Radius", np.dtype("f4"), (1,))

# Central star: real 3D PBR sphere lit by a bright point light at the origin.
# StandardMaterial parameters tuned for a warm, diffuse star surface.
engine.spawn(
    Mesh(sphere(STAR_RADIUS, 32)),
    Material(StandardMaterial("#ffd45a", roughness=0.85, metallic=0.0)),
    Transform(pos=(0.0, 0.0, 0.0)),
    n=1,
)
engine.set_lights(
    [
        PointLight(color="#ffe9a8", intensity=80.0, position=(0.0, 0.0, 0.0)),
        PointLight(color="#ffffff", intensity=8.0, position=(6.0, 8.0, 6.0)),
    ]
)

# Disk particles --------------------------------------------------------------
# The star occupies entity index 0; particles occupy indices 1..NUM_PARTICLES.
# Per-frame physics runs over all alive entities (Plan 1's Query doesn't filter
# per-component). The star's own velocity stays zero — softened gravity at the
# origin returns ~0 acceleration, so it doesn't drift.
rng = np.random.default_rng(7)

# Surface-density-uniform sampling: r² uniform in [r_min², r_max²].
r = np.sqrt(rng.uniform(DISK_INNER**2, DISK_OUTER**2, NUM_PARTICLES)).astype(np.float32)
theta = rng.uniform(0.0, 2.0 * np.pi, NUM_PARTICLES).astype(np.float32)
z = rng.normal(0.0, DISK_THICK, NUM_PARTICLES).astype(np.float32)

positions = np.stack([r * np.cos(theta), z, r * np.sin(theta)], axis=1).astype(np.float32)

# Tangential unit vector in the xz-plane (counter-clockwise looking down +y).
tangent = np.stack([-np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)

# Circular-orbit speed at each radius: v = sqrt(GM / r).
v_circ = np.sqrt(GM / r).astype(np.float32)

# Initial velocities: tangential + small radial/vertical perturbation for
# elliptical orbits and a non-zero scale-height.
particle_velocities = (tangent.T * v_circ).T
particle_velocities += (
    rng.normal(0.0, ECCENTRICITY, particle_velocities.shape).astype(np.float32)
    * v_circ[:, None]
)

# Per-particle radius: brighter dust closer in, dustier farther out.
radii = rng.uniform(0.04, 0.10, NUM_PARTICLES).astype(np.float32)

initial_speeds = np.linalg.norm(particle_velocities, axis=1).astype(np.float32)

engine.spawn(
    PointCloud(),
    Material(ColormapMaterial(cmap="inferno", vmin=2.5, vmax=6.5)),
    Transform(pos=positions),
    ScalarValue(value=initial_speeds),
    Radius(radius=radii),
    n=NUM_PARTICLES,
)

# Velocities live outside the ECS. Slot 0 is the star (always zero); slots
# 1..NUM_PARTICLES are the dust particles.
N_TOTAL = NUM_PARTICLES + 1
velocities = np.zeros((N_TOTAL, 3), dtype=np.float32)
velocities[1:] = particle_velocities


@engine.system
def gravity(query: mx.Query[Transform, ScalarValue], dt: float):
    """Soft gravity from the central star + speed-driven colormap."""
    global velocities

    pos = query[Transform].pos.data  # (N_TOTAL, 3) snapshot

    # Acceleration: a = -GM / (r² + ε²)^(3/2) * pos
    r2 = (pos**2).sum(axis=1) + SOFTENING**2
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    accel = -GM * pos * inv_r3[:, None]

    velocities[:] = velocities + accel * dt
    speeds = np.linalg.norm(velocities, axis=1)

    query[Transform].pos += velocities * dt
    query[ScalarValue].value = speeds.reshape(-1, 1)


if __name__ == "__main__":
    engine.cli()
