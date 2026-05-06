"""Point cloud demo — sci-viz primitives v1 (Plan 1).

5000 particles arranged in a Gaussian cloud, drifting toward the origin
under a soft attractive force plus per-particle Brownian motion. Each
particle's color is mapped from its current speed through the viridis
colormap (slow → dark purple, fast → yellow). Sphere-imposter point
sprites scale with the per-particle Radius.

Demonstrates:
- PointCloud + ColormapMaterial + ScalarValue + Radius
- Per-frame mutation of ScalarValue propagates to GPU
- Camera-facing point sprites at scale (5000 instances → 1 draw call)
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Material, Transform
from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue

NUM_PARTICLES = 5000
EXTENT = 10.0          # initial cloud half-width
ATTRACTION = 0.4       # pull-toward-origin strength
NOISE = 0.6            # per-step random walk magnitude
DRAG = 0.92            # velocity damping per step
MAX_DISPLAY_SPEED = 3.0  # vmax for the colormap

engine = mx.Engine("Point cloud demo")
engine.camera.fit(EXTENT * 1.5)

# Pre-register the viz components — the spawn flow needs them present
# in EntityStore. Plan 4 will hide this behind a functional shim.
engine.store.register_component("PointCloud", np.dtype("f4"), (0,))
engine.store.register_component("ScalarValue", np.dtype("f4"), (1,))
engine.store.register_component("Radius", np.dtype("f4"), (1,))

rng = np.random.default_rng(42)
positions = rng.normal(0.0, EXTENT * 0.5, (NUM_PARTICLES, 3)).astype(np.float32)
radii = rng.uniform(0.04, 0.10, NUM_PARTICLES).astype(np.float32)
initial_speeds = np.zeros(NUM_PARTICLES, dtype=np.float32)

engine.spawn(
    PointCloud(),
    Material(ColormapMaterial(cmap="viridis", vmin=0.0, vmax=MAX_DISPLAY_SPEED)),
    Transform(pos=positions),
    ScalarValue(value=initial_speeds),
    Radius(radius=radii),
    n=NUM_PARTICLES,
)

# Velocities live outside the ECS — pure NumPy state mutated by the system.
velocities = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)


@engine.system
def drift_and_jitter(query: mx.Query[Transform, ScalarValue], dt: float):
    global velocities

    pos = query[Transform].pos.data  # (N, 3) snapshot

    # Soft attraction toward origin: a = -k * r
    forces = -ATTRACTION * pos

    # Brownian noise — uncorrelated per axis per step
    forces += rng.normal(0.0, NOISE, pos.shape).astype(np.float32)

    velocities[:] = velocities * DRAG + forces * dt
    speeds = np.linalg.norm(velocities, axis=1)

    query[Transform].pos += velocities * dt
    query[ScalarValue].value = speeds.reshape(-1, 1)  # column has shape (N, 1)


if __name__ == "__main__":
    engine.cli()
