"""Volume rendering demo — Gaussian blob with inferno colormap.

A static 64^3 scalar field uploaded once and raymarched each frame.
Camera orbits to make the box-shaped bounds and the radial density
falloff visible.
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Transform
from manifoldx.viz import Volume, VolumeMaterial


# ── Scalar field ─────────────────────────────────────────────────────────────
N = 64
xs = np.linspace(-1, 1, N, dtype=np.float32)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
density = np.exp(-(X**2 + Y**2 + Z**2) / 0.15).astype(np.float32)


# ── Engine ───────────────────────────────────────────────────────────────────
engine = mx.Engine("Volume rendering — Gaussian blob")
engine.camera.fit(2.0)

vol_id = engine.register_volume(density, name="gaussian")
engine.spawn(
    Volume(volume_id=vol_id),
    Material(VolumeMaterial(
        cmap="inferno",
        vmin=0.0, vmax=1.0,
        opacity_stops=[(0.0, 0.0), (0.2, 0.04), (0.6, 0.20), (1.0, 0.55)],
        density_scale=1.0,
        max_steps=256,
    )),
    Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
    n=1,
)


if __name__ == "__main__":
    engine.cli()
