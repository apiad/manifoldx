"""Scatter plot — sci-viz primitives v1, plan 4 (Altair-style shim).

The "hello world" of the declarative API. ~30 lines including imports
to render a 500-particle 3D scatter plot with axes, a colormap legend,
a screen-anchored scale bar, lighting, and live physics that updates
the colors via a per-frame callback.

Compare with examples/nbody.py (the imperative-ECS version): same
visual surface, ~5x less code.
"""

import manifoldx.viz as mxv
import numpy as np
from manifoldx.resources import PointLight, StandardMaterial, sphere

# Live data (the user mutates these in-place each frame) ---------------------
N = 500
rng = np.random.default_rng(7)
positions = rng.uniform(-3.0, 3.0, (N, 3)).astype(np.float32)
velocities = rng.normal(0.0, 0.6, (N, 3)).astype(np.float32)
speeds = np.linalg.norm(velocities, axis=1).astype(np.float32)
radii = rng.uniform(0.04, 0.10, N).astype(np.float32)

# Declarative scene ---------------------------------------------------------
chart = (
    mxv.points(
        positions=positions,
        color=mxv.color(speeds, cmap="inferno", domain=(0.0, 2.0), title="Speed"),
        size=radii,
    )
    + mxv.mesh(
        geometry=sphere(0.3, 16),
        material=StandardMaterial("#cccccc", roughness=0.4, metallic=0.1),
    )
    + mxv.axes(extent=4.0)
    + mxv.legend(cmap="inferno", title="Speed", position="bottom-right")
    + mxv.scale_bar(ndc_length=0.25, label="2 units", position="bottom-left")
    + mxv.lights(
        [
            PointLight(color="#ffffff", intensity=10.0, position=(5, 5, 5)),
            PointLight(color="#88aaff", intensity=4.0, position=(-3, -2, -3)),
        ]
    )
)


# Live physics: random drift bouncing inside a (-3, 3)^3 box -----------------
@chart.simulate
def step(dt):
    positions[:] += velocities * dt
    # Reflect at the walls.
    out = np.abs(positions) > 3.0
    velocities[out] *= -1.0
    speeds[:] = np.linalg.norm(velocities, axis=1)


# Camera framing — drop down to the imperative engine via chart.engine. This
# is the escape hatch the shim's design promises: declarative chart on top,
# imperative ECS below for things the shim doesn't yet wrap.
chart.engine.camera.fit(6.0, azimuth=35, elevation=25)


if __name__ == "__main__":
    chart.cli()
