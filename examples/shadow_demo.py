"""Directional shadow mapping demo — a sphere casts a shadow on the ground.

A single directional "sun" lights the scene and casts a hard shadow map.
The camera orbits so you can watch the cast shadow move across the floor.

    uv run python examples/shadow_demo.py
    uv run python examples/shadow_demo.py --render --duration 4 --output /tmp/shadow.mp4
"""

import manifoldx as mx

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import DirectionalLight, StandardMaterial, sphere, plane

engine = mx.Engine("Shadow Mapping", width=1024, height=768)

# The sun: a directional light that also casts the shadow.
engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.4, -1.0, -0.35)))
engine.enable_shadows(target=(0, 0, 0), extent=6.0, resolution=2048, near=0.1, far=40.0, bias=0.004)

# Ground plane. plane()'s quad faces +Z, so rotate -90deg about X to make it a
# floor (normal +Y). Quaternion (x, y, z, w) = (sin(-45), 0, 0, cos(-45)).
FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)
engine.spawn(
    Mesh(plane(16, 16)),
    Material(StandardMaterial(color="#cccccc", roughness=0.95, metallic=0.0)),
    Transform(pos=(0, 0, 0), rot=FLOOR_ROT),
)

# A ball hovering above the floor.
engine.spawn(
    Mesh(sphere(1.0, 48)),
    Material(StandardMaterial(color="#e05a3a", roughness=0.4, metallic=0.1)),
    Transform(pos=(0, 1.6, 0)),
)

engine.camera.fit(radius=6.0, center=(0, 1.0, 0), azimuth=30, elevation=35)


@engine.system
def camera_orbit(query: mx.Query[Transform], dt: float):
    engine.camera.orbit(10 * dt, 0)


if __name__ == "__main__":
    engine.cli()
