"""Directional shadow mapping demo — many objects in motion casting shadows.

A single directional "sun" lights the scene and casts a hard shadow map. Four
objects (three spheres + a cube) orbit at different radii, heights and speeds,
bobbing up and down, over a central cube pillar and a ground plane. Their cast
shadows sweep across the floor, climb the pillar, and cross each other as they
pass overhead. Different surfaces (metallic gold, matte red, polished blue,
white cube) show how the shadow reads against varied materials.

    uv run python examples/shadow_demo.py
    uv run python examples/shadow_demo.py --render --duration 8 --output /tmp/shadow.mp4
"""

import numpy as np

import manifoldx as mx

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import DirectionalLight, StandardMaterial, sphere, cube, plane

engine = mx.Engine("Shadow Mapping", width=1024, height=768)

# The sun: a directional light that also casts the shadow.
engine.set_sun(DirectionalLight(color="#fff4e0", intensity=2.6, direction=(-0.45, -1.0, -0.3)))
engine.enable_shadows(target=(0, 0, 0), extent=8.5, resolution=2048, near=0.1, far=45.0, bias=0.004)

# --- Ground plane (entity 0) ---------------------------------------------
# plane()'s quad faces +Z, so rotate -90deg about X to make it a floor
# (normal +Y). Quaternion (x, y, z, w) = (sin(-45), 0, 0, cos(-45)).
FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)
engine.spawn(
    Mesh(plane(16, 16)),
    Material(StandardMaterial(color="#c8c8c8", roughness=0.95, metallic=0.0)),
    Transform(pos=(0, 0, 0), rot=FLOOR_ROT),
)

# --- Central pillar (entity 1, static) — a surface others' shadows sweep --
engine.spawn(
    Mesh(cube(1.2, 3.0, 1.2)),
    Material(StandardMaterial(color="#4a9d5b", roughness=0.7, metallic=0.0)),
    Transform(pos=(0, 1.5, 0)),
)

# --- Orbiting movers (entities 2..5) -------------------------------------
# Each object: distinct mesh + material + orbit. Positions are recomputed
# every frame from engine.elapsed in the animate system below.
engine.spawn(
    Mesh(sphere(0.8, 40)),
    Material(StandardMaterial(color="#e8b23a", roughness=0.2, metallic=0.9)),  # gold metal
    Transform(pos=(2.6, 1.8, 0)),
)
engine.spawn(
    Mesh(sphere(0.9, 40)),
    Material(StandardMaterial(color="#d43a2f", roughness=0.9, metallic=0.0)),  # matte red
    Transform(pos=(3.6, 1.2, 0)),
)
engine.spawn(
    Mesh(sphere(0.7, 40)),
    Material(StandardMaterial(color="#3a6de0", roughness=0.1, metallic=0.1)),  # polished blue
    Transform(pos=(1.6, 2.7, 0)),
)
engine.spawn(
    Mesh(cube(1.0, 1.0, 1.0)),
    Material(StandardMaterial(color="#e8e8e8", roughness=0.5, metallic=0.0)),  # white cube
    Transform(pos=(3.0, 2.3, 0)),
)

# Orbit params per mover, indexed to entities 2..5:
#   (radius, base_y, bob_amp, bob_speed, ang_speed, phase)
_MOVERS = [
    (3.0, 1.8, 0.7, 1.3, 0.90, 0.0),
    (4.4, 1.1, 0.5, 0.9, -0.55, 2.1),
    (2.2, 2.6, 0.7, 1.7, 1.40, 4.2),
    (3.6, 2.4, 1.0, 1.05, 0.45, 1.0),
]
_FIRST_MOVER = 2  # entities 0 (floor) and 1 (pillar) are static


@engine.system
def animate(query: mx.Query[Transform], dt: float):
    t = engine.elapsed
    positions = query[Transform].pos.data.copy()  # (n, 3); rows 0,1 static
    for i, (r, by, ba, bs, asp, ph) in enumerate(_MOVERS):
        row = _FIRST_MOVER + i
        positions[row, 0] = r * np.cos(t * asp + ph)
        positions[row, 1] = by + ba * np.sin(t * bs + ph)
        positions[row, 2] = r * np.sin(t * asp + ph)
    query[Transform].pos = positions


engine.camera.fit(radius=7.5, center=(0, 1.0, 0), azimuth=35, elevation=32)


@engine.system
def camera_orbit(query: mx.Query[Transform], dt: float):
    engine.camera.orbit(6 * dt, 0)


if __name__ == "__main__":
    engine.cli()
