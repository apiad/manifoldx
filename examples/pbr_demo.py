"""PBR Demo: 3x2 grid of objects with three orbiting lights."""

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, PointLight, cube, sphere

engine = mx.Engine("PBR Demo")
engine.camera.zoom(0.08)
engine.camera.orbit(20, 25)

# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------
cube_geo = cube(1, 1, 1)
sphere_geo = sphere(0.7, 32)

# ---------------------------------------------------------------------------
# Materials – increasing roughness left→right, top row metallic, bottom row dielectric
# ---------------------------------------------------------------------------
#                  col 0 (smooth)      col 1 (medium)       col 2 (rough)
# Row 0 (cubes)   red metallic        gold metallic        copper metallic
# Row 1 (spheres) blue dielectric     green dielectric     purple dielectric

materials = [
    # Top row – metallic cubes
    StandardMaterial(color="#ff3333", roughness=0.10, metallic=1.0),  # red smooth metal
    StandardMaterial(
        color="#ffdd33", roughness=0.45, metallic=1.0
    ),  # gold medium metal
    StandardMaterial(
        color="#dd7744", roughness=0.85, metallic=1.0
    ),  # copper rough metal
    # Bottom row – dielectric spheres
    StandardMaterial(
        color="#3366ff", roughness=0.10, metallic=0.0
    ),  # blue smooth plastic
    StandardMaterial(
        color="#33cc55", roughness=0.45, metallic=0.0
    ),  # green medium plastic
    StandardMaterial(
        color="#8833ff", roughness=0.85, metallic=0.0
    ),  # purple rough plastic
]

# ---------------------------------------------------------------------------
# 3 × 2 grid layout
# ---------------------------------------------------------------------------
spacing_x = 2.2
spacing_y = 2.2
cols, rows = 3, 2
x_offset = -(cols - 1) * spacing_x / 2
y_offset = -(rows - 1) * spacing_y / 2

for row in range(rows):
    for col in range(cols):
        idx = row * cols + col
        geo = cube_geo if row == 0 else sphere_geo
        x = x_offset + col * spacing_x
        y = y_offset + row * spacing_y
        engine.spawn(
            Mesh(geo),
            Material(materials[idx]),
            Transform(pos=(x, y, 0)),
            n=1,
        )

# ---------------------------------------------------------------------------
# Three orbiting lights on different planes
# ---------------------------------------------------------------------------
ORBIT_RADIUS = 5.0
LIGHT_INTENSITY = 15.0

light_equatorial = PointLight(
    color="#ffffff", intensity=LIGHT_INTENSITY, position=(ORBIT_RADIUS, 0, 0)
)
light_polar = PointLight(
    color="#ffaa44", intensity=LIGHT_INTENSITY, position=(0, ORBIT_RADIUS, 0)
)
light_sideways = PointLight(
    color="#4488ff", intensity=LIGHT_INTENSITY, position=(0, 0, ORBIT_RADIUS)
)

engine.set_lights([light_equatorial, light_polar, light_sideways])


@engine.system
def animate_lights(query: mx.Query[Transform], dt: float):
    t = engine.elapsed

    # Equatorial orbit – rotates in the XZ plane (around Y axis)
    light_equatorial.position = (
        ORBIT_RADIUS * np.cos(t * 0.7),
        0.0,
        ORBIT_RADIUS * np.sin(t * 0.7),
    )

    # Polar orbit – rotates in the XY plane (around Z axis)
    light_polar.position = (
        ORBIT_RADIUS * np.cos(t * 0.5),
        ORBIT_RADIUS * np.sin(t * 0.5),
        0.0,
    )

    # Sideways orbit – rotates in the YZ plane (around X axis)
    light_sideways.position = (
        0.0,
        ORBIT_RADIUS * np.sin(t * 0.6),
        ORBIT_RADIUS * np.cos(t * 0.6),
    )


@engine.system
def camera_orbit(query: mx.Query[Transform], dt: float):
    engine.camera.orbit(8 * dt, 0)


engine.run()
