"""PBR Demo: 3x2 grid of objects with three orbiting lights.

Layout (viewed from above):

         col 0 (smooth)    col 1 (medium)    col 2 (rough)
  back   ┌────────────────┬────────────────┬────────────────┐
  row 0  │ Red metal cube │ Gold metal cube│ Copper met cube│
         ├────────────────┼────────────────┼────────────────┤
  front  │ Blue sphere    │ Green sphere   │ Purple sphere  │
  row 1  └────────────────┴────────────────┴────────────────┘

Grid is in the XZ plane (Y is up). Camera looks from above-front.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, PointLight, cube, sphere

engine = mx.Engine("PBR Demo")

# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------
cube_geo = cube(1, 1, 1)
sphere_geo = sphere(0.7, 32)

# ---------------------------------------------------------------------------
# Materials – increasing roughness left→right
# ---------------------------------------------------------------------------
materials = [
    # Row 0 (back) – metallic cubes
    StandardMaterial(color="#ff3333", roughness=0.10, metallic=0.8),
    StandardMaterial(color="#ffdd33", roughness=0.45, metallic=0.8),
    StandardMaterial(color="#dd7744", roughness=0.85, metallic=0.8),
    # Row 1 (front) – dielectric spheres
    StandardMaterial(color="#3366ff", roughness=0.10, metallic=0.2),
    StandardMaterial(color="#33cc55", roughness=0.45, metallic=0.2),
    StandardMaterial(color="#8833ff", roughness=0.85, metallic=0.2),
]

# ---------------------------------------------------------------------------
# 3 × 2 grid in the XZ plane (horizontal table), Y is up
# ---------------------------------------------------------------------------
spacing_x = 2.5
spacing_z = 2.5
cols, rows = 3, 2
x_offset = -(cols - 1) * spacing_x / 2
z_offset = -(rows - 1) * spacing_z / 2

for row in range(rows):
    for col in range(cols):
        idx = row * cols + col
        geo = cube_geo if row == 0 else sphere_geo
        x = x_offset + col * spacing_x
        z = z_offset + row * spacing_z
        engine.spawn(
            Mesh(geo),
            Material(materials[idx]),
            Transform(pos=(x, 0, z)),
            n=1,
        )

# ---------------------------------------------------------------------------
# Camera: frame the scene from above-front
# ---------------------------------------------------------------------------
engine.camera.fit(radius=5.0, center=(0, 0, 0), azimuth=30, elevation=35)

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

    # Equatorial – XZ plane (around Y axis)
    light_equatorial.position = (
        ORBIT_RADIUS * np.cos(t * 0.7),
        1.0,
        ORBIT_RADIUS * np.sin(t * 0.7),
    )

    # Polar – XY plane (around Z axis)
    light_polar.position = (
        ORBIT_RADIUS * np.cos(t * 0.5),
        ORBIT_RADIUS * np.sin(t * 0.5),
        0.0,
    )

    # Sideways – YZ plane (around X axis)
    light_sideways.position = (
        0.0,
        ORBIT_RADIUS * np.sin(t * 0.6),
        ORBIT_RADIUS * np.cos(t * 0.6),
    )


@engine.system
def camera_orbit(query: mx.Query[Transform], dt: float):
    engine.camera.orbit(8 * dt, 0)


engine.run()
