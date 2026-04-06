import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, PointLight, cube, sphere

engine = mx.Engine("PBR Demo")
engine.camera.zoom(0.15)
engine.camera.orbit(30, 20)

# Create geometries
cube_geo = cube(1, 1, 1)
sphere_geo = sphere(0.7, 24)

# Create PBR materials with different properties
red_shiny = StandardMaterial(color="#ff3333", roughness=0.15, metallic=0.9)
green_dull = StandardMaterial(color="#33ff33", roughness=0.8, metallic=0.0)
blue_medium = StandardMaterial(color="#3333ff", roughness=0.45, metallic=0.5)
gold_shiny = StandardMaterial(color="#ffdd33", roughness=0.1, metallic=1.0)
purple_dull = StandardMaterial(color="#8833ff", roughness=0.9, metallic=0.0)
cyan_medium = StandardMaterial(color="#33ffff", roughness=0.5, metallic=0.3)

# Set up lights
light1 = PointLight(color="#ffffff", intensity=2.0, position=(3, 4, 3))
light2 = PointLight(color="#ff8844", intensity=1.5, position=(-3, 2, -3))
light3 = PointLight(color="#4488ff", intensity=1.0, position=(0, -2, 3))

engine.set_lights([light1, light2, light3])

# Spawn 6 objects with different PBR materials
engine.spawn(
    Mesh(cube_geo),
    Material(red_shiny),
    Transform(pos=(-2.5, 0, 0)),
    n=1,
)

engine.spawn(
    Mesh(cube_geo),
    Material(green_dull),
    Transform(pos=(-2.5, 0, 2)),
    n=1,
)

engine.spawn(
    Mesh(sphere_geo),
    Material(blue_medium),
    Transform(pos=(2.5, 0, 0)),
    n=1,
)

engine.spawn(
    Mesh(sphere_geo),
    Material(gold_shiny),
    Transform(pos=(2.5, 0, 2)),
    n=1,
)

engine.spawn(
    Mesh(cube_geo),
    Material(purple_dull),
    Transform(pos=(0, 0, -3), scale=(1.2, 1.2, 1.2)),
    n=1,
)

engine.spawn(
    Mesh(sphere_geo),
    Material(cyan_medium),
    Transform(pos=(0, 2, -2), scale=(0.8, 0.8, 0.8)),
    n=1,
)


# Animate lights
@engine.system
def animate_lights(dt: float):
    t = engine.elapsed

    # Orbit light1 around the scene
    light1.position = (
        4 * np.sin(t * 0.5),
        3 + np.cos(t * 0.3) * 1.5,
        4 * np.cos(t * 0.5),
    )


# Slow camera orbit
@engine.system
def camera_orbit(dt: float):
    engine.camera.orbit(5 * dt, 0)


print("Starting PBR demo with:")
print("  - 6 objects (cubes + spheres)")
print("  - Different PBR materials (roughness/metallic vary)")
print("  - 3 animated point lights")
print("  - Multi-instance per draw call")

engine.run()
