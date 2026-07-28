"""Utah teapot rendered with a textured albedo map under PBR.

The classic Newell teapot with a "blue-and-white china" albedo, four
point lights orbiting overhead, slow camera drift. Showcases the OBJ
loader (Slice 2) feeding into the textured-PBR pipeline (Slice 1).

Run interactively:
    uv run python examples/teapot_demo.py

Render to video:
    uv run python examples/teapot_demo.py --render --duration 8 --fps 30 \
        --output /tmp/teapot.mp4
"""

from pathlib import Path

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, PointLight


ASSETS = Path(__file__).parent / "assets" / "teapot"
TEAPOT_SCALE = 0.08   # the Newell teapot bbox is ~32x16x20 — scale to fit


engine = mx.Engine("Utah Teapot", width=900, height=900)

# Four warm-white lights around the teapot from above, plus a soft cool fill.
LIGHT_DISTANCE = 5.0
LIGHT_HEIGHT = 4.0
for x, z in [(LIGHT_DISTANCE, LIGHT_DISTANCE),
             (-LIGHT_DISTANCE, LIGHT_DISTANCE),
             (LIGHT_DISTANCE, -LIGHT_DISTANCE),
             (-LIGHT_DISTANCE, -LIGHT_DISTANCE)]:
    engine.add_light(PointLight(position=(x, LIGHT_HEIGHT, z),
                                 color="#fff4e0",
                                 intensity=35.0))
engine.add_light(PointLight(position=(0, -3, 0),
                             color="#7799cc",
                             intensity=4.0))


@engine.on("startup")
def setup(_payload):
    teapot_geo = mx.load_obj(ASSETS / "teapot.obj")
    albedo = mx.load_texture(engine, ASSETS / "teapot_albedo.png")

    mat = StandardMaterial(
        color="#ffffff",
        roughness=0.35,
        metallic=0.05,
        ao=1.0,
        albedo_map=albedo,
    )

    engine.spawn(
        Mesh(teapot_geo),
        Material(mat),
        Transform(pos=(0, -0.6, 0),
                  scale=(TEAPOT_SCALE,) * 3),
    )

    # Pull camera back, tilt down toward the teapot.
    engine.camera.set_pose(position=(0, 1.5, 4.5), target=(0, 0, 0))


@engine.system
def slow_orbit(query, dt: float):
    """Slowly orbit the camera so the texture rotates with respect to the
    lighting — makes the PBR specular highlights read as 'this is a glazed
    surface, not a flat sticker'."""
    engine.camera.orbit(10 * dt, 0)


if __name__ == "__main__":
    engine.cli(fps=30, duration=8, output="teapot.mp4")
