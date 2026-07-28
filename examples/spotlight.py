"""Spot-light (flashlight) demo — a moving cone of light casting live shadows.

A single spot light sweeps around a dark scene like a flashlight: a bright cone
with soft angular falloff and distance attenuation. It is the shadow caster, so
as the light orbits, every object's shadow swings around the floor and across
its neighbours. There is no sun — outside the cone the scene falls to ambient
black, which makes the beam (and the moving shadows) read dramatically.

    uv run python examples/spot_demo.py
    uv run python examples/spot_demo.py --render --duration 8 --output /tmp/spot.mp4
"""

from pathlib import Path

import numpy as np

import manifoldx as mx

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import SpotLight, StandardMaterial, sphere, cube, plane

ASSETS = Path(__file__).parent / "assets" / "teapot"

engine = mx.Engine("Flashlight", width=1024, height=768)

# The flashlight: a focused cone (inner→outer falloff) that also casts shadows.
engine.set_spot(
    SpotLight(
        color="#fff2d0",
        intensity=220.0,
        position=(6.0, 4.5, 0.0),
        direction=(-1.0, -0.7, 0.0),
        inner_angle=0.24,
        outer_angle=0.38,
    )
)
engine.enable_shadows(resolution=2048, near=0.2, bias=0.003, pcf_radius=2)

# --- Ground plane --------------------------------------------------------
FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)
engine.spawn(
    Mesh(plane(24, 24)),
    Material(StandardMaterial(color="#b8b8b8", roughness=0.95, metallic=0.0)),
    Transform(pos=(0, 0, 0), rot=FLOOR_ROT),
)

# --- Static scene the flashlight sweeps over -----------------------------
engine.spawn(
    Mesh(mx.load_obj(ASSETS / "teapot.obj")),
    Material(StandardMaterial(color="#2a9d8f", roughness=0.3, metallic=0.05)),
    Transform(pos=(0, 0.945, 0), scale=(0.12,) * 3),
)
for pos, col, rough, metal in [
    ((-2.6, 0.8, 0.6), "#e8b23a", 0.25, 0.9),
    ((2.4, 0.9, -1.4), "#d43a2f", 0.9, 0.0),
    ((1.6, 0.7, 2.2), "#3a6de0", 0.15, 0.1),
]:
    engine.spawn(
        Mesh(sphere(0.8, 40)),
        Material(StandardMaterial(color=col, roughness=rough, metallic=metal)),
        Transform(pos=pos),
    )
engine.spawn(
    Mesh(cube(1.1, 1.1, 1.1)),
    Material(StandardMaterial(color="#e8e8e8", roughness=0.5, metallic=0.0)),
    Transform(pos=(-1.8, 0.55, -2.2)),
)

_AIM = np.array([0.0, 0.6, 0.0])  # the flashlight always points at the scene centre


@engine.system
def sweep_flashlight(query: mx.Query[Transform], dt: float):
    t = engine.elapsed
    px, pz = 6.0 * np.cos(t * 0.6), 6.0 * np.sin(t * 0.6)
    pos = np.array([px, 4.5, pz], dtype=np.float32)
    engine._spot.position = tuple(pos)
    d = _AIM - pos
    engine._spot.direction = tuple(d / np.linalg.norm(d))


engine.camera.fit(radius=10.0, center=(0, 0.6, 0), azimuth=35, elevation=42)


if __name__ == "__main__":
    engine.cli()
