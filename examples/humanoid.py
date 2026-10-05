"""Posable humanoid mannequins: three figures, three proportion styles.

    uv run python examples/humanoid.py
    uv run python examples/humanoid.py --render --duration 2 --output /tmp/humanoid.mp4
"""
import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.humanoid import Proportions, facing, solve, spawn_mannequin
from manifoldx.resources import DirectionalLight, StandardMaterial, plane

engine = mx.Engine("Humanoid", width=1280, height=720)
engine.background_color = (0.86, 0.87, 0.9)
engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.6, -0.8, -0.4)))
engine.enable_shadows(resolution=2048, bias=0.003, pcf_radius=2)
engine.spawn(Mesh(plane(20, 20)), Material(StandardMaterial(color="#b8b8b8", roughness=0.95)),
             Transform(pos=(0, 0, 0), rot=(-0.70710678, 0.0, 0.0, 0.70710678)))

cast = [
    ("stand", Proportions.measured("male"), 1.78, (-1.6, 0, 0), "#d1495b"),
    ({"base": "hands_on_hips", "neck": {"lean": -10}}, Proportions.styled("female", "disney"), 1.65, (0, 0, -0.4), "#edae49"),
    ({"base": "t", "shoulder_r": {"abduct": 150}}, Proportions.styled("male", "chibi"), 1.0, (1.5, 0, 0.2), "#2a9d5c"),
]
for pose, props, height, at, color in cast:
    yaw = facing((at[0], at[2]), (0.0, 4.0))  # everyone looks toward the camera side
    spawn_mannequin(engine, solve(pose, props, height=height, yaw=yaw), at=at, color=color)

engine.camera.set_pose(position=(0, 1.5, 6.5), target=(0, 0.9, 0))

if __name__ == "__main__":
    engine.cli()
