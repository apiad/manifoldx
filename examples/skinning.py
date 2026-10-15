"""Rigid GPU skinning: one mesh, several parts, each posed by its own joint matrix.

The figure is a single mesh whose vertices carry a joint index (Mesh(joints=)).
Every frame, set_joints uploads one 4x4 matrix per joint and the vertex shader
moves each part, in its shadow too; the vertex buffer is never rewritten.

    uv run python examples/skinning.py
    uv run python examples/skinning.py --render --duration 2 --fps 30 --output /tmp/skinning.mp4
"""

import math

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Mesh as Shape
from manifoldx.resources import DirectionalLight, StandardMaterial

# Parts: (size, centre, joint pivot). Joint i moves part i about its pivot.
PARTS = [
    ((0.8, 1.0, 0.4), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),     # 0 body
    ((0.5, 0.5, 0.5), (0.0, 0.8, 0.0), (0.0, 0.5, 0.0)),     # 1 head, turns on the neck
    ((0.2, 0.8, 0.2), (-0.55, 0.1, 0.0), (-0.55, 0.45, 0.0)),  # 2 left arm, swings at the shoulder
    ((0.2, 0.8, 0.2), (0.55, 0.1, 0.0), (0.55, 0.45, 0.0)),   # 3 right arm
]


def figure() -> Shape:
    boxes = [Shape.box(*size).translate(centre).recompute_normals() for size, centre, _ in PARTS]
    offsets = np.cumsum([0] + [len(b.positions) for b in boxes[:-1]])
    return Shape(
        positions=np.concatenate([b.positions for b in boxes]),
        faces=np.concatenate([b.faces + o for b, o in zip(boxes, offsets)]).astype(np.uint32),
        normals=np.concatenate([b.normals for b in boxes]),
        joints=np.concatenate([np.full(len(b.positions), i) for i, b in enumerate(boxes)]),
    )


def about(pivot, axis, angle):
    """4x4 rotation by `angle` about the x or y axis through `pivot`."""
    c, s = math.cos(angle), math.sin(angle)
    r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.asarray(pivot) - r @ np.asarray(pivot)
    return m


engine = mx.Engine("Skinning")
engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.9, -1.0, -0.6)))
engine.enable_shadows()
engine.camera.set_pose(position=(1.5, 1.0, 3.5), target=(0, 0.2, 0))
figure_entity = None


@engine.on("startup")
def spawn(_payload):
    global figure_entity
    engine.spawn(Mesh(Shape.plane(6.0, 6.0).translate((0, -0.5, 0)).to_geometry()),
                 Material(StandardMaterial(color="#d8d8d8", roughness=0.9)), Transform())
    figure_entity = engine.spawn(
        Mesh(figure().to_geometry()),
        Material(StandardMaterial(color="#e07a3a", roughness=0.5)),
        Transform(pos=(0, 0, 0)),
    )


@engine.on("frame")
def pose(payload):
    t = payload["elapsed"]
    swing = 0.9 * math.sin(2.0 * t)
    figure_entity.set_joints([
        about(PARTS[0][2], "y", 0.4 * t),                               # body turns slowly
        about(PARTS[0][2], "y", 0.4 * t) @ about(PARTS[1][2], "y", 0.6 * math.sin(t)),
        about(PARTS[0][2], "y", 0.4 * t) @ about(PARTS[2][2], "x", swing),
        about(PARTS[0][2], "y", 0.4 * t) @ about(PARTS[3][2], "x", -swing),
    ])


if __name__ == "__main__":
    engine.cli()
