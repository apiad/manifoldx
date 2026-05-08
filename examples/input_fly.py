"""WASD fly-cam driven by polling state on `engine.input`.

Demonstrates `engine.input.is_pressed(...)` and `engine.input.mouse_delta`
read from inside a system. W/A/S/D translate along the camera basis;
Space/Shift go up/down; right-mouse-drag rotates the look direction.
"""

import math
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query


engine = mx.Engine("Input Fly")

cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.GREEN)


# Fly state mirrors the orbit example pattern.
_fly = {
    "azimuth": 45.0,
    "elevation": 0.0,
    "speed": 5.0,
}


def _direction_from(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    return np.array([
        math.cos(e) * math.cos(a),
        math.sin(e),
        math.cos(e) * math.sin(a),
    ], dtype=np.float32)


@engine.on("startup")
def setup(_payload):
    # A 5x5 grid of cubes so motion is visible.
    for ix in range(-2, 3):
        for iz in range(-2, 3):
            engine.spawn(
                Mesh(cube_mesh),
                Material(cube_material),
                Transform(pos=(ix * 2.0, 0.0, iz * 2.0)),
                n=1,
            )
    engine.camera.position = np.array([0.0, 1.0, 5.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 1.0, 0.0], dtype=np.float32)


@engine.system
def fly(query: Query[Transform], dt: float):
    # Rotate when right button (2) is held; mouse_delta is this-frame only.
    if engine.input.is_mouse_pressed(2):
        dx, dy = engine.input.mouse_delta
        _fly["azimuth"] += dx * 0.2
        _fly["elevation"] = max(-89.0, min(89.0, _fly["elevation"] - dy * 0.2))

    forward = _direction_from(_fly["azimuth"], _fly["elevation"])
    # Right vector = world-up × forward, then normalize.
    right = np.cross(np.array([0, 1, 0], np.float32), forward)
    n = np.linalg.norm(right)
    if n > 0:
        right = right / n

    move = np.zeros(3, np.float32)
    if engine.input.is_pressed("w"): move += forward
    if engine.input.is_pressed("s"): move -= forward
    if engine.input.is_pressed("d"): move += right
    if engine.input.is_pressed("a"): move -= right
    if engine.input.is_pressed("Space"): move[1] += 1
    if engine.input.is_pressed("Shift"): move[1] -= 1

    nm = np.linalg.norm(move)
    if nm > 0:
        engine.camera.position = engine.camera.position + (move / nm) * _fly["speed"] * dt

    # Always look in the current forward direction.
    engine.camera.target = engine.camera.position + forward


if __name__ == "__main__":
    engine.cli()
