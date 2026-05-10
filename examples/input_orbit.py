"""Event-driven orbit camera: drag with left button to rotate, wheel to zoom.

Demonstrates `PointerEvent.dx/dy` and `WheelEvent.dy` directly off the bus.
The orbit math (azimuth/elevation/distance → camera.position) is local to
this example; it does not (yet) live on the Camera class.
"""

import math
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.input import PointerEvent, WheelEvent


engine = mx.Engine("Input Orbit")

cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.RED)


# Orbit state — module-level so the handlers and recomputation share it.
_orbit = {
    "azimuth": 45.0,    # degrees
    "elevation": 25.0,  # degrees
    "distance": 5.0,
    "target": np.array([0.0, 0.0, 0.0], dtype=np.float32),
}


def _recompute_camera_position() -> None:
    """Map (azimuth, elevation, distance, target) → camera.position."""
    a = math.radians(_orbit["azimuth"])
    e = math.radians(_orbit["elevation"])
    r = _orbit["distance"]
    target = _orbit["target"]
    pos = target + np.array([
        r * math.cos(e) * math.cos(a),
        r * math.sin(e),
        r * math.cos(e) * math.sin(a),
    ], dtype=np.float32)
    engine.camera.position = pos
    engine.camera.target = target


@engine.on("startup")
def setup(_payload):
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(0, 0, 0)),
        n=1,
    )
    _recompute_camera_position()


@engine.on("pointer_move")
def orbit(ev: PointerEvent):
    # Rotate when left button (1) is held during the move.
    if 1 in ev.buttons:
        _orbit["azimuth"] += ev.dx * 0.5
        _orbit["elevation"] = max(-89.0, min(89.0, _orbit["elevation"] + ev.dy * 0.5))
        _recompute_camera_position()


@engine.on("wheel")
def zoom(ev: WheelEvent):
    # rendercanvas wheel dy is ~100 per notch; scale to a 5 % step per notch.
    _orbit["distance"] = max(0.5, _orbit["distance"] * (1.0 - ev.dy * 0.0005))
    _recompute_camera_position()


if __name__ == "__main__":
    engine.cli()
