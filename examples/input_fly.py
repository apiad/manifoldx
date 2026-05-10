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

# One cube mesh, scaled per-entity via Transform — keeps the renderer to two
# draw batches (buildings + ground) regardless of skyline size.
cube_mesh = mx.geometry.cube(1, 1, 1)
building_mat = mx.material.phong(mx.colors.CYAN)
ground_mat = mx.material.phong(mx.colors.WHITE)


# City layout knobs.
_CITY = {
    "blocks": 12,        # blocks per side → blocks × blocks footprint
    "block_pitch": 4.0,  # world units between block centers
    "footprint": 2.4,    # building width/depth (street width = pitch - footprint)
    "h_min": 1.5,
    "h_max": 14.0,
}


def _height_at(ix: int, iz: int) -> float:
    """Deterministic skyline. Mixes two trig waves with a cheap hash so
    neighbours differ without any noise library."""
    wave = math.sin(ix * 0.7) * math.cos(iz * 0.9) + math.sin((ix + iz) * 0.3)
    h_pseudo = ((ix * 73856093) ^ (iz * 19349663)) & 0xFFFF
    jitter = (h_pseudo / 0xFFFF) * 2.0 - 1.0  # in [-1, 1]
    t = (wave + jitter) * 0.25 + 0.5  # roughly in [0, 1]
    t = max(0.0, min(1.0, t))
    return _CITY["h_min"] + t * (_CITY["h_max"] - _CITY["h_min"])


# Fly state mirrors the orbit example pattern.
_fly = {
    "azimuth": 45.0,
    "elevation": -10.0,
    "speed": 8.0,
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
    n = _CITY["blocks"]
    pitch = _CITY["block_pitch"]
    fp = _CITY["footprint"]
    half = (n - 1) * pitch * 0.5

    # Ground plane — one wide, very thin cube under the city.
    ground_extent = n * pitch
    engine.spawn(
        Mesh(cube_mesh),
        Material(ground_mat),
        Transform(pos=(0.0, -0.05, 0.0), scale=(ground_extent, 0.1, ground_extent)),
        n=1,
    )

    # Buildings: one cube per block, scaled to (footprint, height, footprint).
    for ix in range(n):
        for iz in range(n):
            h = _height_at(ix, iz)
            x = ix * pitch - half
            z = iz * pitch - half
            engine.spawn(
                Mesh(cube_mesh),
                Material(building_mat),
                Transform(pos=(x, h * 0.5, z), scale=(fp, h, fp)),
                n=1,
            )

    # Camera: elevated, off-corner, looking in toward the city centre so
    # the skyline reads even before the user moves.
    engine.camera.position = np.array(
        [-half - 6.0, _CITY["h_max"] * 0.9, -half - 6.0], dtype=np.float32
    )
    engine.camera.target = np.array([0.0, _CITY["h_max"] * 0.4, 0.0], dtype=np.float32)
    # Sync _fly's azimuth/elevation with the initial camera so the first
    # WASD frame doesn't snap to a different look direction.
    look = engine.camera.target - engine.camera.position
    look_h = math.sqrt(look[0] ** 2 + look[2] ** 2)
    _fly["azimuth"] = math.degrees(math.atan2(look[2], look[0]))
    _fly["elevation"] = math.degrees(math.atan2(look[1], look_h))


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
