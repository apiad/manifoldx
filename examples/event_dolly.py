"""Event-driven camera dolly: zoom in, hold, zoom out, hold — in a single
async while-True loop using `await engine.delay(...)` between phases."""

import math
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query


engine = mx.Engine("Event Dolly")

cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.RED)


@engine.on("startup")
def setup(_payload):
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(0, 0, 0)),
        n=1,
    )
    engine.emit("dolly")


@engine.on("dolly")
async def dolly(_payload):
    """Loop forever: dolly camera between (0, 1, 4) and (0, 1, 1.5)."""
    far = np.array([0, 1, 4], dtype=np.float32)
    near = np.array([0, 1, 1.5], dtype=np.float32)
    while True:
        # Zoom in over 2 seconds.
        t0 = engine.elapsed
        while engine.elapsed - t0 < 2.0:
            t = (engine.elapsed - t0) / 2.0
            engine.camera.position = (1 - t) * far + t * near
            await engine.tick()

        # Hold for 1 second.
        await engine.delay(1.0)

        # Zoom out over 2 seconds.
        t0 = engine.elapsed
        while engine.elapsed - t0 < 2.0:
            t = (engine.elapsed - t0) / 2.0
            engine.camera.position = (1 - t) * near + t * far
            await engine.tick()

        # Hold for 1 second.
        await engine.delay(1.0)


@engine.system
def rotate(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=0, y=dt * math.pi * 0.5, z=0)


if __name__ == "__main__":
    engine.cli()
