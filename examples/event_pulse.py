"""Sync handler reacting to a periodic event: spawn a new cube every second."""

import math

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query


engine = mx.Engine("Event Pulse")

cube_mesh = mx.geometry.cube(0.4, 0.4, 0.4)
cube_material = mx.material.phong(mx.colors.GREEN)

_state = {"next_pulse_at": 1.0, "count": 0}


@engine.on("frame")
def emit_pulses(payload):
    if payload["elapsed"] >= _state["next_pulse_at"]:
        engine.emit("pulse", {"index": _state["count"]})
        _state["count"] += 1
        _state["next_pulse_at"] += 1.0


@engine.on("pulse")
def on_pulse(payload):
    i = payload["index"]
    angle = i * 0.7
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(math.cos(angle) * 2, 0, math.sin(angle) * 2)),
        n=1,
    )


@engine.system
def rotate(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=0, y=dt * math.pi * 0.5, z=0)


if __name__ == "__main__":
    engine.cli()
