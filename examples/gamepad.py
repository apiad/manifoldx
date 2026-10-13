"""Gamepad input via `engine.input` polling and the `gamepad_*` events.

The d-pad or the left stick moves the cube (the virtual "up"/"down"/
"left"/"right" buttons); A makes it hop. The arrow keys and Space do the
same, so the example also works without a pad. A GUI panel shows the
pressed gamepad buttons and the stick values.

Each pad is printed when it connects (index, name, GUID, and whether GLFW
has an SDL mapping for it). A pad with no mapping falls back to a guessed
raw layout; set MANIFOLDX_GAMEPAD_MAPPINGS to an SDL mapping string or a
gamecontrollerdb.txt path to fix that.
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.gui import Panel, Text
from manifoldx.input import GAMEPAD_BUTTONS, GAMEPAD_DIRECTIONS
from manifoldx.systems import Query

engine = mx.Engine("Gamepad")
engine.camera.position = np.array([0.0, 6.0, 10.0], dtype=np.float32)
engine.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

SPEED = 4.0  # world units per second
HOP_SPEED = 6.0
GRAVITY = 18.0
BOUND = 4.0

cube = {"x": 0.0, "y": 0.0, "z": 0.0, "vy": 0.0}

buttons_text = Text("buttons: -")
sticks_text = Text("sticks: -")
engine.gui.append(Panel(children=[buttons_text, sticks_text], anchor="top-left", offset=(20, 20)))


@engine.on("startup")
def setup(_payload):
    engine.spawn(
        Mesh(mx.geometry.cube(1, 1, 1)),
        Material(mx.material.phong(mx.colors.CYAN)),
        Transform(pos=(0, 0, 0)),
        n=1,
    )


@engine.on("frame")
def report_no_pad(payload):
    # The first poll runs before frame 0's handlers, so this is the
    # connected set at launch.
    if payload["frame"] == 0 and not engine.input.gamepads:
        print("no gamepad found")


@engine.on("gamepad_connected")
def on_connected(info: mx.GamepadInfo):
    print(f"gamepad {info.index}: {info.name!r} guid={info.guid} mapped={info.mapped}")


@engine.on("gamepad_disconnected")
def on_disconnected(info: mx.GamepadInfo):
    print(f"gamepad {info.index} disconnected")


def held(direction: str, key: str) -> bool:
    return engine.input.gamepad_pressed(direction) or engine.input.is_pressed(key)


@engine.system
def move(query: Query[Transform], dt: float):
    inp = engine.input
    dx = held("right", "ArrowRight") - held("left", "ArrowLeft")
    dz = held("down", "ArrowDown") - held("up", "ArrowUp")
    cube["x"] = max(-BOUND, min(BOUND, cube["x"] + dx * SPEED * dt))
    cube["z"] = max(-BOUND, min(BOUND, cube["z"] + dz * SPEED * dt))

    on_ground = cube["y"] <= 0.0
    if on_ground and (inp.gamepad_just_pressed("a") or inp.just_pressed(" ")):
        cube["vy"] = HOP_SPEED
    cube["vy"] -= GRAVITY * dt
    cube["y"] = max(0.0, cube["y"] + cube["vy"] * dt)
    if cube["y"] == 0.0:
        cube["vy"] = 0.0  # landed
    query[Transform].pos = (cube["x"], cube["y"], cube["z"])

    pressed = [
        b for b in GAMEPAD_BUTTONS + GAMEPAD_DIRECTIONS
        if inp.gamepad_pressed(b)
    ]
    buttons_text.text = "buttons: " + (" ".join(pressed) or "-")
    lx, ly = inp.gamepad_axis("left_x"), inp.gamepad_axis("left_y")
    rx, ry = inp.gamepad_axis("right_x"), inp.gamepad_axis("right_y")
    sticks_text.text = f"sticks: L({lx:+.2f}, {ly:+.2f}) R({rx:+.2f}, {ry:+.2f})"


if __name__ == "__main__":
    engine.cli()
