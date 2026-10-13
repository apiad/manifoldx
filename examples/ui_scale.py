"""GUI scale: draw the whole UI larger or smaller without restyling it.

Widget sizes, paddings, radii and font sizes are UI pixels; `engine.gui.scale`
multiplies them on screen. Press `+` / `-` to change it live. Setting
`engine.gui.design_height` instead scales the UI with the window height.

    uv run python examples/ui_scale.py --scale 2
"""

import argparse
import sys

import manifoldx as mx
from manifoldx.gui import Button, Panel, Text, ValueDisplay

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--scale", type=float, default=1.0)
args, sys.argv[1:] = parser.parse_known_args()  # leave the rest to engine.cli()

engine = mx.Engine("GUI scale")
engine.gui.scale = args.scale

engine.gui.append(
    Panel(
        children=[
            Text("+ / - to scale the UI"),
            ValueDisplay(getter=lambda: f"scale = {engine.gui.scale:.2f}", min_width=120),
            Button(name="reset", label="Reset to 1.0"),
        ],
        offset=(20, 20),
        style_overrides={"width": 220, "height": 72, "padding": 8, "gap": 6, "bg": "#1a1a1aE0", "radius": 4},
    )
)


@engine.on("key_down")
def on_key(ev):
    if ev.key in ("+", "="):
        engine.gui.scale = min(engine.gui.scale * 1.25, 8.0)
    elif ev.key in ("-", "_"):
        engine.gui.scale = max(engine.gui.scale / 1.25, 0.25)


@engine.on("gui:button:reset:click")
def on_reset(payload):
    engine.gui.scale = 1.0


if __name__ == "__main__":
    engine.cli()
