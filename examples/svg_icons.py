"""SVG icons in the GUI and an SVG texture on a cube.

The same SVG is drawn at 24, 48, 96 and 192 UI pixels, plain and tinted; every
two seconds the GUI scale switches between 1 and 1.5, and each icon is
rasterized again at its new pixel size, so it stays crisp. The cube wears an
SVG sign as its albedo map; the sign's transparent corners are cut out.

Needs the `svg` extra: pip install 'manifold-gfx[svg]'.

Run interactively:
    uv run python examples/svg_icons.py

Headless stills (scale 1, then 1.5 as <out>-scaled.png):
    uv run python examples/svg_icons.py --still /tmp/svg_icons.png
"""

import sys
from pathlib import Path

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gui import Image, Panel, Text
from manifoldx.resources import PointLight, StandardMaterial
from manifoldx.textures import texture_from_svg

STAR = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffe066"/><stop offset="1" stop-color="#f08c00"/>
  </linearGradient></defs>
  <path d="M32 4 L40 23 L60 24 L44 37 L50 58 L32 46 L14 58 L20 37 L4 24 L24 23 Z"
        fill="url(#g)" stroke="#7a3d00" stroke-width="3" stroke-linejoin="round"/>
</svg>"""

# White art: the tint gives it its colour.
HEART = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <path d="M32 56 C8 40 4 26 12 16 C20 7 30 11 32 19 C34 11 44 7 52 16 C60 26 56 40 32 56 Z"
        fill="#ffffff"/>
</svg>"""

SIGN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="8" y="8" width="240" height="240" rx="48" fill="#1d6fb8"/>
  <circle cx="128" cy="104" r="56" fill="#ffd43b"/>
  <text x="128" y="214" text-anchor="middle" font-family="DejaVuSansMono"
        font-size="34" fill="#ffffff">SVG</text>
</svg>"""

SIZES = (24, 48, 96, 192)

engine = mx.Engine("SVG icons", width=960, height=600)
engine.add_light(PointLight(position=(3, 4, 5), color="#ffffff", intensity=40.0))

# A nested panel needs a height: its own size is zero.
row = {"direction": "h", "gap": 12, "padding": 0, "bg": "#00000000", "height": max(SIZES)}
panel = Panel(
    [
        Text("24, 48, 96 and 192 px"),
        Panel([Image(STAR, style_overrides={"width": s}) for s in SIZES], style_overrides=row),
        Panel(
            [
                Image(HEART, style_overrides={"width": s, "tint": c})
                for s, c in zip(SIZES, ("#ff6b6b", "#51cf66", "#339af0", "#fcc419"))
            ],
            style_overrides=row,
        ),
    ],
    offset=(16, 16),
    style_overrides={"width": 420, "padding": 12, "gap": 10, "bg": "#1a1a1ae0", "radius": 8},
)
engine.gui.append(panel)


def uv_cube(size: float) -> dict:
    """A cube with one full 0..1 UV square per face."""
    h = size / 2
    faces = [  # (normal, u axis, v axis)
        ((0, 0, 1), (1, 0, 0), (0, -1, 0)), ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
        ((1, 0, 0), (0, 0, -1), (0, -1, 0)), ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    ]
    pos, nrm, uvs, idx = [], [], [], []
    for n, u, v in faces:
        n, u, v = map(np.array, (n, u, v))
        base = len(pos)
        for su, sv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            pos.append(h * (n + (2 * su - 1) * u + (2 * sv - 1) * v))
            nrm.append(n)
            uvs.append((su, sv))
        idx += [base, base + 2, base + 1, base, base + 3, base + 2]
    return {
        "positions": np.asarray(pos, np.float32),
        "normals": np.asarray(nrm, np.float32),
        "uvs": np.asarray(uvs, np.float32),
        "indices": np.asarray(idx, np.uint32),
    }


@engine.on("startup")
def setup(_payload):
    sign = texture_from_svg(engine, SIGN, 512)
    engine.spawn(
        Mesh(uv_cube(1.4)),
        Material(StandardMaterial(color="#ffffff", roughness=0.5, albedo_map=sign)),
        Transform(pos=(1.3, -0.2, 0)),
    )
    engine.camera.set_pose(position=(0, 0.8, 4.5), target=(-0.4, 0, 0))


clock = [0.0]


@engine.system
def spin_and_rescale(query, dt: float):
    engine.camera.orbit(12 * dt, 0)
    clock[0] += dt
    engine.gui.scale = 1.5 if int(clock[0] / 2) % 2 else 1.0


def stills(out: Path) -> None:
    from PIL import Image as PILImage

    PILImage.fromarray(engine.render_frame(dt=0.0)).save(out)
    clock[0] = 2.0  # the system flips to scale 1.5 on the next frame
    PILImage.fromarray(engine.render_frame(dt=0.0)).save(out.with_name(out.stem + "-scaled.png"))


if __name__ == "__main__":
    if "--still" in sys.argv:
        stills(Path(sys.argv[sys.argv.index("--still") + 1]))
    else:
        engine.cli(fps=30, duration=6, output="svg_icons.mp4")
