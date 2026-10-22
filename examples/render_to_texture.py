"""A live portrait: a second camera rendered into a texture, shown round in the GUI.

A head stands on a stage far off at x=1000, out of the main camera's view. A
second camera looks at it and renders into a texture every frame; an Image
shows that texture in a circle in the bottom-left corner, beside a speech
bubble whose line is Text in a font file.

Needs the `svg` extra for the bubble: pip install 'manifold-gfx[svg]'.

Run interactively:
    uv run python examples/render_to_texture.py

Headless still:
    uv run python examples/render_to_texture.py --still /tmp/portrait.png
"""

import math
import sys
from importlib.resources import files

import manifoldx as mx
from manifoldx.camera import Camera
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gui import Image, Panel, Text
from manifoldx.resources import PhongMaterial, PointLight, sphere

FONT = str(files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf"))
STAGE = (1000.0, 0.0, 0.0)

BUBBLE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 140">
  <path d="M40 10 H500 A14 14 0 0 1 514 24 V116 A14 14 0 0 1 500 130 H40
           A14 14 0 0 1 26 116 V84 L4 70 L26 56 V24 A14 14 0 0 1 40 10 Z"
        fill="#ffffff" stroke="#2b2b2b" stroke-width="5" stroke-linejoin="round"/>
</svg>"""

engine = mx.Engine("Live portrait", width=960, height=600)
engine.gui.design_height = 600

# The main scene: a ring of balls around the origin.
for i in range(12):
    a = i / 12 * 2 * math.pi
    engine.spawn(Mesh(sphere(0.4, 16)), Material(PhongMaterial("#4dabf7")),
                 Transform(pos=(3 * math.cos(a), 0, 3 * math.sin(a))))
engine.add_light(PointLight(position=(0, 5, 5), color="#ffffff", intensity=40.0))

# The portrait stage: a head with two eyes, and its own light.
x, y, z = STAGE
head = engine.spawn(Mesh(sphere(1.0, 32)), Material(PhongMaterial("#ffd43b")), Transform(pos=STAGE))
eyes = [engine.spawn(Mesh(sphere(0.16, 12)), Material(PhongMaterial("#222222")),
                     Transform(pos=(x + dx, y + 0.25, z + 0.97))) for dx in (-0.35, 0.35)]
engine.add_light(PointLight(position=(x + 1, y + 2, z + 4), color="#ffffff", intensity=30.0))

portrait_camera = Camera()
portrait_camera.set_pose(position=(x, y, z + 3.2), target=STAGE)

clock = [0.0]


@engine.on("startup")
def setup(_payload):
    engine.camera.set_pose(position=(0, 4, 9), target=(0, 0, 0))
    portrait = engine.render_to_texture(portrait_camera, (400, 400),
                                        background=(0.55, 0.8, 0.95, 1.0))
    engine.gui.append(Panel([Image(portrait, style_overrides={"width": 200, "radius": 100})],
                            offset=(20, 380),
                            style_overrides={"padding": 0, "bg": "#00000000",
                                             "width": 200, "height": 200}))
    engine.gui.append(Panel([Image(BUBBLE, style_overrides={"width": 520})],
                            offset=(235, 440),
                            style_overrides={"padding": 0, "bg": "#00000000",
                                             "width": 520, "height": 140}))
    engine.gui.append(Panel([Text("¡Hola! ¿Vamos al bosque?",
                                  style_overrides={"font": FONT, "font_size": 30,
                                                   "fg": "#000000"})],
                            offset=(275, 490),
                            style_overrides={"padding": 0, "bg": "#00000000",
                                             "width": 460, "height": 40}))


@engine.system
def bob(query, dt: float):
    """The head bobs and blinks, so the portrait is visibly live."""
    clock[0] += dt
    t = clock[0]
    bob_y = 0.08 * math.sin(t * 6)
    engine.camera.orbit(10 * dt, 0)
    head.transform.pos = (x, y + bob_y, z)
    blink = 0.1 if (t % 3) > 2.85 else 1.0
    for e, dx in zip(eyes, (-0.35, 0.35)):
        e.transform.pos = (x + dx, y + 0.25 + bob_y, z + 0.97)
        e.transform.scale = (1.0, blink, 1.0)


if __name__ == "__main__":
    if "--still" in sys.argv:
        from PIL import Image as PILImage

        out = sys.argv[sys.argv.index("--still") + 1]
        PILImage.fromarray(engine.render_frame(dt=0.0)).save(out)
    else:
        engine.cli(fps=30, duration=6, output="render_to_texture.mp4")
