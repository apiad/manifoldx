"""Surface patterns: grass, cobbles, rock veins, brick, blocks, planks and roof tiles,
drawn in the fragment shader from world position round each vertex colour.

A geometry's "surfaces" key gives each vertex a pattern id (0 flat, 1 grass, 2 cobbles,
3 veins, 4 brick, 5 blocks, 6 planks, 7 tiles); a StandardMaterial with vertex_colors
draws it. Far surfaces fade to their flat colour, so nothing shimmers.

Run interactively:
    uv run python examples/surface_patterns.py

Headless still (needs Pillow, from the viz extra):
    uv run python examples/surface_patterns.py --still /tmp/surface_patterns.png
"""

import sys

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, material_rgb

# (corner, edge u, edge v, normal, colour, surface id)
QUADS = [
    ((-10, 0, -10), (20, 0, 0), (0, 0, 20), (0, 1, 0), "#7fb653", 1),
    ((-2, 0.01, -10), (4, 0, 0), (0, 0, 20), (0, 1, 0), "#c9b89a", 2),
]
for i, (color, sid) in enumerate([("#f4efe2", 4), ("#bfb6a8", 5), ("#9a958c", 3),
                                  ("#a8774a", 6), ("#d8473a", 7)]):
    QUADS.append(((-7.5 + 3 * i, 0, -4), (2.6, 0, 0), (0, 2.6, 0), (0, 0, 1), color, sid))


def geometry():
    pos, nrm, col, sid, idx = [], [], [], [], []
    for corner, du, dv, n, color, s in QUADS:
        c, du, dv = (np.asarray(v, np.float32) for v in (corner, du, dv))
        base = len(pos)
        pos += [c, c + du, c + du + dv, c + dv]
        nrm += [n] * 4
        col += [material_rgb(color)] * 4
        sid += [s] * 4
        idx += [base, base + 2, base + 1, base, base + 3, base + 2,   # both faces
                base, base + 1, base + 2, base, base + 2, base + 3]
    return {"positions": np.array(pos, np.float32), "normals": np.array(nrm, np.float32),
            "colors": np.array(col, np.float32), "surfaces": np.array(sid, np.int32),
            "indices": np.array(idx, np.uint32)}


engine = mx.Engine("Surface patterns", width=1280, height=720)
engine.background_color = (0.45, 0.65, 0.9)  # a sky, as linear light
engine.set_sun(DirectionalLight(color="#fff3dc", intensity=4.5, direction=(-0.45, -1.0, -0.55)))
engine.spawn(Mesh(geometry()), Material(StandardMaterial(color="#ffffff", vertex_colors=True,
                                                         roughness=0.9)),
             Transform(pos=(0, 0, 0)))


@engine.on("startup")
def aim(_payload):
    engine.camera.set_pose(position=(0, 3.2, 7), target=(0, 1.0, -3))


if __name__ == "__main__":
    if "--still" in sys.argv:
        i = sys.argv.index("--still")
        if i + 1 >= len(sys.argv):
            sys.exit("usage: surface_patterns.py --still OUT.png")
        from PIL import Image

        Image.fromarray(engine.render_frame(dt=0.0)).save(sys.argv[i + 1])
    else:
        engine.cli(fps=30, duration=6, output="surface_patterns.mp4")
