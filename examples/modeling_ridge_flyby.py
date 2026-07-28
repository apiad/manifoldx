"""Infinite ridge flyby — real-time procedural mountains with distance fog.

A fixed camera looks down a terrain grid. Every frame the terrain field is
sampled through a window that scrolls forward (z + travel), so an endless ridge
system flows toward the viewer — the "infinite" terrain is just the field
evaluated over a moving window, re-poked into the live vertex buffer (FFD-style,
vertex count is constant). Per-vertex colour = a height gradient mixed toward the
horizon fog colour by camera distance, so distant peaks dissolve into the sky.

Built entirely from the composable primitives (fields + Gradient + color_by);
no terrain generator, no fog shader — the fog is composed on the CPU each frame.

    uv run python examples/modeling_ridge_flyby.py
    uv run python examples/modeling_ridge_flyby.py --render --duration 10 --output /tmp/flyby.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields, Gradient
from manifoldx.resources import DirectionalLight, StandardMaterial
from manifoldx.systems import Query

engine = mx.Engine("Infinite Ridge Flyby", width=1024, height=768)

FOG = (0.72, 0.78, 0.86)                 # hazy horizon; the terrain melts into it
engine.background_color = FOG
engine.set_sun(DirectionalLight(color="#fff0dc", intensity=3.1, direction=(-0.5, -0.55, -0.65)))

# --- Terrain grid (local XZ; z runs 0..D ahead of the camera) ------------
W, D, SEG = 44, 52, 160
plane = GeoMesh.plane(width=W, depth=D, segments=SEG)
BASE = plane.positions.copy()
BASE[:, 2] += D / 2.0                     # shift z into [0, D]
FACES = plane.faces
N = len(BASE)
AMOUNT, SPEED = 8.0, 6.0

# --- The developer composes the terrain + palette ------------------------
terrain = (
    fields.ridged(seed=5, freq=0.12) * 0.85
    + fields.fbm(seed=7, freq=0.5) * 0.15
).warp(1.3, fx=fields.fbm(seed=2, freq=0.15), fz=fields.fbm(seed=9, freq=0.15))

palette = Gradient([
    (0.00, "#3a5f8a"), (0.06, "#c2b280"), (0.16, "#4a7a3a"),
    (0.45, "#5f6e38"), (0.65, "#6e5a44"), (0.82, "#8a8078"), (0.96, "#ffffff"),
])
FOG_COL = np.array(FOG, dtype=np.float32)
FOG_START, FOG_END = 20.0, 50.0

# --- Spawn once with a vcolor geometry; push new verts each frame --------
geo0 = GeoMesh(positions=BASE.astype(np.float32), faces=FACES).recompute_normals().to_geometry()
geo0["colors"] = np.tile(FOG_COL, (N, 1))  # presence of colours selects the vcolor interleave

land = engine.spawn(
    Mesh(geo0),
    Material(StandardMaterial(color="#ffffff", roughness=0.92, vertex_colors=True)),
    Transform(pos=(0, 0, 0)),
)

# --- Fixed forward-looking flyby camera ----------------------------------
CAM = np.array([0.0, 6.5, -5.0], dtype=np.float32)
engine.camera.position = CAM.copy()
engine.camera.target = np.array([0.0, 2.5, 22.0], dtype=np.float32)


@engine.system
def flyby(query: Query[Transform], dt: float):
    travel = SPEED * engine.elapsed
    sample = np.stack([BASE[:, 0], np.zeros(N), BASE[:, 2] + travel], axis=1)
    height = np.clip(terrain(sample), 0.0, 1.0)

    pos = BASE.copy()
    pos[:, 1] = height * AMOUNT
    m = GeoMesh(positions=pos.astype(np.float32), faces=FACES).recompute_normals()

    color = palette(height)
    dist = np.linalg.norm(pos - CAM[None, :], axis=1)
    fog = np.clip((dist - FOG_START) / (FOG_END - FOG_START), 0.0, 1.0)[:, None]
    color = color * (1.0 - fog) + FOG_COL[None, :] * fog

    land.set_geometry(m.with_colors(color))          # in-place update, no reach-ins


if __name__ == "__main__":
    engine.cli()
