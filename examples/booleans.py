"""CSG gallery — union, difference, intersection, lit + animated + shadowed.

Three boolean solids orbit a floor under a shadow-casting sun, each spinning:
  1. rounded die  = box ∩ sphere        (intersection),
  2. drilled ball = sphere − square bar (difference),
  3. bond         = sphere ∪ sphere     (union).

All three CSG results are computed once at startup (BSP-tree booleans).

    uv run python examples/modeling_boolean.py
    uv run python examples/modeling_boolean.py --render --duration 8 --output /tmp/boolean.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial, plane
from manifoldx.systems import Query

engine = mx.Engine("Modeling — Booleans", width=1024, height=768)

engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.2, direction=(-0.8, -0.7, -0.4)))
engine.enable_shadows(resolution=2048, bias=0.004, pcf_radius=2)

# --- Floor ---------------------------------------------------------------
FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)
engine.spawn(
    Mesh(plane(13, 13)),
    Material(StandardMaterial(color="#c4c4c4", roughness=0.95)),
    Transform(pos=(0, 0, 0), rot=FLOOR_ROT),
)

_SCALE = (1.4, 1.4, 1.4)

# --- 1. Rounded die: box ∩ sphere (intersection) -------------------------
die = GeoMesh.box(1.7, 1.7, 1.7).intersection(GeoMesh.icosphere(subdivisions=2, radius=1.12))
engine.spawn(
    Mesh(die.to_geometry()),
    Material(StandardMaterial(color="#e8b23a", roughness=0.25, metallic=0.85)),  # gold
    Transform(pos=(3.0, 2.0, 0), scale=_SCALE),
)

# --- 2. Drilled ball: sphere − square bar (difference) -------------------
drilled = GeoMesh.icosphere(subdivisions=2, radius=1.0).difference(GeoMesh.box(0.7, 0.7, 3.0))
engine.spawn(
    Mesh(drilled.to_geometry()),
    Material(StandardMaterial(color="#2a9d8f", roughness=0.35, metallic=0.1)),  # teal
    Transform(pos=(-3.0, 2.0, 0), scale=_SCALE),
)

# --- 3. Bond: sphere ∪ sphere (union) ------------------------------------
bond = GeoMesh.icosphere(subdivisions=2, radius=0.85).union(
    GeoMesh.icosphere(subdivisions=2, radius=0.85).translate((1.1, 0, 0))
)
engine.spawn(
    Mesh(bond.to_geometry()),
    Material(StandardMaterial(color="#d4663a", roughness=0.4, metallic=0.15)),  # copper
    Transform(pos=(0, 2.0, 3.0), scale=_SCALE),
)

# Orbit params per solid (entities 1..3): (radius, base_y, bob_amp, bob_speed, ang_speed, phase)
_SOLIDS = [
    (3.2, 2.0, 0.5, 1.1, 0.5, 0.0),
    (3.2, 2.0, 0.6, 1.4, 0.5, 2.094),
    (3.2, 2.0, 0.4, 0.9, 0.5, 4.189),
]
_FIRST = 1


@engine.system
def animate(query: Query[Transform], dt: float):
    t = engine.elapsed
    pos = query[Transform].pos.data.copy()
    for i, (r, by, ba, bs, asp, ph) in enumerate(_SOLIDS):
        row = _FIRST + i
        pos[row, 0] = r * np.cos(t * asp + ph)
        pos[row, 1] = by + ba * np.sin(t * bs + ph)
        pos[row, 2] = r * np.sin(t * asp + ph)
    query[Transform].pos = pos
    query[Transform].rot += Transform.rotation(x=dt * 0.2, y=dt * 0.6, z=0)


engine.camera.fit(radius=8.0, center=(0, 1.7, 0), azimuth=35, elevation=44)


@engine.system
def camera_orbit(query: Query[Transform], dt: float):
    engine.camera.orbit(6 * dt, 0)


if __name__ == "__main__":
    engine.cli()
