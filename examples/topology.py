"""Topology gallery — subdivide, extrude, decimate, lit + animated + shadowed.

Three procedural forms orbit a floor under a shadow-casting sun, each spinning:
  1. a Loop-subdivided, noise-displaced organic blob,
  2. an icosphere with a +y cap extruded into a raised crystal facet,
  3. the Batch-1 asteroid decimated into a faceted low-poly rock.

    uv run python examples/modeling_topology.py
    uv run python examples/modeling_topology.py --render --duration 8 --output /tmp/topology.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, noise
from manifoldx.resources import DirectionalLight, StandardMaterial, plane
from manifoldx.systems import Query

engine = mx.Engine("Modeling — Topology", width=1024, height=768)

engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.2, direction=(-0.8, -0.7, -0.4)))
engine.enable_shadows(resolution=2048, bias=0.004, pcf_radius=2)

# --- Floor ---------------------------------------------------------------
FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)  # lay the +Z quad flat (normal +Y)
engine.spawn(
    Mesh(plane(13, 13)),
    Material(StandardMaterial(color="#c4c4c4", roughness=0.95)),
    Transform(pos=(0, 0, 0), rot=FLOOR_ROT),
)

_SCALE = (1.45, 1.45, 1.45)

# --- Form 1: Loop-subdivided organic blob --------------------------------
blob = (
    GeoMesh.icosphere(subdivisions=1)
    .subdivide(iterations=2, scheme="loop")
    .displace(noise.fbm(seed=3, octaves=4), amount=0.22)
)
engine.spawn(
    Mesh(blob.to_geometry()),
    Material(StandardMaterial(color="#2a9d8f", roughness=0.35, metallic=0.05)),
    Transform(pos=(3.0, 2.0, 0), scale=_SCALE),
)

# --- Form 2: extruded crystal facet on a sphere --------------------------
ico = GeoMesh.icosphere(subdivisions=2)
cap = ico.positions[ico.faces].mean(axis=1)[:, 1] > 0.55   # faces on the +y cap
crystal = ico.extrude(cap, distance=0.5)
engine.spawn(
    Mesh(crystal.to_geometry()),
    Material(StandardMaterial(color="#4a72e8", roughness=0.15, metallic=0.6)),
    Transform(pos=(-3.0, 2.0, 0), scale=_SCALE),
)

# --- Form 3: decimated low-poly asteroid ---------------------------------
lowpoly = (
    GeoMesh.icosphere(subdivisions=4)
    .displace(noise.fbm(seed=7, octaves=5), amount=0.35)
    .decimate(grid=7)
)
engine.spawn(
    Mesh(lowpoly.to_geometry()),
    Material(StandardMaterial(color="#b0857a", roughness=0.85)),
    Transform(pos=(0, 2.0, 3.0), scale=_SCALE),
)

# Orbit params per form (entities 1..3): (radius, base_y, bob_amp, bob_speed, ang_speed, phase)
_FORMS = [
    (3.2, 2.0, 0.5, 1.1, 0.55, 0.0),
    (3.2, 2.0, 0.6, 1.4, 0.55, 2.094),
    (3.2, 2.0, 0.4, 0.9, 0.55, 4.189),
]
_FIRST = 1  # entity 0 is the floor


@engine.system
def animate(query: Query[Transform], dt: float):
    t = engine.elapsed
    pos = query[Transform].pos.data.copy()
    for i, (r, by, ba, bs, asp, ph) in enumerate(_FORMS):
        row = _FIRST + i
        pos[row, 0] = r * np.cos(t * asp + ph)
        pos[row, 1] = by + ba * np.sin(t * bs + ph)
        pos[row, 2] = r * np.sin(t * asp + ph)
    query[Transform].pos = pos
    query[Transform].rot += Transform.rotation(x=dt * 0.15, y=dt * 0.5, z=0)


engine.camera.fit(radius=8.0, center=(0, 1.7, 0), azimuth=35, elevation=44)


@engine.system
def camera_orbit(query: Query[Transform], dt: float):
    engine.camera.orbit(6 * dt, 0)


if __name__ == "__main__":
    engine.cli()
