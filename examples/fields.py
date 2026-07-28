"""Composable fields — a developer-built terrain, lit + shadowed.

There is no terrain generator here: the terrain is assembled by the developer
from field primitives — ridged mountains + fbm roughness, domain-warped into
meandering ridges, masked by a radial island falloff — then used to displace a
plane. Monochrome clay render (per-vertex colour arrives in the next sub-project).

    uv run python examples/modeling_fields.py
    uv run python examples/modeling_fields.py --render --duration 8 --output /tmp/fields.mp4
"""

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields
from manifoldx.resources import DirectionalLight, StandardMaterial
from manifoldx.systems import Query

engine = mx.Engine("Modeling — Fields", width=1024, height=768)

engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.4, direction=(-0.8, -0.55, -0.3)))
engine.enable_shadows(resolution=2048, bias=0.004, pcf_radius=2)

# --- The developer composes their own terrain field ----------------------
relief = (
    fields.ridged(seed=5, freq=0.22) * 0.8          # big mountain ridges
    + fields.fbm(seed=7, freq=0.8) * 0.2            # fine roughness
).warp(                                             # domain warp → meandering ridges
    1.4,
    fx=fields.fbm(seed=2, freq=0.25),
    fz=fields.fbm(seed=9, freq=0.25),
)

island = fields.distance().remap(0.0, 8.0, 1.0, 0.0).clamp(0.0, 1.0)  # 1 at centre → 0 at edge
terrain = (relief * island).power(1.3)             # sharpen peaks, flatten shore

land = GeoMesh.plane(width=16, depth=16, segments=150).displace(terrain, amount=4.4)

engine.spawn(
    Mesh(land.to_geometry()),
    Material(StandardMaterial(color="#b8a58c", roughness=0.9)),  # clay
    Transform(pos=(0, 0, 0)),
)

engine.camera.fit(radius=10.5, center=(0, 1.4, 0), azimuth=35, elevation=27)


@engine.system
def camera_orbit(query: Query[Transform], dt: float):
    engine.camera.orbit(6 * dt, 0)


if __name__ == "__main__":
    engine.cli()
