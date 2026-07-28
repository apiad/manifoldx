"""Procedural asteroid — smoke test + showcase for manifoldx.modeling.

Batch 0: a bare icosphere, proving the modeling -> GPU seam.
Batch 1 (Task 10) upgrades this in place into a noise-displaced asteroid.

Render a clip:
    uv run python examples/modeling_asteroid.py --render --duration 3 --fps 30 \
        --output /tmp/asteroid.mp4
"""

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, noise
from manifoldx.resources import PointLight
from manifoldx.systems import Query

engine = mx.Engine("Asteroid")

rock = (
    GeoMesh.icosphere(subdivisions=4)
    .displace(noise.fbm(seed=7, octaves=5), amount=0.35)
    .twist(angle=0.4, axis="y")
    .taper(factor=0.2, axis="y")
)
rock_geometry = rock.to_geometry()
rock_material = mx.material.standard(color="#8c8073", roughness=0.9)  # hex string, per pbr_demo

engine.set_lights([
    PointLight(color="#fff2e0", intensity=6.0, position=(4, 5, 4)),
    PointLight(color="#8090ff", intensity=2.0, position=(-4, -2, -3)),
])


@engine.on("startup")
def create_asteroid(_payload):
    engine.spawn(
        Mesh(rock_geometry),
        Material(rock_material),
        Transform(pos=(0, 0, 0)),
        n=1,
    )


@engine.system
def spin(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=dt * 0.2, y=dt * 0.5, z=0)


if __name__ == "__main__":
    engine.cli()
