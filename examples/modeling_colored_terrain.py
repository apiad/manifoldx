"""Color-by-field — the composed terrain, shaded by height, lit + shadowed.

Same developer-composed terrain field as examples/modeling_fields.py, now
coloured: a height Field mapped through a developer-defined Gradient
(water → sand → grass → rock → scree → snow) into per-vertex colour, rendered
with a vertex-colour StandardMaterial. No opinionated biome logic — the
developer supplies the field and the gradient.

    uv run python examples/modeling_colored_terrain.py
    uv run python examples/modeling_colored_terrain.py --render --duration 8 --output /tmp/terrain.mp4
"""

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields, Gradient
from manifoldx.resources import DirectionalLight, StandardMaterial
from manifoldx.systems import Query

engine = mx.Engine("Modeling — Colored Terrain", width=1024, height=768)

engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.4, direction=(-0.8, -0.55, -0.3)))
engine.enable_shadows(resolution=2048, bias=0.004, pcf_radius=2)

# --- Compose the terrain field (shape) -----------------------------------
relief = (
    fields.ridged(seed=5, freq=0.22) * 0.8
    + fields.fbm(seed=7, freq=0.8) * 0.2
).warp(1.4, fx=fields.fbm(seed=2, freq=0.25), fz=fields.fbm(seed=9, freq=0.25))
island = fields.distance().remap(0.0, 8.0, 1.0, 0.0).clamp(0.0, 1.0)
terrain = (relief * island).power(1.3)

AMOUNT = 4.4
land = GeoMesh.plane(width=16, depth=16, segments=160).displace(terrain, amount=AMOUNT)

# --- Compose the colour (a height Field through a developer gradient) -----
elevation = fields.coord("y").remap(0.0, AMOUNT, 0.0, 1.0)
palette = Gradient([
    (0.00, "#2f5a8c"),   # water
    (0.05, "#c2b280"),   # sand
    (0.12, "#4a7a3a"),   # grass
    (0.42, "#5f6e38"),   # upland grass
    (0.62, "#6e5a44"),   # rock
    (0.80, "#8a8078"),   # scree
    (0.95, "#ffffff"),   # snow (only the highest peaks)
])
colored = land.color_by(elevation, palette)

engine.spawn(
    Mesh(colored.to_geometry()),
    Material(StandardMaterial(color="#ffffff", roughness=0.9, vertex_colors=True)),
    Transform(pos=(0, 0, 0)),
)

engine.camera.fit(radius=10.5, center=(0, 1.4, 0), azimuth=35, elevation=27)


@engine.system
def camera_orbit(query: Query[Transform], dt: float):
    engine.camera.orbit(6 * dt, 0)


if __name__ == "__main__":
    engine.cli()
