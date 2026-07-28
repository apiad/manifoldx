"""Planet dive — fly from orbit down to a procedural planet's surface.

A signed terrain field displaces an icosphere into continents + ocean basins,
biome-colored; a glossy ocean sphere fills the basins; a fresnel `AtmosphereMaterial`
shell gives the limb halo from space. A scripted descent eases the camera in while
altitude drives the fog + background from space-black to sky-blue. The planet mesh
is generated once in a worker process (no render-thread stall).

Tier-1 slice (CPU). Next: GPU fields for near-surface detail.

    uv run python demos/planet_dive.py
    uv run python demos/planet_dive.py --render --duration 10 --output /tmp/planet.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields, Gradient
from manifoldx.resources import DirectionalLight, StandardMaterial, AtmosphereMaterial

# ---- pure module-level world (imported cleanly by the worker process) --------
R, SEA, PEAK, SUBDIV = 10.0, 1.1, 1.6, 6
ALT_START, ALT_SKIM, DESCENT_FRAMES = 30.0, 3.0, 260   # frame-paced (deterministic under --render)
SPACE = np.array([0.02, 0.03, 0.06])
SKY = np.array([0.55, 0.72, 0.96])

terrain = (
    fields.ridged(seed=4, freq=0.28) * 0.72
    + fields.fbm(seed=8, freq=0.9) * 0.18
).warp(0.6, fx=fields.fbm(2, 0.4), fz=fields.fbm(9, 0.4)).remap(0.0, 1.0, -SEA, PEAK)

biome = fields.distance().remap(R - SEA, R + PEAK, 0.0, 1.0).clamp(0.0, 1.0)
palette = Gradient([(0.00, "#243a5e"), (0.26, "#c2b280"), (0.36, "#3f6f2e"),
                    (0.60, "#5f6e38"), (0.78, "#7a6a52"), (0.90, "#9a9088"), (1.0, "#ffffff")])


def build_planet():
    """Pure, picklable: the biome-colored planet as a baked geometry dict."""
    return (GeoMesh.icosphere(subdivisions=SUBDIV, radius=R)
            .displace(terrain, amount=1.0)
            .color_by(biome, palette)
            .to_geometry())


def main():
    engine = mx.Engine("Planet Dive", width=1024, height=768)
    engine.background_color = tuple(SPACE)
    engine.set_sun(DirectionalLight(color="#fff4e8", intensity=3.2, direction=(-0.6, -0.3, -0.55)))

    planet_geo = engine.submit_process(build_planet).wait()   # generate off-thread, block once
    engine.spawn(Mesh(planet_geo),
                 Material(StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True)),
                 Transform())
    engine.spawn(Mesh(GeoMesh.icosphere(SUBDIV - 1, R).to_geometry()),
                 Material(StandardMaterial("#1b3a6b", roughness=0.12, metallic=0.1)),
                 Transform())
    # atmosphere shell LAST so it composites over the opaque planet + ocean
    engine.spawn(Mesh(GeoMesh.icosphere(4, R * 1.25).to_geometry()),
                 Material(AtmosphereMaterial("#8fb8ff", intensity=1.0)),
                 Transform())

    approach = np.array([0.25, 0.35, 1.0])
    approach /= np.linalg.norm(approach)
    landing = approach * R
    clock = {"frame": 0}

    @engine.system
    def descend(query, dt):
        clock["frame"] += 1
        p = min(clock["frame"] / DESCENT_FRAMES, 1.0)        # frame-paced, wall-clock independent
        pe = p * p * (3 - 2 * p)                              # smoothstep ease
        cam_r = (R + ALT_START) * (1 - pe) + (R + ALT_SKIM) * pe
        engine.camera.set_pose(tuple(approach * cam_r), tuple(landing))

        alt = cam_r - R
        t = float(np.clip((alt - ALT_SKIM) / (ALT_START - ALT_SKIM), 0.0, 1.0))
        sky = SKY * (1 - t) + SPACE * t
        engine.background_color = tuple(sky)
        engine.enable_fog(5.0 + 5000 * t, 60.0 + 1e4 * t, color=tuple(sky))

    engine.cli()


if __name__ == "__main__":
    main()
