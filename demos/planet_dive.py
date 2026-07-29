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
R, SEA, PEAK, SUBDIV = 10.0, 0.65, 0.85, 6              # gentle terrain (peaks ~8.5% of radius)
ALT_START = 32.0                                        # high-orbit start altitude
ORBIT_ALT = 2.0                                         # low-orbit altitude (plane-over-terrain feel)
PITCH = 0.37                                           # down-tilt in low orbit (horizon ~40% from bottom)
DESCENT_FRAMES = 190                                    # descent completes ~here; then orbit
AZ_RATE = 0.010                                         # radians/frame the camera circles
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

    clock = {"frame": 0}

    @engine.system
    def fly(query, dt):
        clock["frame"] += 1
        f = clock["frame"]
        dp = min(f / DESCENT_FRAMES, 1.0)                     # descent progress → holds at 1
        dpe = dp * dp * (3 - 2 * dp)                          # smoothstep ease
        cam_r = (R + ALT_START) * (1 - dpe) + (R + ORBIT_ALT) * dpe
        az = 0.4 + f * AZ_RATE                               # continuously circle the planet (equatorial)

        radial = np.array([np.sin(az), 0.0, np.cos(az)])     # outward (down = -radial)
        forward = np.array([np.cos(az), 0.0, -np.sin(az)])   # orbit tangent (direction of travel)
        cam = cam_r * radial
        # Look at planet centre while high; ease to a forward + down gaze in low orbit.
        orbit_look = forward - PITCH * radial                # forward, tilted down
        orbit_look /= np.linalg.norm(orbit_look)
        target = (1 - dpe) * np.zeros(3) + dpe * (cam + orbit_look * 22.0)
        engine.camera.set_pose(tuple(cam), tuple(target))
        # Bank so the terrain is DOWN (not sideways): up eases world-Y -> radially outward.
        up = (1 - dpe) * np.array([0.0, 1.0, 0.0]) + dpe * radial
        engine.camera.up = (up / np.linalg.norm(up)).astype(np.float32)

        alt = cam_r - R
        t = float(np.clip((alt - ORBIT_ALT) / (ALT_START - ORBIT_ALT), 0.0, 1.0))
        sky = SKY * (1 - t) + SPACE * t
        engine.background_color = tuple(sky)
        engine.enable_fog(6.0 + 5000 * t, 45.0 + 1e4 * t, color=tuple(sky))

    engine.cli()


if __name__ == "__main__":
    main()
