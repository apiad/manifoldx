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
from manifoldx.resources import DirectionalLight, StandardMaterial, AtmosphereMaterial, WaterMaterial
from manifoldx.sky import starfield

# ---- pure module-level world (imported cleanly by the worker process) --------
R, SEA, PEAK, SUBDIV = 10.0, 0.65, 0.85, 6              # gentle terrain (peaks ~8.5% of radius)
ALT_START = 32.0                                        # high-orbit start altitude
ORBIT_ALT = 2.0                                         # low-orbit altitude (plane-over-terrain feel)
PITCH = 0.37                                           # down-tilt in low orbit (horizon ~40% from bottom)
DESCENT_FRAMES = 190                                    # descent completes ~here; then orbit
AZ_RATE = 0.010                                         # radians/frame the camera circles
SUN_RATE = 0.0015                                       # slow sun; the orbit itself carries us into night
SPACE = np.array([0.02, 0.03, 0.06])
SKY = np.array([0.55, 0.72, 0.96])
SUNSET = np.array([0.95, 0.48, 0.32])


def _smoothstep(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)

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

    starfield(engine, count=1800, radius=R * 40.0, seed=11)   # space backdrop
    planet_geo = engine.submit_process(build_planet).wait()   # generate off-thread, block once
    engine.spawn(Mesh(planet_geo),
                 Material(StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True)),
                 Transform())
    engine.spawn(Mesh(GeoMesh.icosphere(SUBDIV - 1, R).to_geometry()),
                 Material(WaterMaterial(color="#0a2540", fresnel_power=4.0)),
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
        # Day/night: sweep the sun; water, terrain, and atmosphere all react to it.
        sun_az = 0.7 + f * SUN_RATE
        sd = np.array([np.cos(sun_az), -0.28, np.sin(sun_az)])
        engine.set_sun(DirectionalLight(color="#fff4e8", intensity=3.2,
                                        direction=tuple(sd / np.linalg.norm(sd))))
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

        # Day/night sky: dark (stars show) on the night side, blue by day, sunset at the terminator.
        local_sun = float(np.dot(radial, -sd))               # sun elevation in the local sky
        day = _smoothstep(-0.15, 0.25, local_sun)            # 0 night, 1 day
        dusk = max(0.0, 1.0 - abs(local_sun) / 0.30)         # peaks at the terminator
        day_sky = SKY * day + SPACE * (1 - day)
        day_sky = day_sky * (1 - 0.55 * dusk) + SUNSET * (0.55 * dusk)

        alt = cam_r - R
        t = float(np.clip((alt - ORBIT_ALT) / (ALT_START - ORBIT_ALT), 0.0, 1.0))
        sky = day_sky * (1 - t) + SPACE * t                  # space (high) is always dark
        engine.background_color = tuple(sky)
        # Haze in the daylit atmosphere; thin + dark at night so terrain fades to night and stars show.
        haze = (1 - t) * (0.25 + 0.75 * day)
        engine.enable_fog(6.0 + 44.0 * (1 - haze), 45.0 + 255.0 * (1 - haze), color=tuple(sky))

    engine.cli()


if __name__ == "__main__":
    main()
