"""Planet dive — fly from orbit down to a procedural planet's surface.

A signed terrain field displaces an icosphere into continents + ocean basins,
biome-colored; a glossy ocean sphere fills the basins; a physically-based
single-scattering `AtmosphereScatteringMaterial` shell (Rayleigh + Mie) renders
the whole sky — blue day, sunset limb, dark starry night — directly from the sun
angle. A scripted descent spirals the camera from high orbit down to a low
equatorial pass (horizon ~40%, terrain down); the orbit itself carries us through
day into night. The planet mesh is generated once in a worker process (no
render-thread stall).

Tier-1 slice (CPU). Next: GPU fields for near-surface detail.

    uv run python demos/planet_dive.py
    uv run python demos/planet_dive.py --render --duration 10 --output /tmp/planet.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields, Gradient
from manifoldx.resources import (
    DirectionalLight, StandardMaterial, AtmosphereScatteringMaterial, WaterMaterial, CloudMaterial,
)
from manifoldx.sky import starfield

# ---- pure module-level world (imported cleanly by the worker process) --------
R, SEA, PEAK, SUBDIV = 10.0, 0.65, 0.85, 7             # gentle terrain (peaks ~8.5% of radius)
RA = R * 1.20                                           # atmosphere top (the low orbit flies inside it)
RC = R * 1.10                                           # cloud deck (just above the peaks)
CLOUD_DRIFT = 0.0016                                    # radians/frame the cloud shell rotates
ALT_START = 32.0                                        # high-orbit start altitude
ORBIT_ALT = 1.5                                         # low-orbit altitude (inside the atmosphere)
PITCH = 0.37                                           # down-tilt in low orbit (horizon ~40% from bottom)
DESCENT_FRAMES = 190                                    # descent completes ~here; then orbit
AZ_RATE = 0.010                                         # radians/frame the camera circles
SUN_RATE = 0.0040                                       # sun trails the orbit; day holds through the dive,
                                                        # then sunset ~10s and night ~19s as we pull ahead
SUN_I = 1.5                                             # sun intensity (kept low so land colours don't clip)
SPACE = np.array([0.02, 0.03, 0.06])

# Continents (ridged) + rolling hills + fine detail, domain-warped for organic coastlines.
terrain = (
    fields.ridged(seed=4, freq=0.24, octaves=5) * 0.72
    + fields.fbm(seed=8, freq=0.8, octaves=5) * 0.20
    + fields.fbm(seed=15, freq=2.6, octaves=3) * 0.06      # medium detail
    + fields.ridged(seed=21, freq=6.5, octaves=4) * 0.035  # fine near-surface roughness
).warp(0.7, fx=fields.fbm(2, 0.38), fz=fields.fbm(9, 0.38)).remap(0.0, 1.0, -SEA, PEAK)

# Biome by elevation, with snow caps pushed in at high latitude.
height01 = fields.distance().remap(R - SEA, R + PEAK, 0.0, 1.0).clamp(0.0, 1.0)
polar = fields.coord("y").scale(1.0 / R).abs().remap(0.55, 0.92, 0.0, 1.0).clamp(0.0, 1.0)
biome = (height01 + polar * 0.45).clamp(0.0, 1.0)
palette = Gradient([(0.00, "#0a1a3a"), (0.40, "#14314f"), (0.44, "#d9c89a"),
                    (0.50, "#3f8f2f"), (0.62, "#226b1f"), (0.72, "#6e5236"),
                    (0.82, "#8a8378"), (0.90, "#f4f8ff"), (1.00, "#ffffff")])


def build_planet():
    """Pure, picklable: the biome-colored planet as a baked geometry dict."""
    return (GeoMesh.icosphere(subdivisions=SUBDIV, radius=R)
            .displace(terrain, amount=1.0)
            .color_by(biome, palette)
            .to_geometry())


def main():
    engine = mx.Engine("Planet Dive", width=1280, height=720)
    engine.background_color = tuple(SPACE)
    engine.set_sun(DirectionalLight(color="#fff4e8", intensity=SUN_I, direction=(-0.6, -0.3, -0.55)))

    starfield(engine, count=1800, radius=R * 40.0, seed=11,   # space backdrop; fades under daylight
              ground_radius=R, atmo_top=RA - R)
    planet_geo = engine.submit_process(build_planet).wait()   # generate off-thread, block once
    engine.spawn(Mesh(planet_geo),
                 Material(StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True)),
                 Transform())
    engine.spawn(Mesh(GeoMesh.icosphere(SUBDIV - 1, R).to_geometry()),
                 Material(WaterMaterial(color="#0a2352", fresnel_power=4.0)),
                 Transform())
    # Procedural cloud deck (fbm coverage, sun-lit, alpha) — drifts by slowly rotating the shell.
    cloud = engine.spawn(Mesh(GeoMesh.icosphere(6, RC).to_geometry()),
                         Material(CloudMaterial(coverage=0.46, softness=0.17, freq=4.4, opacity=0.48)),
                         Transform())
    # Physically-based single-scattering atmosphere (Rayleigh+Mie) on a shell at RA.
    # Additive; provides the whole sky (blue day / sunset / dark night) from the sun angle.
    engine.spawn(Mesh(GeoMesh.icosphere(5, RA).to_geometry()),
                 Material(AtmosphereScatteringMaterial(R, RA, intensity=6.0, exposure=1.4)),
                 Transform())

    clock = {"frame": 0}

    @engine.system
    def fly(query, dt):
        clock["frame"] += 1
        f = clock["frame"]
        # Day/night: sweep the sun; water, terrain, and atmosphere all react to it.
        sun_az = 0.7 + f * SUN_RATE
        sd = np.array([np.cos(sun_az), -0.28, np.sin(sun_az)])
        engine.set_sun(DirectionalLight(color="#fff4e8", intensity=SUN_I,
                                        direction=tuple(sd / np.linalg.norm(sd))))
        # Drift the clouds by slowly spinning their shell about a slightly tilted axis.
        ang = f * CLOUD_DRIFT
        ax = np.array([0.12, 1.0, 0.05])
        ax /= np.linalg.norm(ax)
        cloud.transform.rot = (*(ax * np.sin(ang / 2)), np.cos(ang / 2))
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
        # Background is always deep space; the scattering atmosphere provides the sky
        # (blue day / sunset / dark night) physically from the sun angle.
        engine.background_color = tuple(SPACE)

    engine.cli()


if __name__ == "__main__":
    main()
