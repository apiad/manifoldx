"""Real-time terrain streaming — CPU patches generated on a background thread.

A ring of K static, world-aligned terrain patches tiles the ground ahead of a
forward-flying camera. Each patch is generated off the render thread by
`@engine.background`; when ready it is swapped into the rearmost slot in place
(`handle.set_geometry`) and repositioned. Shader distance fog dissolves the
horizon. The render loop stays GPU-bound (~60 fps) — generation is amortized off
the hot path — versus `modeling_ridge_flyby.py`'s per-frame regen (~1.7 fps).

    uv run python examples/modeling_terrain_stream.py
    uv run python examples/modeling_terrain_stream.py --render --duration 10 --output /tmp/stream.mp4
"""

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields, Gradient
from manifoldx.resources import DirectionalLight, StandardMaterial

engine = mx.Engine("Terrain Stream", width=1024, height=768)
engine.background_color = (0.72, 0.78, 0.86)
engine.set_sun(DirectionalLight(color="#fff0dc", intensity=3.1, direction=(-0.5, -0.55, -0.65)))
engine.enable_fog(start=30, end=110)

# ---- the world, composed from primitives (pure, thread-safe) ----
HEIGHT, DEPTH, SPEED, K = 8.0, 52.0, 9.0, 3
TEMPLATE = GeoMesh.plane(width=44, depth=DEPTH, segments=150)
terrain = (fields.ridged(seed=5, freq=0.12) * 0.85
           + fields.fbm(seed=7, freq=0.5) * 0.15).warp(1.3, fx=fields.fbm(2, 0.15), fz=fields.fbm(9, 0.15))
palette = Gradient([(0.00, "#3a5f8a"), (0.06, "#c2b280"), (0.16, "#4a7a3a"),
                    (0.45, "#5f6e38"), (0.65, "#6e5a44"), (0.82, "#8a8078"), (0.96, "#ffffff")])


@engine.background
def patch_at(world_z):
    return (TEMPLATE
            .displace(terrain.shift((0, 0, world_z)), amount=HEIGHT)
            .color_by(fields.coord("y").remap(0.0, HEIGHT, 0.0, 1.0), palette))


# ---- a ring of K recycled slots (built synchronously at startup) ----
slots = [
    engine.spawn(
        Mesh(patch_at(i * DEPTH).wait().to_geometry()),
        Material(StandardMaterial("#ffffff", roughness=0.92, vertex_colors=True)),
        Transform(pos=(0, 0, i * DEPTH)),
    )
    for i in range(K)
]
engine.camera.set_pose((0, 6.5, -5), (0, 2.5, 22))

st = {"next_z": K * DEPTH, "pending": None, "rear": 0}


@engine.system
def stream(query, dt):
    engine.camera.move_by((0, 0, SPEED * dt))
    if st["pending"] is None and engine.camera.position[2] > (st["rear"] + 1) * DEPTH:
        st["pending"] = patch_at(st["next_z"])            # generate the next patch off-thread
    if st["pending"] is not None and st["pending"].ready:
        slot = slots[st["rear"] % K]
        slot.set_geometry(st["pending"].result)           # main-thread swap-in (in place)
        slot.transform.pos = (0, 0, st["next_z"])
        st["next_z"] += DEPTH
        st["rear"] += 1
        st["pending"] = None


if __name__ == "__main__":
    engine.cli()
