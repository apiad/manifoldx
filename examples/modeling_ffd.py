"""Free-form deformation — a lattice-warped mesh that breathes, lit + shadowed.

An icosphere is embedded in a 4x4x4 Bezier control lattice. Every frame the
control points oscillate, and the deformed vertices are pushed straight into
the existing GPU vertex buffer (FFD preserves vertex count), so the mesh warps
in real time. The shadow pass reads the same buffer, so the cast shadow writhes
with it.

    uv run python examples/modeling_ffd.py
    uv run python examples/modeling_ffd.py --render --duration 8 --output /tmp/ffd.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial, plane
from manifoldx.systems import Query

engine = mx.Engine("Modeling — FFD", width=1024, height=768)

engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.3, direction=(-0.7, -0.75, -0.4)))
engine.enable_shadows(resolution=2048, bias=0.004, pcf_radius=2)

# --- Floor ---------------------------------------------------------------
FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)
engine.spawn(
    Mesh(plane(14, 14)),
    Material(StandardMaterial(color="#c4c4c4", roughness=0.95)),
    Transform(pos=(0, 0, 0), rot=FLOOR_ROT),
)

# --- The warping blob ----------------------------------------------------
base = GeoMesh.icosphere(subdivisions=4)          # smooth, enough verts to deform cleanly
geo0 = base.to_geometry()
GEO_ID = engine._geometry_registry.register(geo0)  # capture id so we can push new verts each frame
ffd = base.ffd(resolution=(3, 3, 3))
_LATTICE0 = ffd.points.copy()
_N = base.positions.shape[0]

engine.spawn(
    Mesh(geo0),
    Material(StandardMaterial(color="#8a5cd0", roughness=0.35, metallic=0.15)),  # violet
    Transform(pos=(0, 2.2, 0), scale=(1.8, 1.8, 1.8)),
)


@engine.system
def warp(query: Query[Transform], dt: float):
    t = engine.elapsed
    p = _LATTICE0.copy()
    p[..., 0] += 0.35 * np.sin(t * 1.3 + _LATTICE0[..., 1] * 2.2)
    p[..., 1] += 0.30 * np.sin(t * 1.7 + _LATTICE0[..., 0] * 2.0)
    p[..., 2] += 0.35 * np.cos(t * 1.1 + _LATTICE0[..., 1] * 2.2)
    ffd.points[:] = p
    m = ffd.apply()

    bufs = engine._geometry_registry.get_gpu_buffers(GEO_ID)
    if bufs is None:
        return  # buffers not created until the first render pass
    geo = m.to_geometry()
    interleaved = np.empty((_N, 6), dtype=np.float32)
    interleaved[:, :3] = geo["positions"]
    interleaved[:, 3:] = geo["normals"]
    engine._device.queue.write_buffer(bufs["vertex_buffer"], 0, interleaved.tobytes())

    # A slow tumble so the warp reads from every angle.
    query[Transform].rot += Transform.rotation(x=dt * 0.1, y=dt * 0.3, z=0)


engine.camera.fit(radius=7.5, center=(0, 2.0, 0), azimuth=35, elevation=40)


@engine.system
def camera_orbit(query: Query[Transform], dt: float):
    engine.camera.orbit(5 * dt, 0)


if __name__ == "__main__":
    engine.cli()
