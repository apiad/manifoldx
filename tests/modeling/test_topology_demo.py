import numpy as np
from manifoldx.modeling import Mesh, noise


def _valid(geo):
    assert geo["indices"].max() < geo["positions"].shape[0]
    assert np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)


def test_topology_demo_blob_valid():
    blob = (
        Mesh.icosphere(subdivisions=1)
        .subdivide(iterations=2, scheme="loop")
        .displace(noise.fbm(seed=3, octaves=4), amount=0.22)
    )
    _valid(blob.to_geometry())


def test_topology_demo_extruded_crystal_valid():
    ico = Mesh.icosphere(subdivisions=2)
    cap = ico.positions[ico.faces].mean(axis=1)[:, 1] > 0.55
    _valid(ico.extrude(cap, distance=0.5).to_geometry())


def test_topology_demo_lowpoly_valid():
    lowpoly = (
        Mesh.icosphere(subdivisions=4)
        .displace(noise.fbm(seed=7, octaves=5), amount=0.35)
        .decimate(grid=9)
    )
    _valid(lowpoly.to_geometry())
