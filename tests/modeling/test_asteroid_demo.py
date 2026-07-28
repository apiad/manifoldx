import numpy as np
from manifoldx.modeling import Mesh, noise


def test_icosphere_geometry_is_spawn_ready():
    """The demo's geometry dict must satisfy the GeometryRegistry contract."""
    geo = Mesh.icosphere(subdivisions=3).to_geometry()
    assert set(geo) >= {"positions", "normals", "indices"}
    assert geo["positions"].shape[1] == 3
    assert geo["normals"].shape == geo["positions"].shape
    assert geo["indices"].ndim == 1 and geo["indices"].shape[0] % 3 == 0
    assert geo["indices"].max() < geo["positions"].shape[0]


def test_asteroid_pipeline_produces_valid_geometry():
    rock = (
        Mesh.icosphere(subdivisions=4)
        .displace(noise.fbm(seed=7, octaves=5), amount=0.35)
        .twist(angle=0.4, axis="y")
        .taper(factor=0.2, axis="y")
    )
    geo = rock.to_geometry()
    assert geo["indices"].max() < geo["positions"].shape[0]
    assert np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
