import numpy as np
from manifoldx.modeling import Mesh


def test_icosphere_geometry_is_spawn_ready():
    """The demo's geometry dict must satisfy the GeometryRegistry contract."""
    geo = Mesh.icosphere(subdivisions=3).to_geometry()
    assert set(geo) >= {"positions", "normals", "indices"}
    assert geo["positions"].shape[1] == 3
    assert geo["normals"].shape == geo["positions"].shape
    assert geo["indices"].ndim == 1 and geo["indices"].shape[0] % 3 == 0
    assert geo["indices"].max() < geo["positions"].shape[0]
