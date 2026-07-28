import numpy as np
from manifoldx.modeling import Mesh, fields


def test_composed_terrain_field_displaces_plane():
    terrain = (
        fields.ridged(seed=3, freq=0.4) * 0.7
        + fields.fbm(seed=7, freq=1.2) * 0.15
    ).warp(0.3, fx=fields.fbm(seed=2), fz=fields.fbm(seed=9)).remap(-1, 1, 0, 1)

    base = Mesh.plane(width=10, depth=10, segments=40)
    out = base.displace(terrain, amount=2.0)
    geo = out.to_geometry()
    assert out.positions.shape == base.positions.shape
    assert np.all(np.isfinite(geo["positions"]))
    assert np.ptp(out.positions[:, 1]) > 0.1        # terrain has relief
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
