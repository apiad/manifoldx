import numpy as np
from manifoldx.modeling import Mesh, fields, Gradient


def test_planet_pipeline_valid_with_oceans():
    R = 10.0
    land = fields.ridged(seed=3, freq=0.3) * 0.8 + fields.fbm(seed=7, freq=0.9) * 0.2
    terrain = land.remap(0.0, 1.0, -1.5, 2.5)           # signed: basins below 0
    planet = (
        Mesh.icosphere(subdivisions=4, radius=R)
        .displace(terrain, amount=1.0)
        .color_by(
            fields.distance().remap(R - 1.5, R + 2.5, 0.0, 1.0),
            Gradient([(0, "#123"), (0.5, "#4a7a3a"), (1, "#fff")]),
        )
    )
    geo = planet.to_geometry()
    r = np.linalg.norm(geo["positions"], axis=1)
    assert r.min() < R and r.max() > R                  # basins below, mountains above
    assert "colors" in geo and np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
