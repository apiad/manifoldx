import numpy as np
from manifoldx.modeling import Mesh


def test_ffd_demo_warp_pose_valid_and_preserves_count():
    base = Mesh.icosphere(subdivisions=3)
    ffd = base.ffd(resolution=(3, 3, 3))
    ffd.points[..., 0] += 0.3 * np.sin(ffd.points[..., 1] * 2.0)
    warped = ffd.apply()
    assert warped.positions.shape == base.positions.shape   # FFD preserves vertex count
    geo = warped.to_geometry()
    assert geo["positions"].shape[0] == base.positions.shape[0]
    assert np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
