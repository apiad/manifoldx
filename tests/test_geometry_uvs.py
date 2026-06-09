import numpy as np
from manifoldx.resources import sphere, plane


def test_sphere_emits_uvs():
    geo = sphere(1.0, segments=8)
    assert "uvs" in geo
    uvs = geo["uvs"]
    assert uvs.dtype == np.float32
    assert uvs.shape == (geo["positions"].shape[0], 2)
    assert uvs.min() >= 0.0
    assert uvs.max() <= 1.0


def test_plane_emits_uvs():
    geo = plane(2.0, 2.0)
    assert "uvs" in geo
    uvs = geo["uvs"]
    assert uvs.shape == (4, 2)
    expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    np.testing.assert_allclose(uvs, expected)
