import numpy as np
from manifoldx.modeling import Mesh


def _volume(m):
    p = m.positions.astype(np.float64)
    f = m.faces.astype(np.int64)
    v0, v1, v2 = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    return abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0


def _valid(m):
    geo = m.to_geometry()
    assert geo["indices"].max() < geo["positions"].shape[0]
    assert np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)


# A box and a sphere whose surfaces cross (sphere pokes out of the box faces).
def _box():
    return Mesh.box(2, 2, 2)


def _sphere():
    return Mesh.icosphere(subdivisions=3, radius=1.3)


def test_union_volume_between_max_and_sum():
    a, b = _box(), _sphere()
    va, vb = _volume(a), _volume(b)
    u = a.union(b)
    vu = _volume(u)
    assert max(va, vb) - 0.2 < vu < va + vb + 0.2
    _valid(u)


def test_difference_less_than_minuend():
    a, b = _box(), _sphere()
    d = a.difference(b)
    vd = _volume(d)
    assert 0.0 < vd < _volume(a)          # material was carved away
    _valid(d)


def test_intersection_less_than_both():
    a, b = _box(), _sphere()
    i = a.intersection(b)
    vi = _volume(i)
    assert 0.0 < vi < min(_volume(a), _volume(b)) + 0.2
    _valid(i)


def test_difference_of_disjoint_is_unchanged_volume():
    a = Mesh.box(1, 1, 1)
    b = Mesh.box(1, 1, 1).translate((5, 0, 0)) if hasattr(Mesh, "translate") else Mesh(
        positions=Mesh.box(1, 1, 1).positions + np.array([5, 0, 0], np.float32),
        faces=Mesh.box(1, 1, 1).faces,
    )
    d = a.difference(b)
    assert np.isclose(_volume(d), _volume(a), atol=0.05)
