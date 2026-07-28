import numpy as np
from manifoldx.modeling import Mesh, FFD


def test_ffd_identity_when_lattice_unmoved():
    m = Mesh.icosphere(subdivisions=3)
    ffd = FFD(m, resolution=(2, 2, 2))
    out = ffd.apply()
    assert out.positions.shape == m.positions.shape
    assert np.allclose(out.positions, m.positions, atol=1e-4)


def test_ffd_lattice_shape():
    m = Mesh.box(2, 2, 2)
    ffd = FFD(m, resolution=(2, 3, 4))
    assert ffd.points.shape == (3, 4, 5, 3)   # (res + 1) control points per axis


def test_ffd_moving_top_layer_raises_top():
    m = Mesh.icosphere(subdivisions=3, radius=1.0)
    ffd = FFD(m, resolution=(2, 2, 2))
    top_before = m.positions[:, 1].max()
    ffd.points[:, -1, :, 1] += 1.5          # lift the whole top control layer (+y)
    out = ffd.apply()
    assert out.positions[:, 1].max() > top_before + 0.5
    assert out.positions.shape == m.positions.shape
    assert np.all(np.isfinite(out.positions))


def test_ffd_via_mesh_method():
    m = Mesh.icosphere(subdivisions=3, radius=1.0)
    ffd = m.ffd(resolution=(2, 2, 2))
    ffd.points[-1, :, :, 0] += 0.6          # push the +x control plane further out
    out = ffd.apply()
    assert out.positions[:, 0].max() > m.positions[:, 0].max() + 0.2
    assert out.faces.shape == m.faces.shape
