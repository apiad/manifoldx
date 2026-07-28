import numpy as np
from manifoldx.modeling import Mesh


def _valid_nonempty(m):
    assert m.faces.shape[0] > 0
    geo = m.to_geometry()
    assert geo["indices"].max() < geo["positions"].shape[0]
    assert np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)


def test_boolean_demo_rounded_die_intersection():
    die = Mesh.box(1.7, 1.7, 1.7).intersection(Mesh.icosphere(subdivisions=2, radius=1.12))
    _valid_nonempty(die)


def test_boolean_demo_drilled_sphere_difference():
    drilled = Mesh.icosphere(subdivisions=2, radius=1.0).difference(Mesh.box(0.7, 0.7, 3.0))
    _valid_nonempty(drilled)


def test_boolean_demo_bond_union():
    bond = Mesh.icosphere(subdivisions=2, radius=0.85).union(
        Mesh.icosphere(subdivisions=2, radius=0.85).translate((1.1, 0, 0))
    )
    _valid_nonempty(bond)
