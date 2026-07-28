import numpy as np
from manifoldx.modeling import Mesh


def _unit_triangle():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    return Mesh(positions=positions, faces=faces)


def test_with_positions_invalidates_normals():
    m = _unit_triangle().recompute_normals()
    assert m.normals is not None
    moved = m.with_positions(m.positions + np.array([0, 0, 1], dtype=np.float32))
    assert moved.normals is None
    assert np.allclose(moved.positions[:, 2], 1.0)


def test_recompute_normals_unit_length_and_direction():
    m = _unit_triangle().recompute_normals()
    lengths = np.linalg.norm(m.normals, axis=1)
    assert np.allclose(lengths, 1.0)
    # A CCW triangle in the z=0 plane faces +z.
    assert np.allclose(m.normals[0], [0, 0, 1], atol=1e-5)


def test_to_geometry_shapes_and_dtypes():
    geo = _unit_triangle().to_geometry()
    assert geo["positions"].dtype == np.float32 and geo["positions"].shape == (3, 3)
    assert geo["normals"].dtype == np.float32 and geo["normals"].shape == (3, 3)
    assert geo["indices"].dtype == np.uint32 and geo["indices"].shape == (3,)
    assert list(geo["indices"]) == [0, 1, 2]


def test_from_geometry_roundtrip():
    m = _unit_triangle().recompute_normals()
    back = Mesh.from_geometry(m.to_geometry())
    assert np.allclose(back.positions, m.positions)
    assert back.faces.shape == (1, 3)
    assert np.array_equal(back.faces[0], [0, 1, 2])
