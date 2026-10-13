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


def _reference_normals(positions, faces):
    """The original np.add.at accumulation, kept as the reference for the fast path."""
    f = faces.astype(np.intp)
    face_n = np.cross(positions[f[:, 1]] - positions[f[:, 0]], positions[f[:, 2]] - positions[f[:, 0]])
    normals = np.zeros_like(positions)
    for k in range(3):
        np.add.at(normals, f[:, k], face_n)
    return normals


def test_recompute_normals_matches_the_scatter_reference():
    from manifoldx.modeling.primitives import icosphere

    rng = np.random.default_rng(0)
    sphere = icosphere(subdivisions=3)
    noisy = sphere.with_positions(sphere.positions + rng.normal(0, 0.05, sphere.positions.shape))
    # A random soup with shared, repeated and unreferenced vertices.
    soup = Mesh(positions=rng.normal(size=(60, 3)).astype(np.float32),
                faces=rng.integers(0, 50, size=(200, 3)).astype(np.uint32))
    for m in (sphere, noisy, soup):
        ref = _reference_normals(m.positions, m.faces)
        ok = np.linalg.norm(ref, axis=1) > 0
        ref = ref[ok] / np.linalg.norm(ref[ok], axis=1, keepdims=True)
        got = m.recompute_normals().normals
        assert got.dtype == np.float32 and got.shape == m.positions.shape
        assert np.abs(got[ok] - ref).max() < 1e-6
        assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-6)
