import numpy as np
from manifoldx.modeling import Mesh


def test_subdivide_midpoint_face_count_x4():
    base = Mesh.icosphere(subdivisions=0)          # 20 faces, 12 verts
    one = base.subdivide(iterations=1)
    assert one.faces.shape[0] == 80
    assert one.subdivide(iterations=1).faces.shape[0] == 320


def test_subdivide_shares_midpoints_watertight():
    # Euler: a closed genus-0 mesh has V - E + F = 2. Midpoint keeps it closed.
    m = Mesh.icosphere(subdivisions=0).subdivide(iterations=2)
    V, F = m.positions.shape[0], m.faces.shape[0]
    E = F * 3 // 2                                   # closed triangle mesh
    assert V - E + F == 2
    assert m.faces.max() < V


def test_subdivide_zero_iterations_identity():
    m = Mesh.box(1, 1, 1)
    out = m.subdivide(iterations=0)
    assert out.faces.shape == m.faces.shape


def test_subdivide_loop_face_count_x4():
    m = Mesh.icosphere(subdivisions=1).subdivide(iterations=1, scheme="loop")
    assert m.faces.shape[0] == 80 * 4


def test_subdivide_loop_smooths_a_cube():
    cube = Mesh.box(1, 1, 1)
    r0 = np.linalg.norm(cube.positions, axis=1).max()
    loop = cube.subdivide(iterations=2, scheme="loop")
    r1 = np.linalg.norm(loop.positions, axis=1).max()
    assert r1 < r0 - 0.05
    assert np.all(np.isfinite(loop.positions))


def test_subdivide_loop_keeps_sphere_on_sphere():
    m = Mesh.icosphere(subdivisions=2).subdivide(iterations=1, scheme="loop")
    r = np.linalg.norm(m.positions, axis=1)
    assert r.min() > 0.9 and r.max() <= 1.001


def test_extrude_raises_region_and_adds_walls():
    base = Mesh.plane(width=4, depth=4, segments=8)   # flat y=0, normal +y
    cx = base.positions[base.faces].mean(axis=1)      # face centroids
    mask = np.linalg.norm(cx[:, [0, 2]], axis=1) < 1.0
    out = base.extrude(mask, distance=0.5)
    assert out.faces.shape[0] > base.faces.shape[0]   # walls added
    assert np.isclose(out.positions[:, 1].max(), 0.5, atol=1e-4)
    assert out.faces.max() < out.positions.shape[0]


def test_extrude_empty_mask_identity():
    base = Mesh.box(1, 1, 1)
    out = base.extrude(np.zeros(base.faces.shape[0], bool), distance=1.0)
    assert out.faces.shape == base.faces.shape
