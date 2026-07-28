import numpy as np
from manifoldx.modeling import Mesh, Falloff


def test_adjacency_of_single_triangle():
    m = Mesh(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    adj = m.adjacency()
    assert adj.offsets.shape == (4,)
    # Every vertex of a triangle neighbors the other two.
    for i in range(3):
        nbrs = set(adj.neighbors[adj.offsets[i]:adj.offsets[i + 1]].tolist())
        assert nbrs == {j for j in range(3) if j != i}


def test_adjacency_is_cached():
    m = Mesh.icosphere(subdivisions=2)
    assert m.adjacency() is m.adjacency()   # same object, built once


def test_adjacency_symmetric_on_icosphere():
    m = Mesh.icosphere(subdivisions=1)
    adj = m.adjacency()
    pairs = set()
    for i in range(len(m.positions)):
        for j in adj.neighbors[adj.offsets[i]:adj.offsets[i + 1]]:
            pairs.add((i, int(j)))
    assert all((j, i) in pairs for (i, j) in pairs)   # undirected


def test_falloff_weights_bounds():
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    w = Falloff(center=(0, 0, 0), radius=1.0).weights(pts)
    assert w[0] == 1.0                 # at center
    assert w[2] == 0.0 and w[3] == 0.0  # at/outside radius
    assert 0.0 < w[1] < 1.0            # partway
    assert w.dtype == np.float32


def test_draw_raises_bump_near_center_only():
    base = Mesh.plane(width=4, depth=4, segments=20)
    out = base.draw(center=(0, 0, 0), radius=1.0, strength=0.5)
    disp = out.positions - base.positions
    d = np.linalg.norm(base.positions - np.array([0, 0, 0]), axis=1)
    assert np.abs(disp[d < 0.2]).max() > 0.1      # bump near center
    assert np.allclose(disp[d > 1.5], 0.0)        # untouched far away
