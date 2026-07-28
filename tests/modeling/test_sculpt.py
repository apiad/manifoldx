import numpy as np
from manifoldx.modeling import Mesh


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
