"""Pin the exact outputs of adjacency, subdivide, decimate and smooth against
plain-Python references on small meshes, so a faster implementation must give
the same arrays: integer outputs exactly, float outputs to float32 rounding."""

import numpy as np
import pytest

from manifoldx.modeling import Mesh


def _meshes():
    rng = np.random.default_rng(7)
    sphere = Mesh.icosphere(subdivisions=1)
    bumpy = Mesh.icosphere(subdivisions=2)
    bumpy = bumpy.with_positions(bumpy.positions * (1 + 0.1 * rng.random((len(bumpy.positions), 1))))
    return {"sphere": sphere, "bumpy": bumpy, "plane": Mesh.plane(2.0, 1.0, segments=3)}


MESHES = _meshes()


def _ref_neighbors(faces, n):
    nbrs = [set() for _ in range(n)]
    for a, b, c in faces.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            nbrs[u].add(v)
            nbrs[v].add(u)
    return [sorted(s) for s in nbrs]


def _ref_edges(faces):
    """Sorted unique undirected edges and the edge id of each face side (ab, bc, ca)."""
    sides = [[tuple(sorted(p)) for p in ((a, b), (b, c), (c, a))] for a, b, c in faces.tolist()]
    edges = sorted({e for s in sides for e in s})
    ids = {e: i for i, e in enumerate(edges)}
    return edges, [[ids[e] for e in s] for s in sides]


def _ref_subdivide(mesh, scheme):
    pos = mesh.positions.astype(np.float64)
    faces = mesh.faces.astype(np.int64)
    n = len(pos)
    edges, side_ids = _ref_edges(faces)
    opposite = {e: [] for e in edges}
    for (a, b, c), (ab, bc, ca) in zip(faces.tolist(), side_ids):
        opposite[edges[ab]].append(c)
        opposite[edges[bc]].append(a)
        opposite[edges[ca]].append(b)

    if scheme == "midpoint":
        mids = [(pos[u] + pos[v]) / 2.0 for u, v in edges]
        orig = pos.copy()
    else:
        mids = []
        for u, v in edges:
            o = opposite[(u, v)]
            if len(o) == 2:
                mids.append(3.0 / 8.0 * (pos[u] + pos[v]) + 1.0 / 8.0 * (pos[o[0]] + pos[o[1]]))
            else:
                mids.append(0.5 * (pos[u] + pos[v]))
        boundary = [e for e in edges if len(opposite[e]) == 1]
        bnbrs = [[] for _ in range(n)]
        for u, v in boundary:
            bnbrs[u].append(v)
            bnbrs[v].append(u)
        nbrs = _ref_neighbors(faces, n)
        orig = np.empty_like(pos)
        for i in range(n):
            if bnbrs[i]:
                orig[i] = 0.75 * pos[i] + 0.125 * sum(pos[j] for j in bnbrs[i])
            else:
                k = len(nbrs[i])
                beta = (1.0 / k) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / k)) ** 2)
                orig[i] = (1.0 - k * beta) * pos[i] + beta * sum(pos[j] for j in nbrs[i])

    new_faces = []
    for quarter in range(4):
        for (a, b, c), (ab, bc, ca) in zip(faces.tolist(), side_ids):
            mab, mbc, mca = n + ab, n + bc, n + ca
            new_faces.append([[a, mab, mca], [b, mbc, mab], [c, mca, mbc], [mab, mbc, mca]][quarter])
    return np.concatenate([orig, np.array(mids)]), np.array(new_faces)


@pytest.mark.parametrize("name", MESHES)
def test_adjacency_csr_exact(name):
    m = MESHES[name]
    adj = Mesh(positions=m.positions, faces=m.faces).adjacency()
    ref = _ref_neighbors(m.faces, len(m.positions))
    assert adj.offsets.dtype == np.int64 and adj.neighbors.dtype == np.int32
    np.testing.assert_array_equal(adj.offsets, np.concatenate([[0], np.cumsum([len(r) for r in ref])]))
    np.testing.assert_array_equal(adj.neighbors, np.concatenate(ref))


@pytest.mark.parametrize("scheme", ["midpoint", "loop"])
@pytest.mark.parametrize("name", MESHES)
def test_subdivide_exact(name, scheme):
    m = MESHES[name]
    out = m.subdivide(1, scheme=scheme)
    ref_pos, ref_faces = _ref_subdivide(m, scheme)
    assert out.faces.dtype == np.uint32 and out.positions.dtype == np.float32
    np.testing.assert_array_equal(out.faces, ref_faces)
    np.testing.assert_allclose(out.positions, ref_pos, rtol=0, atol=1e-6)


@pytest.mark.parametrize("grid", [3, 4, 8])
@pytest.mark.parametrize("name", MESHES)
def test_decimate_exact(name, grid):
    m = MESHES[name]
    out = m.decimate(grid)
    pos = m.positions.astype(np.float64)
    lo = pos.min(axis=0)
    cell = (pos.max(axis=0) - lo).max() / grid
    keys = [tuple(k) for k in np.floor((pos - lo) / cell).astype(np.int64).tolist()]
    ids = {k: i for i, k in enumerate(sorted(set(keys)))}
    members = [[] for _ in ids]
    for v, k in enumerate(keys):
        members[ids[k]].append(pos[v])
    faces = [[ids[keys[v]] for v in f] for f in m.faces.tolist()]
    faces = [f for f in faces if len(set(f)) == 3]
    used = sorted({c for f in faces for c in f})
    remap = {c: i for i, c in enumerate(used)}
    assert out.faces.dtype == np.uint32 and out.positions.dtype == np.float32
    np.testing.assert_array_equal(out.faces, [[remap[c] for c in f] for f in faces])
    np.testing.assert_allclose(out.positions, [np.mean(members[c], axis=0) for c in used], rtol=0, atol=1e-6)


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("name", MESHES)
def test_smooth_exact(name, iterations):
    m = MESHES[name]
    out = m.smooth(iterations=iterations, strength=0.7)
    nbrs = _ref_neighbors(m.faces, len(m.positions))
    pos = m.positions.astype(np.float64)
    for _ in range(iterations):
        avg = np.array([pos[r].mean(axis=0) for r in nbrs])
        pos = pos + 0.7 * (avg - pos)
    assert out.positions.dtype == np.float32
    np.testing.assert_allclose(out.positions, pos, rtol=0, atol=1e-6)
