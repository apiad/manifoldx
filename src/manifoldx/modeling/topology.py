"""Topology operators: subdivide, extrude, decimate (change vertex/face counts)."""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh


def _unique_edges(faces: np.ndarray):
    """Return (uniq (E,2) sorted, inv (3,M)) mapping each face side to its edge id.

    Side order per face: 0=(a,b), 1=(b,c), 2=(c,a).
    """
    f = faces.astype(np.int64)
    sides = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    sides_sorted = np.sort(sides, axis=1)
    uniq, inv = np.unique(sides_sorted, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)          # numpy 2.x may return (K,1)
    inv = inv.reshape(3, -1)                    # rows: ab, bc, ca
    return uniq, inv


def subdivide(mesh: Mesh, iterations: int = 1, scheme: str = "midpoint") -> Mesh:
    m = mesh
    for _ in range(max(0, int(iterations))):
        m = _subdivide_once(m, scheme)
    return m


def _subdivide_once(mesh: Mesh, scheme: str) -> Mesh:
    pos = mesh.positions.astype(np.float64)
    faces = mesh.faces.astype(np.int64)
    n = len(pos)
    uniq, inv = _unique_edges(faces)
    edge_mid = n + np.arange(len(uniq))

    if scheme == "midpoint":
        mid_pos = (pos[uniq[:, 0]] + pos[uniq[:, 1]]) / 2.0
        new_orig = pos.copy()
    elif scheme == "loop":
        mid_pos, new_orig = _loop_positions(pos, faces, uniq, inv)
    else:
        raise ValueError(f"unknown subdivide scheme: {scheme!r}")

    new_pos = np.concatenate([new_orig, mid_pos], axis=0).astype(np.float32)
    m_ab, m_bc, m_ca = edge_mid[inv[0]], edge_mid[inv[1]], edge_mid[inv[2]]
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    new_faces = np.concatenate([
        np.stack([a, m_ab, m_ca], axis=1),
        np.stack([b, m_bc, m_ab], axis=1),
        np.stack([c, m_ca, m_bc], axis=1),
        np.stack([m_ab, m_bc, m_ca], axis=1),
    ], axis=0).astype(np.uint32)
    return Mesh(positions=new_pos, faces=new_faces)


def _loop_positions(pos, faces, uniq, inv):
    """Loop edge-vertex + repositioned original-vertex coordinates."""
    n = len(pos)
    E = len(uniq)
    # Opposite vertex per (side, face): side0=ab->c, side1=bc->a, side2=ca->b.
    opp = np.stack([faces[:, 2], faces[:, 0], faces[:, 1]], axis=0)  # (3, M)
    edge_of = inv.reshape(-1)              # (3M,)
    opp_flat = opp.reshape(-1)             # (3M,)
    order = np.argsort(edge_of, kind="stable")
    e_sorted = edge_of[order]
    o_sorted = opp_flat[order]
    counts = np.bincount(e_sorted, minlength=E)
    offsets = np.zeros(E + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    first = o_sorted[offsets[:-1]]
    second = o_sorted[np.minimum(offsets[:-1] + 1, offsets[1:] - 1)]
    interior = counts == 2

    v0, v1 = pos[uniq[:, 0]], pos[uniq[:, 1]]
    o0, o1 = pos[first], pos[second]
    mid = np.where(
        interior[:, None],
        3.0 / 8.0 * (v0 + v1) + 1.0 / 8.0 * (o0 + o1),
        0.5 * (v0 + v1),
    )

    # Reposition original vertices.
    # Boundary edges (count == 1) define boundary vertices + their boundary neighbours.
    boundary_edges = uniq[counts == 1]
    is_boundary = np.zeros(n, dtype=bool)
    is_boundary[boundary_edges.ravel()] = True

    # One-ring neighbour sums (undirected) for interior rule.
    both = np.concatenate([uniq, uniq[:, ::-1]], axis=0)
    nbr_sum = np.zeros((n, 3))
    np.add.at(nbr_sum, both[:, 0], pos[both[:, 1]])
    valence = np.bincount(both[:, 0], minlength=n).astype(np.float64)
    valence[valence == 0] = 1.0
    k = valence
    beta = (1.0 / k) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / k)) ** 2)
    interior_new = (1.0 - k * beta)[:, None] * pos + beta[:, None] * nbr_sum

    # Boundary rule: 3/4 v + 1/8 (sum of two boundary neighbours).
    bsum = np.zeros((n, 3))
    np.add.at(bsum, boundary_edges[:, 0], pos[boundary_edges[:, 1]])
    np.add.at(bsum, boundary_edges[:, 1], pos[boundary_edges[:, 0]])
    boundary_new = 0.75 * pos + 0.125 * bsum

    new_orig = np.where(is_boundary[:, None], boundary_new, interior_new)
    return mid, new_orig
