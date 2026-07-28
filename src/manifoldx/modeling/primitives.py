"""Procedural primitive generators returning modeling.Mesh values."""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> Mesh:
    """Geodesic sphere from a subdivided icosahedron (uniform-ish triangles)."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(subdivisions):
        verts, faces = _subdivide_midpoint(verts, faces)

    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return Mesh(
        positions=verts.astype(np.float32),
        faces=faces.astype(np.uint32),
    )


def _subdivide_midpoint(verts: np.ndarray, faces: np.ndarray):
    """Split each triangle into 4 via edge midpoints, sharing midpoints across edges."""
    verts = list(map(tuple, verts))
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        va, vb = np.asarray(verts[a]), np.asarray(verts[b])
        verts.append(tuple((va + vb) / 2.0))
        cache[key] = len(verts) - 1
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)
