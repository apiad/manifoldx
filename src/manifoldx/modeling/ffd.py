"""Free-form deformation (Sederberg-Parry): warp a mesh through a Bezier control lattice."""

from __future__ import annotations

import math

import numpy as np

from manifoldx.modeling.mesh import Mesh


def _bernstein_matrix(t: np.ndarray, degree: int) -> np.ndarray:
    """Return (N, degree + 1) Bernstein basis weights B_i^degree(t) for each t."""
    i = np.arange(degree + 1)
    coeff = np.array([math.comb(degree, int(ii)) for ii in i], dtype=np.float64)
    t = t[:, None]
    return coeff[None, :] * (t ** i[None, :]) * ((1.0 - t) ** (degree - i)[None, :])


class FFD:
    """A Bezier control lattice embedding a mesh.

    Move `points` (an (I, J, K, 3) array of control points, initially a regular
    grid over the mesh's bounding box), then call `apply()` to get the deformed
    Mesh. An unmoved lattice reproduces the original mesh exactly.
    """

    def __init__(self, mesh: Mesh, resolution=(2, 2, 2)):
        self.mesh = mesh
        res = np.asarray(resolution, dtype=int)
        p = mesh.positions.astype(np.float64)
        self.lo = p.min(axis=0)
        self.hi = p.max(axis=0)
        span = self.hi - self.lo
        span[span == 0] = 1.0
        stu = (p - self.lo) / span                      # (N, 3) parametric coords in [0, 1]

        grids = [np.linspace(self.lo[d], self.hi[d], res[d] + 1) for d in range(3)]
        gx, gy, gz = np.meshgrid(*grids, indexing="ij")
        self.points = np.stack([gx, gy, gz], axis=-1)   # (I, J, K, 3), mutable

        # Precompute per-vertex Bernstein weights for each axis.
        self._bern = [_bernstein_matrix(stu[:, d], int(res[d])) for d in range(3)]

    def move(self, i: int, j: int, k: int, offset) -> "FFD":
        """Offset a single control point; returns self for chaining."""
        self.points[i, j, k] += np.asarray(offset, dtype=np.float64)
        return self

    def apply(self) -> Mesh:
        """Evaluate the lattice and return the deformed Mesh."""
        bx, by, bz = self._bern
        out = np.einsum("ni,nj,nk,ijkd->nd", bx, by, bz, self.points)
        return self.mesh.with_positions(out.astype(np.float32)).recompute_normals()
