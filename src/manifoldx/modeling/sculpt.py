"""Programmatic sculpt brushes: Falloff-weighted Mesh -> Mesh operators."""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh


class Falloff:
    """A radial brush region: weight 1 at center, 0 at/outside radius."""

    def __init__(self, center, radius: float, profile: str = "smooth"):
        self.center = np.asarray(center, dtype=np.float32).reshape(3)
        self.radius = float(radius)
        self.profile = profile

    def weights(self, positions: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(positions - self.center, axis=1)
        s = np.clip(1.0 - d / self.radius, 0.0, 1.0)
        if self.profile == "smooth":
            w = s * s * (3.0 - 2.0 * s)     # smoothstep
        elif self.profile == "linear":
            w = s
        elif self.profile == "constant":
            w = (s > 0.0).astype(np.float32)
        else:
            raise ValueError(f"unknown falloff profile: {self.profile!r}")
        return w.astype(np.float32)


def _selected(mesh: Mesh, center, radius, profile):
    w = Falloff(center, radius, profile).weights(mesh.positions)
    base = mesh if mesh.normals is not None else mesh.recompute_normals()
    return w, base


def draw(mesh: Mesh, center, radius: float, strength: float, profile: str = "smooth") -> Mesh:
    w, base = _selected(mesh, center, radius, profile)
    sel = w > 0.0
    if not sel.any():
        return mesh
    region_normal = base.normals[sel].mean(axis=0)
    region_normal /= np.linalg.norm(region_normal) or 1.0
    out = mesh.positions + (strength * w)[:, None] * region_normal[None, :]
    return mesh.with_positions(out).recompute_normals()
