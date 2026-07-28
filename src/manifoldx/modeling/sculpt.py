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


def inflate(mesh: Mesh, center, radius: float, strength: float, profile: str = "smooth") -> Mesh:
    w, base = _selected(mesh, center, radius, profile)
    out = mesh.positions + (strength * w)[:, None] * base.normals
    return mesh.with_positions(out).recompute_normals()


def pinch(mesh: Mesh, center, radius: float, strength: float, profile: str = "smooth") -> Mesh:
    w = Falloff(center, radius, profile).weights(mesh.positions)
    c = np.asarray(center, dtype=np.float32).reshape(3)
    to_center = c[None, :] - mesh.positions
    out = mesh.positions + (strength * w)[:, None] * to_center
    return mesh.with_positions(out).recompute_normals()


def flatten(mesh: Mesh, center, radius: float, strength: float, profile: str = "smooth") -> Mesh:
    w, base = _selected(mesh, center, radius, profile)
    sel = w > 0.0
    if not sel.any():
        return mesh
    ws = w[sel][:, None]
    plane_point = (ws * mesh.positions[sel]).sum(axis=0) / ws.sum()
    plane_normal = (ws * base.normals[sel]).sum(axis=0)
    plane_normal /= np.linalg.norm(plane_normal) or 1.0
    signed = (mesh.positions - plane_point[None, :]) @ plane_normal   # (N,)
    out = mesh.positions - (strength * w * signed)[:, None] * plane_normal[None, :]
    return mesh.with_positions(out).recompute_normals()


def smooth(mesh: Mesh, iterations: int = 1, strength: float = 1.0,
           center=None, radius=None, profile: str = "smooth") -> Mesh:
    if iterations <= 0:
        return mesh
    adj = mesh.adjacency()
    n = len(mesh.positions)
    deg = np.maximum(np.diff(adj.offsets), 1)
    src = np.repeat(np.arange(n), np.diff(adj.offsets))
    if center is not None and radius is not None:
        gate = Falloff(center, radius, profile).weights(mesh.positions)
    else:
        gate = np.ones(n, dtype=np.float32)

    positions = mesh.positions.astype(np.float32)
    for _ in range(iterations):
        sums = np.zeros_like(positions)
        np.add.at(sums, src, positions[adj.neighbors])
        avg = sums / deg[:, None]
        positions = positions + (strength * gate)[:, None] * (avg - positions)

    return mesh.with_positions(positions).recompute_normals()
