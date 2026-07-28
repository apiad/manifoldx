"""Deformers: per-vertex Mesh -> Mesh operators (no topology change)."""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh

_AXIS = {"x": 0, "y": 1, "z": 2}


def displace(mesh: Mesh, field, amount: float = 1.0, along="normal") -> Mesh:
    values = np.asarray(field(mesh.positions), dtype=np.float32).reshape(-1, 1)
    if along == "normal":
        base = mesh if mesh.normals is not None else mesh.recompute_normals()
        direction = base.normals
    else:
        direction = np.asarray(along, dtype=np.float32).reshape(1, 3)
    new_positions = mesh.positions + amount * values * direction
    return mesh.with_positions(new_positions).recompute_normals()


def twist(mesh: Mesh, angle: float, axis: str = "y") -> Mesh:
    ax = _AXIS[axis]
    u, v = [i for i in range(3) if i != ax]
    p = mesh.positions
    theta = angle * p[:, ax]
    cos, sin = np.cos(theta), np.sin(theta)
    out = p.copy()
    out[:, u] = p[:, u] * cos - p[:, v] * sin
    out[:, v] = p[:, u] * sin + p[:, v] * cos
    return mesh.with_positions(out).recompute_normals()


def bend(mesh: Mesh, angle: float, axis: str = "z", along: str = "x") -> Mesh:
    rot_ax = _AXIS[axis]
    drive = _AXIS[along]
    # The two axes rotated in the bend plane are the ones != rotation axis.
    u, v = [i for i in range(3) if i != rot_ax]
    p = mesh.positions
    extent = np.abs(p[:, drive]).max()
    k = angle / extent if extent > 0 else 0.0
    theta = k * p[:, drive]
    cos, sin = np.cos(theta), np.sin(theta)
    out = p.copy()
    out[:, u] = p[:, u] * cos - p[:, v] * sin
    out[:, v] = p[:, u] * sin + p[:, v] * cos
    return mesh.with_positions(out).recompute_normals()


def taper(mesh: Mesh, factor: float, axis: str = "y") -> Mesh:
    ax = _AXIS[axis]
    u, v = [i for i in range(3) if i != ax]
    p = mesh.positions
    extent = np.abs(p[:, ax]).max()
    norm_coord = p[:, ax] / extent if extent > 0 else np.zeros(len(p))
    scale = (1.0 + factor * norm_coord).reshape(-1, 1)
    out = p.copy()
    out[:, [u, v]] = p[:, [u, v]] * scale
    return mesh.with_positions(out).recompute_normals()
