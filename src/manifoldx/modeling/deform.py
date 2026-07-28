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
