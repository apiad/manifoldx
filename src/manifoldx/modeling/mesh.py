"""Host-side, immutable procedural mesh value type."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np


class VertexAdjacency(NamedTuple):
    """One-ring vertex adjacency in CSR form.

    Vertex i's neighbors are neighbors[offsets[i]:offsets[i + 1]].
    """

    offsets: np.ndarray    # (N + 1,) int64
    neighbors: np.ndarray  # (E,) int32


def _build_adjacency(faces: np.ndarray, n_vertices: int) -> VertexAdjacency:
    f = faces.astype(np.int64)
    # Directed edges for all three triangle sides, both ways.
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edges = np.unique(edges, axis=0)                 # dedup, sorts by (src, dst)
    counts = np.bincount(edges[:, 0], minlength=n_vertices)
    offsets = np.zeros(n_vertices + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return VertexAdjacency(offsets=offsets, neighbors=edges[:, 1].astype(np.int32))


@dataclass(frozen=True)
class Mesh:
    """An immutable triangle mesh. Every operator returns a new Mesh."""

    positions: np.ndarray            # (N, 3) float32
    faces: np.ndarray                # (M, 3) uint32
    normals: np.ndarray | None = None  # (N, 3) float32, lazily computed
    uvs: np.ndarray | None = None      # (N, 2) float32, optional

    def with_positions(self, positions: np.ndarray) -> "Mesh":
        """Return a copy with new positions; normals are invalidated."""
        return replace(
            self,
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            normals=None,
        )

    def recompute_normals(self) -> "Mesh":
        """Return a copy with area-weighted, unit-length vertex normals."""
        p = self.positions
        f = self.faces.astype(np.intp)
        v0, v1, v2 = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
        # Cross-product magnitude is proportional to triangle area -> area weighting.
        face_n = np.cross(v1 - v0, v2 - v0)
        normals = np.zeros_like(p)
        for k in range(3):
            np.add.at(normals, f[:, k], face_n)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        # Degenerate vertices (e.g. folds where opposing face normals cancel)
        # get a fallback direction so every output normal is unit-length.
        degenerate = lengths[:, 0] == 0.0
        if degenerate.any():
            fallback = p[degenerate] - p.mean(axis=0)
            flen = np.linalg.norm(fallback, axis=1, keepdims=True)
            fallback = np.where(flen == 0, np.array([0.0, 0.0, 1.0]), fallback / np.where(flen == 0, 1.0, flen))
            normals[degenerate] = fallback
            lengths[degenerate, 0] = 1.0
        normals = (normals / lengths).astype(np.float32)
        return replace(self, normals=normals)

    def to_geometry(self) -> dict:
        """Emit the geometry dict consumed by the ECS Mesh component."""
        mesh = self if self.normals is not None else self.recompute_normals()
        geo = {
            "positions": np.ascontiguousarray(mesh.positions, dtype=np.float32),
            "normals": np.ascontiguousarray(mesh.normals, dtype=np.float32),
            "indices": np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.uint32),
        }
        if mesh.uvs is not None:
            geo["uvs"] = np.ascontiguousarray(mesh.uvs, dtype=np.float32)
        return geo

    @staticmethod
    def from_geometry(geo: dict) -> "Mesh":
        """Build a Mesh from a geometry dict (inverse of to_geometry)."""
        positions = np.ascontiguousarray(geo["positions"], dtype=np.float32)
        faces = np.asarray(geo["indices"]).reshape(-1, 3).astype(np.uint32)
        normals = geo.get("normals")
        uvs = geo.get("uvs")
        return Mesh(
            positions=positions,
            faces=faces,
            normals=None if normals is None else np.ascontiguousarray(normals, dtype=np.float32),
            uvs=None if uvs is None else np.ascontiguousarray(uvs, dtype=np.float32),
        )

    def adjacency(self) -> "VertexAdjacency":
        """One-ring vertex adjacency (CSR), built once and cached on the instance."""
        cached = getattr(self, "_adjacency_cache", None)
        if cached is None:
            cached = _build_adjacency(self.faces, len(self.positions))
            object.__setattr__(self, "_adjacency_cache", cached)
        return cached

    # --- Primitives (delegated to modeling.primitives; lazy import avoids a cycle) ---

    @classmethod
    def icosphere(cls, subdivisions: int = 2, radius: float = 1.0) -> "Mesh":
        from manifoldx.modeling import primitives
        return primitives.icosphere(subdivisions, radius)

    @classmethod
    def box(cls, width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> "Mesh":
        from manifoldx.modeling import primitives
        return primitives.box(width, height, depth)

    @classmethod
    def plane(cls, width: float = 1.0, depth: float = 1.0, segments: int = 1) -> "Mesh":
        from manifoldx.modeling import primitives
        return primitives.plane(width, depth, segments)

    @classmethod
    def cylinder(cls, radius: float = 1.0, height: float = 2.0, segments: int = 32) -> "Mesh":
        from manifoldx.modeling import primitives
        return primitives.cylinder(radius, height, segments)

    @classmethod
    def torus(cls, major: float = 1.0, minor: float = 0.35,
              major_segments: int = 32, minor_segments: int = 16) -> "Mesh":
        from manifoldx.modeling import primitives
        return primitives.torus(major, minor, major_segments, minor_segments)

    # --- Deformers (delegated to modeling.deform; lazy import avoids a cycle) ---

    def displace(self, field, amount: float = 1.0, along="normal") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.displace(self, field, amount, along)

    def twist(self, angle: float, axis: str = "y") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.twist(self, angle, axis)

    def bend(self, angle: float, axis: str = "z", along: str = "x") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.bend(self, angle, axis, along)

    def taper(self, factor: float, axis: str = "y") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.taper(self, factor, axis)

    # --- Sculpt brushes (delegated to modeling.sculpt; lazy import avoids a cycle) ---

    def draw(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.draw(self, center, radius, strength, profile)

    def inflate(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.inflate(self, center, radius, strength, profile)

    def pinch(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.pinch(self, center, radius, strength, profile)

    def flatten(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.flatten(self, center, radius, strength, profile)

    def smooth(self, iterations: int = 1, strength: float = 1.0,
               center=None, radius=None, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.smooth(self, iterations, strength, center, radius, profile)

    # --- Topology (delegated to modeling.topology; lazy import avoids a cycle) ---

    def subdivide(self, iterations: int = 1, scheme: str = "midpoint") -> "Mesh":
        from manifoldx.modeling import topology
        return topology.subdivide(self, iterations, scheme)

    def extrude(self, face_mask, distance: float) -> "Mesh":
        from manifoldx.modeling import topology
        return topology.extrude(self, face_mask, distance)

    def decimate(self, grid: int = 32) -> "Mesh":
        from manifoldx.modeling import topology
        return topology.decimate(self, grid)
