"""Constructive Solid Geometry via BSP trees (union / difference / intersection).

A faithful pure-numpy port of the classic BSP-tree CSG algorithm
(Naylor-Amanatides-Thibault, popularised by csg.js). See
`.knowledge/analysis/2026-07-28-modeling-booleans-v1-design.md`.
"""

from __future__ import annotations

import sys

import numpy as np

from manifoldx.modeling.mesh import Mesh

_EPS = 1e-5
_COPLANAR, _FRONT, _BACK, _SPANNING = 0, 1, 2, 3


class _Plane:
    __slots__ = ("normal", "w")

    def __init__(self, normal, w):
        self.normal = normal
        self.w = w

    @classmethod
    def from_points(cls, a, b, c):
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / length
        return cls(n, float(np.dot(n, a)))

    def clone(self):
        return _Plane(self.normal.copy(), self.w)

    def flip(self):
        self.normal = -self.normal
        self.w = -self.w

    def split_polygon(self, polygon, coplanar_front, coplanar_back, front, back):
        types = []
        polygon_type = 0
        for v in polygon.vertices:
            t = float(np.dot(self.normal, v)) - self.w
            typ = _BACK if t < -_EPS else (_FRONT if t > _EPS else _COPLANAR)
            polygon_type |= typ
            types.append(typ)

        if polygon_type == _COPLANAR:
            target = coplanar_front if np.dot(self.normal, polygon.plane.normal) > 0 else coplanar_back
            target.append(polygon)
        elif polygon_type == _FRONT:
            front.append(polygon)
        elif polygon_type == _BACK:
            back.append(polygon)
        else:  # SPANNING
            f, b = [], []
            n = len(polygon.vertices)
            for i in range(n):
                j = (i + 1) % n
                ti, tj = types[i], types[j]
                vi, vj = polygon.vertices[i], polygon.vertices[j]
                if ti != _BACK:
                    f.append(vi)
                if ti != _FRONT:
                    b.append(vi)
                if (ti | tj) == _SPANNING:
                    t = (self.w - float(np.dot(self.normal, vi))) / float(np.dot(self.normal, vj - vi))
                    v = vi + (vj - vi) * t
                    f.append(v)
                    b.append(v.copy())
            if len(f) >= 3:
                front.append(_Polygon(f))
            if len(b) >= 3:
                back.append(_Polygon(b))


class _Polygon:
    __slots__ = ("vertices", "plane")

    def __init__(self, vertices):
        self.vertices = vertices
        self.plane = _Plane.from_points(vertices[0], vertices[1], vertices[2])

    def flip(self):
        self.vertices = self.vertices[::-1]
        self.plane.flip()


class _Node:
    __slots__ = ("plane", "front", "back", "polygons")

    def __init__(self, polygons=None):
        self.plane = None
        self.front = None
        self.back = None
        self.polygons = []
        if polygons:
            self.build(polygons)

    def invert(self):
        for p in self.polygons:
            p.flip()
        if self.plane is not None:
            self.plane.flip()
        if self.front is not None:
            self.front.invert()
        if self.back is not None:
            self.back.invert()
        self.front, self.back = self.back, self.front

    def clip_polygons(self, polygons):
        if self.plane is None:
            return list(polygons)
        front, back = [], []
        for p in polygons:
            self.plane.split_polygon(p, front, back, front, back)
        if self.front is not None:
            front = self.front.clip_polygons(front)
        if self.back is not None:
            back = self.back.clip_polygons(back)
        else:
            back = []
        return front + back

    def clip_to(self, bsp):
        self.polygons = bsp.clip_polygons(self.polygons)
        if self.front is not None:
            self.front.clip_to(bsp)
        if self.back is not None:
            self.back.clip_to(bsp)

    def all_polygons(self):
        result = list(self.polygons)
        if self.front is not None:
            result += self.front.all_polygons()
        if self.back is not None:
            result += self.back.all_polygons()
        return result

    def build(self, polygons):
        if not polygons:
            return
        if self.plane is None:
            self.plane = polygons[0].plane.clone()
        front, back = [], []
        for p in polygons:
            self.plane.split_polygon(p, self.polygons, self.polygons, front, back)
        if front:
            if self.front is None:
                self.front = _Node()
            self.front.build(front)
        if back:
            if self.back is None:
                self.back = _Node()
            self.back.build(back)


def _to_polygons(mesh: Mesh):
    p = mesh.positions.astype(np.float64)
    return [_Polygon([p[i], p[j], p[k]]) for i, j, k in mesh.faces]


def _from_polygons(polys) -> Mesh:
    verts, faces = [], []
    for poly in polys:
        vs = poly.vertices
        base = len(verts)
        verts.extend(vs)
        for t in range(1, len(vs) - 1):
            faces.append([base, base + t, base + t + 1])
    positions = np.array(verts, dtype=np.float32) if verts else np.zeros((0, 3), np.float32)
    face_arr = np.array(faces, dtype=np.uint32) if faces else np.zeros((0, 3), np.uint32)
    return Mesh(positions=positions, faces=face_arr).recompute_normals()


def _csg(a_mesh: Mesh, b_mesh: Mesh, kind: str) -> Mesh:
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 100000))
    try:
        a = _Node(_to_polygons(a_mesh))
        b = _Node(_to_polygons(b_mesh))
        if kind == "union":
            a.clip_to(b)
            b.clip_to(a)
            b.invert()
            b.clip_to(a)
            b.invert()
            a.build(b.all_polygons())
        elif kind == "difference":
            a.invert()
            a.clip_to(b)
            b.clip_to(a)
            b.invert()
            b.clip_to(a)
            b.invert()
            a.build(b.all_polygons())
            a.invert()
        elif kind == "intersection":
            a.invert()
            b.clip_to(a)
            b.invert()
            a.clip_to(b)
            b.clip_to(a)
            a.build(b.all_polygons())
            a.invert()
        else:
            raise ValueError(f"unknown boolean op: {kind!r}")
        return _from_polygons(a.all_polygons())
    finally:
        sys.setrecursionlimit(limit)


def union(a: Mesh, b: Mesh) -> Mesh:
    return _csg(a, b, "union")


def difference(a: Mesh, b: Mesh) -> Mesh:
    return _csg(a, b, "difference")


def intersection(a: Mesh, b: Mesh) -> Mesh:
    return _csg(a, b, "intersection")
