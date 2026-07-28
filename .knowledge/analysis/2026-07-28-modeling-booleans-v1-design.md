# Modeling Booleans v1 — Design (Batch 4)

**Date:** 2026-07-28
**Scope:** Constructive Solid Geometry for `manifoldx.modeling` — `union`, `difference`, `intersection` on `Mesh` values, reimplemented (no vendored CSG dependency) via a BSP-tree algorithm.

## Motivation

Batch 4 of the geometric-modeling roadmap. The umbrella design (`2026-07-28-geometric-modeling-v1-design.md`) fenced booleans off to their own cycle because robust CSG is the one genuinely deep algorithm in the subsystem — not in the "no novel algorithm" register of deformers/sculpt/topology. Alex's call: **reimplement, do not vendor `manifold3d`**, keeping the pure-numpy ethos.

## Approach — BSP-tree CSG (first cut)

A faithful port of the classic BSP-tree CSG algorithm (Naylor–Amanatides–Thibault, popularised by `csg.js`). Each solid is a set of convex coplanar polygons; a BSP node partitions space by a polygon's plane. The three operations are expressed by clipping each solid's polygons against the other's BSP tree, with well-placed inversions:

```
union(a, b):        a.clipTo(b); b.clipTo(a); b.invert(); b.clipTo(a); b.invert(); a.build(b.polys)
difference(a, b):   a.invert(); a.clipTo(b); b.clipTo(a); b.invert(); b.clipTo(a); b.invert(); a.build(b.polys); a.invert()
intersection(a, b): a.invert(); b.clipTo(a); b.invert(); a.clipTo(b); b.clipTo(a); a.build(b.polys); a.invert()
```

This is the robust, well-understood core. It is exact for surface-intersecting solids; coplanar-face and near-degenerate cases use an `EPSILON` classification. The Manifold-style halfedge kernel (exact predicates, guaranteed watertight under any degeneracy) is the **deferred** quality tier, to be built only if the BSP cut's numerical fragility bites in practice.

## Components

- `src/manifoldx/modeling/boolean.py`:
  - `_Plane(normal, w)` — `from_points`, `flip`, `split_polygon` (COPLANAR/FRONT/BACK/SPANNING classification + spanning split at the plane).
  - `_Polygon(vertices)` — convex coplanar vertex loop + its plane; `flip`.
  - `_Node(polygons)` — BSP node: `build`, `invert`, `clip_polygons`, `clip_to`, `all_polygons`.
  - `_to_polygons(mesh)` / `_from_polygons(polys)` — triangle mesh ↔ polygon list (fan-triangulate on the way back; normals recomputed).
  - `union(a, b)`, `difference(a, b)`, `intersection(a, b)` → `Mesh`.
- `Mesh.union/difference/intersection(other)` fluent wrappers.

Recursion depth follows BSP tree height; the ops raise the interpreter recursion limit for their duration and restore it after.

## Non-goals (v1)

- Vertex welding of the result (duplicated per-polygon vertices are valid for rendering and give clean flat CSG edges).
- Exact-predicate robustness / guaranteed watertightness under arbitrary coplanar degeneracy (the deferred Manifold tier).
- N-way boolean trees as a single call (compose the binary ops).

## Testing

Signed-volume checks via the divergence theorem on the closed result: `union` volume between `max(a,b)` and `a+b`; `difference` strictly less than `a`; `intersection` less than `min(a,b)`; all on robustly surface-intersecting inputs (e.g. a box and a sphere whose surfaces cross). Plus validity (finite positions, in-range indices, unit normals) and an animated + lit + shadowed demo.
