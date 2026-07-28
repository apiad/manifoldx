# Geometric Modeling v1 — Plan 3: Topology

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Batch 3 of `manifoldx.modeling` — topology operators `subdivide` (midpoint + Loop), `extrude` (region), and `decimate` (grid vertex-clustering), plus a rich animated + lit + shadowed demo.

**Architecture:** New `src/manifoldx/modeling/topology.py`, operators `Mesh -> Mesh`, exposed as fluent methods. These change vertex/face counts (unlike deformers/sculpt), so tests assert on count invariants + validity (in-range indices, finite, watertight-ish) rather than fixed positions.

**Tech Stack:** Python 3.13+, numpy, `uv`. Design: `.knowledge/analysis/2026-07-28-geometric-modeling-v1-design.md` (Batch 3, amended: `decimate` = grid clustering first cut; QEM deferred).

## Global Constraints

- Pure numpy, `uv run`, dependency-free. Operators in `topology.py`; `Mesh` wrappers lazy-import.
- Count-changing ops still produce a valid `Mesh`: `faces` int in `[0, N)`, `positions` finite, normals recomputed at `to_geometry()`.
- Conventional commits `feat(modeling):`. Not in this plan: booleans (Batch 4).

## Consumed from Plans 1–2 (real, on `main`)

`Mesh(positions (N,3) f32, faces (M,3) u32, normals|None, uvs|None)`, `with_positions`, `recompute_normals`, `adjacency() -> VertexAdjacency(offsets, neighbors)`, primitives, deformers, sculpt brushes.

---

### Task 1: `subdivide(iterations, scheme="midpoint")` — midpoint (linear)

**Files:** Create `src/manifoldx/modeling/topology.py`; modify `mesh.py` (`Mesh.subdivide`); test `tests/modeling/test_topology.py`.

**Interface:** `topology.subdivide(mesh, iterations=1, scheme="midpoint") -> Mesh`; `Mesh.subdivide(iterations=1, scheme="midpoint")`. Each iteration splits every triangle into 4 via shared edge midpoints (each unique edge → one shared midpoint vertex, so the result stays connected/watertight).

**Algorithm (midpoint):** unique undirected edges via `np.unique(np.sort(edges), axis=0, return_inverse=True)`; midpoint vertex index = `N + edge_id`; midpoint pos = `(v0+v1)/2`; each face `(a,b,c)` → `(a,mab,mca)`,`(b,mbc,mab)`,`(c,mca,mbc)`,`(mab,mbc,mca)`.

**Tests:**
```python
# tests/modeling/test_topology.py
import numpy as np
from manifoldx.modeling import Mesh

def test_subdivide_midpoint_face_count_x4():
    base = Mesh.icosphere(subdivisions=0)          # 20 faces, 12 verts
    one = base.subdivide(iterations=1)
    assert one.faces.shape[0] == 80
    assert one.subdivide(iterations=1).faces.shape[0] == 320

def test_subdivide_shares_midpoints_watertight():
    # Euler: a closed genus-0 mesh has V - E + F = 2. Midpoint keeps it closed.
    m = Mesh.icosphere(subdivisions=0).subdivide(iterations=2)
    V, F = m.positions.shape[0], m.faces.shape[0]
    E = F * 3 // 2                                   # closed triangle mesh
    assert V - E + F == 2
    assert m.faces.max() < V

def test_subdivide_zero_iterations_identity():
    m = Mesh.box(1, 1, 1)
    out = m.subdivide(iterations=0)
    assert out.faces.shape == m.faces.shape
```

**Commit:** `feat(modeling): subdivide (midpoint, shared edge vertices)`

---

### Task 2: `subdivide(scheme="loop")` — smooth subdivision

**Files:** modify `topology.py`.

**Interface:** same signature, `scheme="loop"`. Midpoint topology, but vertices repositioned by Loop weights so the surface smooths toward the limit surface.

**Algorithm (Loop):**
- New edge vertices: interior edge (2 adjacent faces, opposite verts `c,d`) → `3/8·(v0+v1) + 1/8·(c+d)`; boundary edge (1 face) → `1/2·(v0+v1)`.
- Original vertices, interior valence `k`: `β = (1/k)·(5/8 − (3/8 + 1/4·cos(2π/k))²)`; `v' = (1−kβ)·v + β·Σneighbors`. Boundary vertex: `3/4·v + 1/8·(two boundary neighbors)`.
- Opposite-vertex-per-edge computed by sorting the flattened `(edge_id, opposite_vertex)` pairs and slicing per edge via CSR offsets (fully vectorized; no python face loop).

**Tests:**
```python
def test_subdivide_loop_face_count_x4():
    m = Mesh.icosphere(subdivisions=1).subdivide(iterations=1, scheme="loop")
    assert m.faces.shape[0] == 80 * 4

def test_subdivide_loop_smooths_a_cube():
    # Loop-subdividing a cube rounds it: max radius shrinks below the corner radius sqrt(3)/2.
    cube = Mesh.box(1, 1, 1)
    r0 = np.linalg.norm(cube.positions, axis=1).max()
    loop = cube.subdivide(iterations=2, scheme="loop")
    r1 = np.linalg.norm(loop.positions, axis=1).max()
    assert r1 < r0 - 0.05
    assert np.all(np.isfinite(loop.positions))

def test_subdivide_loop_keeps_sphere_on_sphere():
    m = Mesh.icosphere(subdivisions=2).subdivide(iterations=1, scheme="loop")
    r = np.linalg.norm(m.positions, axis=1)
    assert r.min() > 0.9 and r.max() <= 1.001     # stays ~unit sphere
```

**Commit:** `feat(modeling): subdivide loop scheme (smooth)`

---

### Task 3: `extrude(face_mask, distance)` — region extrude

**Files:** modify `topology.py`, `mesh.py` (`Mesh.extrude`).

**Interface:** `topology.extrude(mesh, face_mask, distance) -> Mesh`; `Mesh.extrude(face_mask, distance)`. `face_mask`: bool array length `M`. Raises the selected region along vertex normals by `distance` and stitches side walls around the region boundary. Region-interior vertices (all incident faces selected) move in place; region-boundary vertices (incident to selected *and* unselected faces) get a raised duplicate for the selected side; each boundary edge (in exactly one selected face) spawns a 2-triangle wall.

**Tests:**
```python
def test_extrude_raises_region_and_adds_walls():
    base = Mesh.plane(width=4, depth=4, segments=8)   # flat y=0
    cx = base.positions[base.faces].mean(axis=1)      # face centroids
    mask = np.linalg.norm(cx[:, [0, 2]], axis=1) < 1.0
    out = base.extrude(mask, distance=0.5)
    assert out.faces.shape[0] > base.faces.shape[0]   # walls added
    # some vertices lifted to ~+0.5 in y (plane normal is +y)
    assert np.isclose(out.positions[:, 1].max(), 0.5, atol=1e-4)
    assert out.faces.max() < out.positions.shape[0]

def test_extrude_empty_mask_identity():
    base = Mesh.box(1, 1, 1)
    out = base.extrude(np.zeros(base.faces.shape[0], bool), distance=1.0)
    assert out.faces.shape == base.faces.shape
```

**Commit:** `feat(modeling): extrude (region raise + side walls)`

---

### Task 4: `decimate(grid)` — grid vertex-clustering

**Files:** modify `topology.py`, `mesh.py` (`Mesh.decimate`).

**Interface:** `topology.decimate(mesh, grid=32) -> Mesh`; `Mesh.decimate(grid=32)`. Overlays a `grid`-resolution lattice over the bounding box (longest axis = `grid` cells), snaps each vertex to its cell's average position, remaps faces, drops faces whose corners collapse into fewer than 3 distinct cells. Robust, always valid. (QEM edge-collapse is the quality refinement, deferred.)

**Tests:**
```python
def test_decimate_reduces_faces_and_stays_valid():
    base = Mesh.icosphere(subdivisions=4)            # 20480 faces
    low = base.decimate(grid=8)
    assert low.faces.shape[0] < base.faces.shape[0] // 4
    assert low.faces.shape[0] > 0
    assert low.faces.max() < low.positions.shape[0]
    assert np.all(np.isfinite(low.positions))

def test_decimate_preserves_bounding_box_roughly():
    base = Mesh.icosphere(subdivisions=4, radius=2.0)
    low = base.decimate(grid=10)
    assert np.allclose(np.abs(low.positions).max(axis=0), 2.0, atol=0.4)

def test_decimate_no_degenerate_faces():
    low = Mesh.icosphere(subdivisions=3).decimate(grid=6)
    a, b, c = low.faces[:, 0], low.faces[:, 1], low.faces[:, 2]
    assert np.all((a != b) & (b != c) & (a != c))
```

**Commit:** `feat(modeling): decimate (grid vertex-clustering)`

---

### Task 5: Topology demo + design amendment + CHANGELOG

**Files:** create `examples/modeling_topology.py`; modify design doc (decimate line) + `CHANGELOG.md`; test `tests/modeling/test_topology_demo.py`.

**Demo — `examples/modeling_topology.py`:** a lit, animated, shadowed gallery of three procedural forms on a floor:
1. **Loop-subdivided organic blob** — `icosphere(1).subdivide(2, "loop").displace(fbm)`.
2. **Extruded plateau** — `plane(segments=12)` with a central region extruded into a mesa, then `smooth(1)`.
3. **Decimated low-poly asteroid** — the Batch-1 asteroid `.decimate(grid=10)` for a faceted crystal look.

Sun (`DirectionalLight`) + `enable_shadows(resolution=2048, pcf_radius=2)`; the three forms orbit a center and bob (system over `engine.elapsed`), each also spinning; `camera.orbit` for a slow turntable. Renders via `--render`.

**Test (`tests/modeling/test_topology_demo.py`):** build each of the three pipeline meshes, assert `to_geometry()` valid (indices in range, finite, unit normals).

**Design amendment:** in the Batch 3 line of the design doc, change `decimate` from "quadric-error-metric simplification" to "grid vertex-clustering (v1); QEM deferred".

**Steps:** render `uv run python examples/modeling_topology.py --render --duration 6 --fps 30 --output /tmp/topology.mp4`; `uv run pytest tests/modeling/ -v`; `make lint`. CHANGELOG under `### Features`. Commit `feat(modeling): topology demo (subdivide/extrude/decimate, lit + animated + shadows)`.

---

## Self-Review

**Coverage:** subdivide midpoint (T1) + loop (T2), extrude (T3), decimate (T4), demo (T5) — all Batch 3 design items. Decimate deviates to clustering with a documented amendment. **Types:** all ops `(mesh, ...) -> Mesh` with matching `Mesh` wrappers; `face_mask` is bool `(M,)`; `subdivide` scheme string. **Placeholders:** none — algorithms specified; impl developed against the tests above.
