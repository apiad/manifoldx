# Geometric Modeling v1 — Plan 2: Sculpt Brushes + Adjacency

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Batch 2 of `manifoldx.modeling` — a `Falloff` region model, vertex one-ring `adjacency`, and the programmatic sculpt brushes `draw`/`inflate`/`pinch`/`flatten`/`smooth`, composed fluently on `Mesh`.

**Architecture:** Brushes are `Mesh -> Mesh` operators that select a weighted vertex region via a radial `Falloff` and move those vertices (along normals, toward a point, toward a plane, or toward their neighbor-average). `smooth` is Laplacian relaxation over a lazily-built, cached one-ring adjacency (CSR). No renderer/ECS changes; everything stays host-side numpy.

**Tech Stack:** Python 3.13+, numpy, `uv`. Design: `.knowledge/analysis/2026-07-28-geometric-modeling-v1-design.md` (Batch 2). Builds on Plan 1 (`.knowledge/plans/2026-07-28-geometric-modeling-v1-plan-1-foundation-deformers.md`).

## Global Constraints

- Python 3.13+, all invocations via `uv run`.
- Dependency-free: pure numpy only.
- Brushes live in `src/manifoldx/modeling/sculpt.py` as `(mesh, ...) -> Mesh` functions; `Mesh` exposes thin fluent wrappers with lazy imports (matching how `deform`/`primitives` are wired).
- Every brush that moves vertices recomputes normals on its result (consistent with the deformers).
- Adjacency is built once and cached on the `Mesh` instance via `object.__setattr__` (the dataclass is frozen but not slotted). `replace()`-derived meshes rebuild it — correct, since their geometry differs.
- Conventional commits: `feat(modeling):`, `test(modeling):`.
- **Not in this plan:** topology ops (Batch 3), booleans (Batch 4), interactive/viewport sculpting.

## Consumed from Plan 1 (real, on `main`)

- `Mesh(positions (N,3) f32, faces (M,3) u32, normals|None, uvs|None)`.
- `Mesh.with_positions(positions) -> Mesh` (invalidates normals).
- `Mesh.recompute_normals() -> Mesh` (area-weighted, unit-length).
- `Mesh.icosphere/box/plane/cylinder/torus`, `Mesh.displace/twist/bend/taper`.

---

## File Structure

- Modify `src/manifoldx/modeling/mesh.py` — add `VertexAdjacency` + `Mesh.adjacency()`; add fluent methods `draw`/`inflate`/`pinch`/`flatten`/`smooth`.
- Create `src/manifoldx/modeling/sculpt.py` — `Falloff` + the five brush functions.
- Modify `src/manifoldx/modeling/__init__.py` — re-export `Falloff`.
- Create `tests/modeling/test_sculpt.py`.
- Modify `examples/modeling_asteroid.py` — add a `.smooth(iterations=1)` polish pass.
- Modify `CHANGELOG.md`.

---

### Task 1: Vertex one-ring adjacency

**Files:**
- Modify: `src/manifoldx/modeling/mesh.py`
- Test: `tests/modeling/test_sculpt.py`

**Interfaces:**
- Produces:
  - `VertexAdjacency(offsets: np.ndarray (N+1,) int64, neighbors: np.ndarray (E,) int32)` — CSR one-ring: vertex `i`'s neighbors are `neighbors[offsets[i]:offsets[i+1]]`.
  - `Mesh.adjacency() -> VertexAdjacency` — computed once, cached on the instance.

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_sculpt.py
import numpy as np
from manifoldx.modeling import Mesh


def test_adjacency_of_single_triangle():
    m = Mesh(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    adj = m.adjacency()
    assert adj.offsets.shape == (4,)
    # Every vertex of a triangle neighbors the other two.
    for i in range(3):
        nbrs = set(adj.neighbors[adj.offsets[i]:adj.offsets[i + 1]].tolist())
        assert nbrs == {j for j in range(3) if j != i}


def test_adjacency_is_cached():
    m = Mesh.icosphere(subdivisions=2)
    assert m.adjacency() is m.adjacency()   # same object, built once


def test_adjacency_symmetric_on_icosphere():
    m = Mesh.icosphere(subdivisions=1)
    adj = m.adjacency()
    pairs = set()
    for i in range(len(m.positions)):
        for j in adj.neighbors[adj.offsets[i]:adj.offsets[i + 1]]:
            pairs.add((i, int(j)))
    assert all((j, i) in pairs for (i, j) in pairs)   # undirected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_sculpt.py -k adjacency -v`
Expected: FAIL — `AttributeError: 'Mesh' object has no attribute 'adjacency'`.

- [ ] **Step 3: Write minimal implementation**

Add to `mesh.py`, above the `Mesh` class:

```python
from typing import NamedTuple


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
```

Add to the `Mesh` class (after `from_geometry`):

```python
    def adjacency(self) -> "VertexAdjacency":
        cached = getattr(self, "_adjacency_cache", None)
        if cached is None:
            cached = _build_adjacency(self.faces, len(self.positions))
            object.__setattr__(self, "_adjacency_cache", cached)
        return cached
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_sculpt.py -k adjacency -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/mesh.py tests/modeling/test_sculpt.py
git commit -m "feat(modeling): one-ring vertex adjacency (CSR, cached)"
```

---

### Task 2: `Falloff` + `draw` brush

**Files:**
- Create: `src/manifoldx/modeling/sculpt.py`
- Modify: `src/manifoldx/modeling/mesh.py` (add `Mesh.draw`)
- Modify: `src/manifoldx/modeling/__init__.py` (re-export `Falloff`)
- Test: `tests/modeling/test_sculpt.py` (append)

**Interfaces:**
- Produces:
  - `Falloff(center, radius, profile="smooth")` with `.weights(positions (N,3)) -> (N,) f32` in [0,1]; 0 outside `radius`; `profile` in `{"smooth", "linear", "constant"}`.
  - `sculpt.draw(mesh, center, radius, strength, profile="smooth") -> Mesh` — displaces selected vertices along the **average region normal** (single direction) by `strength * weight`.
  - `Mesh.draw(center, radius, strength, profile="smooth") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_sculpt.py
from manifoldx.modeling import Falloff


def test_falloff_weights_bounds():
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    w = Falloff(center=(0, 0, 0), radius=1.0).weights(pts)
    assert w[0] == 1.0                 # at center
    assert w[2] == 0.0 and w[3] == 0.0  # at/outside radius
    assert 0.0 < w[1] < 1.0            # partway
    assert w.dtype == np.float32


def test_draw_raises_bump_near_center_only():
    base = Mesh.plane(width=4, depth=4, segments=20)
    out = base.draw(center=(0, 0, 0), radius=1.0, strength=0.5)
    disp = out.positions - base.positions
    d = np.linalg.norm(base.positions - np.array([0, 0, 0]), axis=1)
    assert np.abs(disp[d < 0.2]).max() > 0.1      # bump near center
    assert np.allclose(disp[d > 1.5], 0.0)        # untouched far away
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_sculpt.py -k "falloff or draw" -v`
Expected: FAIL — `ImportError: cannot import name 'Falloff'` / `AttributeError: ... 'draw'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/manifoldx/modeling/sculpt.py
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
```

Add to `Mesh`:

```python
    def draw(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.draw(self, center, radius, strength, profile)
```

Add to `__init__.py`:

```python
from manifoldx.modeling.sculpt import Falloff
```
and add `"Falloff"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_sculpt.py -k "falloff or draw" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/sculpt.py src/manifoldx/modeling/mesh.py src/manifoldx/modeling/__init__.py tests/modeling/test_sculpt.py
git commit -m "feat(modeling): Falloff region + draw brush"
```

---

### Task 3: `inflate` brush

**Files:**
- Modify: `src/manifoldx/modeling/sculpt.py`, `src/manifoldx/modeling/mesh.py`
- Test: `tests/modeling/test_sculpt.py` (append)

**Interfaces:**
- Produces:
  - `sculpt.inflate(mesh, center, radius, strength, profile="smooth") -> Mesh` — like `draw`, but each selected vertex moves along **its own** normal (per-vertex), so it puffs the region out rather than translating it rigidly.
  - `Mesh.inflate(center, radius, strength, profile="smooth") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_sculpt.py
def test_inflate_pushes_along_vertex_normals():
    base = Mesh.icosphere(subdivisions=3, radius=1.0)
    out = base.inflate(center=(0, 0, 1), radius=0.6, strength=0.3)
    r_in = np.linalg.norm(base.positions, axis=1)
    r_out = np.linalg.norm(out.positions, axis=1)
    near_pole = base.positions[:, 2] > 0.7
    assert (r_out[near_pole] > r_in[near_pole] + 0.05).any()   # bulges outward
    far = base.positions[:, 2] < 0.0
    assert np.allclose(r_out[far], r_in[far], atol=1e-4)       # untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_sculpt.py -k inflate -v`
Expected: FAIL — `AttributeError: ... 'inflate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to sculpt.py
def inflate(mesh: Mesh, center, radius: float, strength: float, profile: str = "smooth") -> Mesh:
    w, base = _selected(mesh, center, radius, profile)
    out = mesh.positions + (strength * w)[:, None] * base.normals
    return mesh.with_positions(out).recompute_normals()
```

Add to `Mesh`:

```python
    def inflate(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.inflate(self, center, radius, strength, profile)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_sculpt.py -k inflate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/sculpt.py src/manifoldx/modeling/mesh.py tests/modeling/test_sculpt.py
git commit -m "feat(modeling): inflate brush (per-vertex normal)"
```

---

### Task 4: `pinch` brush

**Files:**
- Modify: `src/manifoldx/modeling/sculpt.py`, `src/manifoldx/modeling/mesh.py`
- Test: `tests/modeling/test_sculpt.py` (append)

**Interfaces:**
- Produces:
  - `sculpt.pinch(mesh, center, radius, strength, profile="smooth") -> Mesh` — moves selected vertices toward `center` by a fraction `strength * weight` of their offset from center.
  - `Mesh.pinch(center, radius, strength, profile="smooth") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_sculpt.py
def test_pinch_pulls_vertices_toward_center():
    base = Mesh.plane(width=4, depth=4, segments=20)
    center = (0.0, 0.0, 0.0)
    out = base.pinch(center=center, radius=1.5, strength=0.5)
    c = np.asarray(center)
    d_in = np.linalg.norm(base.positions - c, axis=1)
    d_out = np.linalg.norm(out.positions - c, axis=1)
    sel = (d_in > 0.1) & (d_in < 1.0)
    assert np.all(d_out[sel] < d_in[sel])          # closer to center
    assert np.allclose(d_out[d_in > 2.0], d_in[d_in > 2.0])  # untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_sculpt.py -k pinch -v`
Expected: FAIL — `AttributeError: ... 'pinch'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to sculpt.py
def pinch(mesh: Mesh, center, radius: float, strength: float, profile: str = "smooth") -> Mesh:
    w = Falloff(center, radius, profile).weights(mesh.positions)
    c = np.asarray(center, dtype=np.float32).reshape(3)
    to_center = c[None, :] - mesh.positions
    out = mesh.positions + (strength * w)[:, None] * to_center
    return mesh.with_positions(out).recompute_normals()
```

Add to `Mesh`:

```python
    def pinch(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.pinch(self, center, radius, strength, profile)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_sculpt.py -k pinch -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/sculpt.py src/manifoldx/modeling/mesh.py tests/modeling/test_sculpt.py
git commit -m "feat(modeling): pinch brush (pull toward center)"
```

---

### Task 5: `flatten` brush

**Files:**
- Modify: `src/manifoldx/modeling/sculpt.py`, `src/manifoldx/modeling/mesh.py`
- Test: `tests/modeling/test_sculpt.py` (append)

**Interfaces:**
- Produces:
  - `sculpt.flatten(mesh, center, radius, strength, profile="smooth") -> Mesh` — projects selected vertices toward the best-fit plane of the selected region (plane point = weighted centroid, plane normal = weighted average vertex normal), by fraction `strength * weight` of their signed distance to the plane.
  - `Mesh.flatten(center, radius, strength, profile="smooth") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_sculpt.py
def test_flatten_reduces_height_variation_in_region():
    base = Mesh.icosphere(subdivisions=3, radius=1.0)
    # Flatten the +x cap toward its own plane.
    out = base.flatten(center=(1, 0, 0), radius=0.7, strength=1.0)
    sel = base.positions[:, 0] > 0.7
    # Radial spread (distance from x-axis) of the selected cap shrinks as it flattens.
    spread_in = np.linalg.norm(base.positions[sel][:, [1, 2]], axis=1).std()
    spread_out = np.linalg.norm(out.positions[sel][:, [1, 2]], axis=1).std()
    assert out.positions.shape == base.positions.shape
    assert spread_out <= spread_in + 1e-4
    far = base.positions[:, 0] < 0.0
    assert np.allclose(out.positions[far], base.positions[far], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_sculpt.py -k flatten -v`
Expected: FAIL — `AttributeError: ... 'flatten'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to sculpt.py
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
```

Add to `Mesh`:

```python
    def flatten(self, center, radius: float, strength: float, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.flatten(self, center, radius, strength, profile)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_sculpt.py -k flatten -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/sculpt.py src/manifoldx/modeling/mesh.py tests/modeling/test_sculpt.py
git commit -m "feat(modeling): flatten brush (project region to best-fit plane)"
```

---

### Task 6: `smooth` (Laplacian relaxation)

**Files:**
- Modify: `src/manifoldx/modeling/sculpt.py`, `src/manifoldx/modeling/mesh.py`
- Test: `tests/modeling/test_sculpt.py` (append)

**Interfaces:**
- Produces:
  - `sculpt.smooth(mesh, iterations=1, strength=1.0, center=None, radius=None, profile="smooth") -> Mesh` — moves each vertex toward its one-ring neighbor average. Global when `center`/`radius` are `None`; when both given, the per-vertex step is scaled by the `Falloff` weight (local smoothing). Uses `mesh.adjacency()`.
  - `Mesh.smooth(iterations=1, strength=1.0, center=None, radius=None, profile="smooth") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_sculpt.py
def test_smooth_reduces_roughness_globally():
    rough = Mesh.icosphere(subdivisions=3).displace(
        __import__("manifoldx.modeling", fromlist=["noise"]).noise.fbm(seed=5, octaves=5),
        amount=0.25,
    )

    def roughness(m):
        adj = m.adjacency()
        deg = np.diff(adj.offsets)
        src = np.repeat(np.arange(len(m.positions)), deg)
        sums = np.zeros_like(m.positions)
        np.add.at(sums, src, m.positions[adj.neighbors])
        avg = sums / np.maximum(deg, 1)[:, None]
        return np.linalg.norm(m.positions - avg, axis=1).mean()

    smoothed = rough.smooth(iterations=3)
    assert smoothed.positions.shape == rough.positions.shape
    assert roughness(smoothed) < roughness(rough)


def test_smooth_zero_iterations_is_identity():
    m = Mesh.icosphere(subdivisions=2)
    assert np.allclose(m.smooth(iterations=0).positions, m.positions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_sculpt.py -k smooth -v`
Expected: FAIL — `AttributeError: ... 'smooth'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to sculpt.py
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
```

Add to `Mesh`:

```python
    def smooth(self, iterations: int = 1, strength: float = 1.0,
               center=None, radius=None, profile: str = "smooth") -> "Mesh":
        from manifoldx.modeling import sculpt
        return sculpt.smooth(self, iterations, strength, center, radius, profile)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_sculpt.py -k smooth -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/sculpt.py src/manifoldx/modeling/mesh.py tests/modeling/test_sculpt.py
git commit -m "feat(modeling): smooth (Laplacian, global + local via falloff)"
```

---

### Task 7: Asteroid polish pass + CHANGELOG

**Files:**
- Modify: `examples/modeling_asteroid.py`
- Modify: `tests/modeling/test_asteroid_demo.py` (append)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `Mesh.smooth`.

Add a light Laplacian relaxation as the asteroid's final step (relaxes the sharpest fbm spikes without washing out the form).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_asteroid_demo.py
def test_asteroid_with_smooth_pass_valid():
    rock = (
        Mesh.icosphere(subdivisions=4)
        .displace(noise.fbm(seed=7, octaves=5), amount=0.35)
        .twist(angle=0.4, axis="y")
        .taper(factor=0.2, axis="y")
        .smooth(iterations=1, strength=0.5)
    )
    geo = rock.to_geometry()
    assert np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_asteroid_demo.py -k smooth_pass -v`
Expected: PASS if Task 6 is done (guards the composition). Fails only on a real `smooth` regression.

- [ ] **Step 3: Update the demo pipeline**

Append `.smooth(iterations=1, strength=0.5)` to the `rock = (...)` chain in `examples/modeling_asteroid.py`, after `.taper(...)`.

- [ ] **Step 4: Render + full suite + lint**

Run: `uv run python examples/modeling_asteroid.py --render --duration 3 --fps 30 --output /tmp/asteroid.mp4` (skip + record if no GPU backend).
Run: `uv run pytest tests/modeling/ -v` → all PASS.
Run: `make lint` → clean.

- [ ] **Step 5: Commit + CHANGELOG**

Under `## [Unreleased]` → `### Features`, extend the geometric-modeling entry (or add a sibling bullet) noting Batch 2: `Falloff` + `draw`/`inflate`/`pinch`/`flatten`/`smooth` brushes and cached one-ring `adjacency`.

```bash
git add examples/modeling_asteroid.py tests/modeling/test_asteroid_demo.py CHANGELOG.md
git commit -m "feat(modeling): asteroid smooth polish pass + batch-2 changelog"
```

---

## Self-Review

**Spec coverage (design Batch 2):** `Falloff` region → Task 2. `draw`/`inflate`/`pinch`/`flatten` → Tasks 2–5. `smooth` (Laplacian, needs adjacency) → Tasks 1, 6. Adjacency (deferred from Batch 0) → Task 1. ✓

**Placeholder scan:** every code + test step has real content. ✓

**Type consistency:** brushes are `(mesh, center, radius, strength, profile="smooth") -> Mesh`; `smooth` adds `iterations`/`center`/`radius`; `Falloff(center, radius, profile).weights(positions) -> (N,) f32` used identically in every brush; `VertexAdjacency(offsets, neighbors)` consumed the same way in `smooth` and the test's `roughness` helper. All `Mesh` wrappers lazy-import `sculpt`, matching the Plan-1 `deform` wiring. ✓
