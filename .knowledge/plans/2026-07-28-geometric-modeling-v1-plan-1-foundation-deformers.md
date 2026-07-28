# Geometric Modeling v1 — Plan 1: Foundation + Deformers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `manifoldx.modeling` subsystem (Batch 0 foundation + Batch 1 deformers) so a procedural mesh can be built in code and rendered, culminating in `examples/modeling_asteroid.py`.

**Architecture:** An immutable `modeling.Mesh` value type over numpy arrays. Primitives generate a `Mesh`; deformers are `Mesh -> Mesh` functions exposed as fluent methods. The single integration seam is `Mesh.to_geometry()` → the geometry dict the ECS `Mesh` component + `GeometryRegistry` already consume. No renderer/ECS changes.

**Tech Stack:** Python 3.13+, numpy, `uv`. Design: `.knowledge/analysis/2026-07-28-geometric-modeling-v1-design.md`.

## Global Constraints

- Python 3.13+, all invocations via `uv run` (`uv run pytest`, `uv run python examples/...`).
- Dependency-free: pure numpy only. No new third-party dependency in this plan.
- The host-side value type is `manifoldx.modeling.Mesh`, **never** re-exported at top level (the ECS component `manifoldx.Mesh` keeps that name).
- Conventional commits: `feat(modeling):`, `test(modeling):`, `docs(modeling):`.
- Faces stored internally as `(M, 3)` uint32 triangles; flattened to a 1-D `indices` stream only at `to_geometry()`.
- Randomness (noise) seeds via a `seed` arg resolved exactly like `manifoldx.random._resolve_rng` (None → `default_rng()`, int → `default_rng(seed)`, Generator → itself). Same seed ⇒ identical output.
- GPU/render tests gate on `manifoldx.backends.get_offscreen_canvas` and `pytest.skip` if unavailable (repo convention).
- **Deferred to later plans (do NOT implement here):** vertex `adjacency` + `smooth` (Batch 2, first consumer of adjacency), sculpt brushes (Batch 2), topology ops (Batch 3), booleans (Batch 4), and the Batch-1 tail `ffd`/`bend_along_curve`.

---

## File Structure

- Create `src/manifoldx/modeling/__init__.py` — re-exports `Mesh`, `noise`.
- Create `src/manifoldx/modeling/mesh.py` — `Mesh` dataclass + core (`with_positions`, `recompute_normals`, `to_geometry`, `from_geometry`) + thin operator/primitive methods (lazy-import delegation).
- Create `src/manifoldx/modeling/primitives.py` — `icosphere`, `box`, `plane`, `cylinder`, `torus` generators returning `Mesh`.
- Create `src/manifoldx/modeling/noise.py` — `perlin`, `fbm` seeded field callables.
- Create `src/manifoldx/modeling/deform.py` — `displace`, `twist`, `bend`, `taper` (`Mesh -> Mesh`).
- Create `examples/modeling_asteroid.py` — the seam/smoke demo, grows across Batch 0 → Batch 1.
- Create tests under `tests/modeling/`: `test_mesh.py`, `test_primitives.py`, `test_noise.py`, `test_deform.py`, `test_asteroid_demo.py`.

---

## BATCH 0 — Foundation

### Task 1: `Mesh` value type core

**Files:**
- Create: `src/manifoldx/modeling/mesh.py`
- Create: `src/manifoldx/modeling/__init__.py`
- Test: `tests/modeling/test_mesh.py`

**Interfaces:**
- Produces:
  - `Mesh(positions: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None, uvs: np.ndarray | None = None)` — frozen dataclass. `positions (N,3) float32`, `faces (M,3) uint32`.
  - `Mesh.with_positions(positions) -> Mesh` — copy with positions replaced, `normals=None` (invalidated).
  - `Mesh.recompute_normals() -> Mesh` — area-weighted vertex normals, unit-length.
  - `Mesh.to_geometry() -> dict` — `{"positions": (N,3) f32, "normals": (N,3) f32, "indices": (3M,) u32[, "uvs": (N,2) f32]}`. Auto-computes normals if `None`.
  - `Mesh.from_geometry(geo: dict) -> Mesh` — static; inverse of `to_geometry`, accepting `positions`/`normals`/`uvs`/`indices` (reshapes `indices` to `(M,3)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_mesh.py
import numpy as np
from manifoldx.modeling import Mesh


def _unit_triangle():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    return Mesh(positions=positions, faces=faces)


def test_with_positions_invalidates_normals():
    m = _unit_triangle().recompute_normals()
    assert m.normals is not None
    moved = m.with_positions(m.positions + np.array([0, 0, 1], dtype=np.float32))
    assert moved.normals is None
    assert np.allclose(moved.positions[:, 2], 1.0)


def test_recompute_normals_unit_length_and_direction():
    m = _unit_triangle().recompute_normals()
    lengths = np.linalg.norm(m.normals, axis=1)
    assert np.allclose(lengths, 1.0)
    # A CCW triangle in the z=0 plane faces +z.
    assert np.allclose(m.normals[0], [0, 0, 1], atol=1e-5)


def test_to_geometry_shapes_and_dtypes():
    geo = _unit_triangle().to_geometry()
    assert geo["positions"].dtype == np.float32 and geo["positions"].shape == (3, 3)
    assert geo["normals"].dtype == np.float32 and geo["normals"].shape == (3, 3)
    assert geo["indices"].dtype == np.uint32 and geo["indices"].shape == (3,)
    assert list(geo["indices"]) == [0, 1, 2]


def test_from_geometry_roundtrip():
    m = _unit_triangle().recompute_normals()
    back = Mesh.from_geometry(m.to_geometry())
    assert np.allclose(back.positions, m.positions)
    assert back.faces.shape == (1, 3)
    assert np.array_equal(back.faces[0], [0, 1, 2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_mesh.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'manifoldx.modeling'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/manifoldx/modeling/mesh.py
"""Host-side, immutable procedural mesh value type."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


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
        lengths[lengths == 0] = 1.0
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
```

```python
# src/manifoldx/modeling/__init__.py
"""Procedural geometric modeling: numpy-first Mesh value type + operator pipeline."""

from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling import noise

__all__ = ["Mesh", "noise"]
```

Note: the `noise` import will fail until Task 5. For Task 1, temporarily set `__init__.py` to only `from manifoldx.modeling.mesh import Mesh` / `__all__ = ["Mesh"]`; restore the `noise` line in Task 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_mesh.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/ tests/modeling/test_mesh.py
git commit -m "feat(modeling): Mesh value type core (normals, to/from_geometry)"
```

---

### Task 2: `icosphere` primitive

**Files:**
- Create: `src/manifoldx/modeling/primitives.py`
- Modify: `src/manifoldx/modeling/mesh.py` (add `Mesh.icosphere` classmethod)
- Test: `tests/modeling/test_primitives.py`

**Interfaces:**
- Consumes: `Mesh` (Task 1).
- Produces:
  - `primitives.icosphere(subdivisions: int = 2, radius: float = 1.0) -> Mesh`.
  - `Mesh.icosphere(subdivisions=2, radius=1.0) -> Mesh` (classmethod delegating to `primitives`).

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_primitives.py
import numpy as np
from manifoldx.modeling import Mesh


def test_icosphere_face_count_grows_by_four():
    base = Mesh.icosphere(subdivisions=0)
    assert base.faces.shape[0] == 20            # icosahedron
    one = Mesh.icosphere(subdivisions=1)
    assert one.faces.shape[0] == 80             # x4 per subdivision


def test_icosphere_vertices_on_radius():
    m = Mesh.icosphere(subdivisions=2, radius=2.0)
    r = np.linalg.norm(m.positions, axis=1)
    assert np.allclose(r, 2.0, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_primitives.py -v`
Expected: FAIL — `AttributeError: type object 'Mesh' has no attribute 'icosphere'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/manifoldx/modeling/primitives.py
"""Procedural primitive generators returning modeling.Mesh values."""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> Mesh:
    """Geodesic sphere from a subdivided icosahedron (uniform-ish triangles)."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(subdivisions):
        verts, faces = _subdivide_midpoint(verts, faces)

    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return Mesh(
        positions=verts.astype(np.float32),
        faces=faces.astype(np.uint32),
    )


def _subdivide_midpoint(verts: np.ndarray, faces: np.ndarray):
    """Split each triangle into 4 via edge midpoints, sharing midpoints across edges."""
    verts = list(map(tuple, verts))
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        va, vb = np.asarray(verts[a]), np.asarray(verts[b])
        verts.append(tuple((va + vb) / 2.0))
        cache[key] = len(verts) - 1
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)
```

Add the classmethod to `Mesh` (in `mesh.py`, inside the class):

```python
    @classmethod
    def icosphere(cls, subdivisions: int = 2, radius: float = 1.0) -> "Mesh":
        from manifoldx.modeling import primitives
        return primitives.icosphere(subdivisions, radius)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_primitives.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/primitives.py src/manifoldx/modeling/mesh.py tests/modeling/test_primitives.py
git commit -m "feat(modeling): icosphere primitive"
```

---

### Task 3: Seam demo — `examples/modeling_asteroid.py` (bare icosphere)

**Files:**
- Create: `examples/modeling_asteroid.py`
- Test: `tests/modeling/test_asteroid_demo.py`

**Interfaces:**
- Consumes: `Mesh.icosphere`, `Mesh.to_geometry`, `mx.material.standard`, `PointLight`, `engine.set_lights`, `engine.spawn`.

This task proves the `modeling → GPU` seam end-to-end before any operator exists. The demo builds a bare icosphere and spawns it PBR-lit and rotating.

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_asteroid_demo.py
import numpy as np
from manifoldx.modeling import Mesh


def test_icosphere_geometry_is_spawn_ready():
    """The demo's geometry dict must satisfy the GeometryRegistry contract."""
    geo = Mesh.icosphere(subdivisions=3).to_geometry()
    assert set(geo) >= {"positions", "normals", "indices"}
    assert geo["positions"].shape[1] == 3
    assert geo["normals"].shape == geo["positions"].shape
    assert geo["indices"].ndim == 1 and geo["indices"].shape[0] % 3 == 0
    assert geo["indices"].max() < geo["positions"].shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_asteroid_demo.py -v`
Expected: PASS actually is possible here since it only uses Task 1–2 APIs. If it fails it is a real regression in Task 1/2 — fix there. (This test guards the demo's contract; the demo file itself is exercised by the render smoke test below.)

- [ ] **Step 3: Write the demo**

```python
# examples/modeling_asteroid.py
"""Procedural asteroid — smoke test + showcase for manifoldx.modeling.

Batch 0: a bare icosphere, proving the modeling -> GPU seam.
Batch 1 (Task 10) upgrades this in place into a noise-displaced asteroid.

Render a clip:
    uv run python examples/modeling_asteroid.py --render --duration 3 --fps 30 \
        --output /tmp/asteroid.mp4
"""

import math

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import PointLight
from manifoldx.systems import Query

engine = mx.Engine("Asteroid")

rock = GeoMesh.icosphere(subdivisions=4)
rock_geometry = rock.to_geometry()
rock_material = mx.material.standard(color="#8c8073", roughness=0.9)  # hex string, per pbr_demo

engine.set_lights([
    PointLight(color="#fff2e0", intensity=6.0, position=(4, 5, 4)),
    PointLight(color="#8090ff", intensity=2.0, position=(-4, -2, -3)),
])


@engine.on("startup")
def create_asteroid(_payload):
    engine.spawn(
        Mesh(rock_geometry),
        Material(rock_material),
        Transform(pos=(0, 0, 0)),
        n=1,
    )


@engine.system
def spin(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=dt * 0.2, y=dt * 0.5, z=0)


if __name__ == "__main__":
    engine.cli()
```

Color is a hex string (`"#8c8073"`), matching `examples/pbr_demo.py` (`StandardMaterial(color="#ff3333", ...)`); there is no `mx.colors.rgb()` helper. `mx.material.standard(color, roughness, metallic)` and `PointLight(color=, intensity=, position=)` are both confirmed on `main`.

- [ ] **Step 4: Run the render smoke test**

Run: `uv run python examples/modeling_asteroid.py --render --duration 2 --fps 24 --output /tmp/asteroid.mp4`
Expected: exits 0, writes `/tmp/asteroid.mp4` (a rotating grey sphere). If the wgpu backend is unavailable on the machine, this step is skipped — record that. Then run the contract test:
Run: `uv run pytest tests/modeling/test_asteroid_demo.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/modeling_asteroid.py tests/modeling/test_asteroid_demo.py
git commit -m "feat(modeling): asteroid seam demo (bare icosphere) + geometry contract test"
```

---

### Task 4: Remaining primitives — `box`, `plane`, `cylinder`, `torus`

**Files:**
- Modify: `src/manifoldx/modeling/primitives.py`
- Modify: `src/manifoldx/modeling/mesh.py` (classmethods `box`, `plane`, `cylinder`, `torus`)
- Test: `tests/modeling/test_primitives.py` (append)

**Interfaces:**
- Produces (module fns + `Mesh` classmethods delegating to them):
  - `box(width=1.0, height=1.0, depth=1.0) -> Mesh`
  - `plane(width=1.0, depth=1.0, segments=1) -> Mesh` (subdividable grid in the XZ plane, +Y normals)
  - `cylinder(radius=1.0, height=2.0, segments=32) -> Mesh`
  - `torus(major=1.0, minor=0.35, major_segments=32, minor_segments=16) -> Mesh`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_primitives.py
def test_box_is_closed_triangle_soup():
    m = Mesh.box(2, 1, 1)
    assert m.faces.shape[1] == 3
    assert m.faces.shape[0] == 12           # 6 faces x 2 tris
    assert m.positions[:, 0].max() == 1.0   # width 2 -> [-1, 1]


def test_plane_segments_grid_counts():
    m = Mesh.plane(width=1.0, depth=1.0, segments=2)
    assert m.positions.shape[0] == 9        # (segments+1)^2
    assert m.faces.shape[0] == 8            # segments^2 * 2


def test_cylinder_and_torus_watertight_vertex_counts():
    cyl = Mesh.cylinder(radius=1.0, height=2.0, segments=8)
    assert cyl.positions.shape[0] > 0 and cyl.faces.shape[1] == 3
    tor = Mesh.torus(major=1.0, minor=0.3, major_segments=8, minor_segments=6)
    assert tor.faces.shape[0] == 8 * 6 * 2  # quad per (i,j) -> 2 tris
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_primitives.py -v`
Expected: FAIL — `AttributeError: type object 'Mesh' has no attribute 'box'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/manifoldx/modeling/primitives.py

def box(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> Mesh:
    hx, hy, hz = width / 2, height / 2, depth / 2
    corners = np.array(
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
         [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
        dtype=np.float32,
    )
    quads = [
        (0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7),
        (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0),
    ]
    faces = []
    for a, b, c, d in quads:
        faces += [[a, b, c], [a, c, d]]
    return Mesh(positions=corners, faces=np.array(faces, dtype=np.uint32))


def plane(width: float = 1.0, depth: float = 1.0, segments: int = 1) -> Mesh:
    n = segments + 1
    xs = np.linspace(-width / 2, width / 2, n)
    zs = np.linspace(-depth / 2, depth / 2, n)
    gx, gz = np.meshgrid(xs, zs)
    positions = np.stack([gx.ravel(), np.zeros(n * n), gz.ravel()], axis=1).astype(np.float32)
    faces = []
    for i in range(segments):
        for j in range(segments):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces += [[a, c, b], [b, c, d]]
    return Mesh(positions=positions, faces=np.array(faces, dtype=np.uint32))


def cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 32) -> Mesh:
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(ang) * radius, np.sin(ang) * radius], axis=1)
    top = np.column_stack([ring[:, 0], np.full(segments, height / 2), ring[:, 1]])
    bot = np.column_stack([ring[:, 0], np.full(segments, -height / 2), ring[:, 1]])
    center_top = [0, height / 2, 0]
    center_bot = [0, -height / 2, 0]
    positions = np.vstack([top, bot, center_top, center_bot]).astype(np.float32)
    ct, cb = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        n = (i + 1) % segments
        faces += [[i, n, segments + i], [n, segments + n, segments + i]]  # side
        faces += [[ct, n, i]]                                             # top cap
        faces += [[cb, segments + i, segments + n]]                       # bottom cap
    return Mesh(positions=positions, faces=np.array(faces, dtype=np.uint32))


def torus(major: float = 1.0, minor: float = 0.35,
          major_segments: int = 32, minor_segments: int = 16) -> Mesh:
    u = np.linspace(0, 2 * np.pi, major_segments, endpoint=False)
    v = np.linspace(0, 2 * np.pi, minor_segments, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = (major + minor * np.cos(vv)) * np.cos(uu)
    y = minor * np.sin(vv)
    z = (major + minor * np.cos(vv)) * np.sin(uu)
    positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
    faces = []
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = ((i + 1) % major_segments) * minor_segments + j
            c = i * minor_segments + (j + 1) % minor_segments
            d = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            faces += [[a, b, c], [b, d, c]]
    return Mesh(positions=positions, faces=np.array(faces, dtype=np.uint32))
```

Add the four classmethods to `Mesh` (lazy-import delegation, mirroring `icosphere`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_primitives.py -v`
Expected: PASS (all primitive tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/primitives.py src/manifoldx/modeling/mesh.py tests/modeling/test_primitives.py
git commit -m "feat(modeling): box/plane/cylinder/torus primitives"
```

---

## BATCH 1 — Deformers

### Task 5: `noise` — seeded Perlin + fbm field callables

**Files:**
- Create: `src/manifoldx/modeling/noise.py`
- Modify: `src/manifoldx/modeling/__init__.py` (restore the `noise` re-export)
- Test: `tests/modeling/test_noise.py`

**Interfaces:**
- Produces:
  - `noise.perlin(seed=None, freq: float = 1.0) -> Callable[[np.ndarray], np.ndarray]` — the callable maps `points (K,3) float` → `values (K,) float32` in ~[-1, 1].
  - `noise.fbm(seed=None, freq=1.0, octaves=4, lacunarity=2.0, gain=0.5) -> Callable[[np.ndarray], np.ndarray]` — summed octaves of `perlin`.
  - Seed resolution matches `manifoldx.random._resolve_rng` (None/int/Generator).

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_noise.py
import numpy as np
from manifoldx.modeling import noise


def _grid(k=200):
    rng = np.random.default_rng(0)
    return rng.uniform(-3, 3, size=(k, 3)).astype(np.float32)


def test_perlin_deterministic_same_seed():
    pts = _grid()
    a = noise.perlin(seed=42)(pts)
    b = noise.perlin(seed=42)(pts)
    assert np.array_equal(a, b)


def test_perlin_differs_across_seeds():
    pts = _grid()
    a = noise.perlin(seed=1)(pts)
    b = noise.perlin(seed=2)(pts)
    assert not np.allclose(a, b)


def test_perlin_range_and_shape():
    pts = _grid(500)
    vals = noise.perlin(seed=7)(pts)
    assert vals.shape == (500,)
    assert vals.min() >= -1.5 and vals.max() <= 1.5  # gradient noise stays bounded


def test_fbm_deterministic_and_shaped():
    pts = _grid()
    a = noise.fbm(seed=3, octaves=4)(pts)
    b = noise.fbm(seed=3, octaves=4)(pts)
    assert np.array_equal(a, b) and a.shape == (pts.shape[0],)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_noise.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'manifoldx.modeling.noise'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/manifoldx/modeling/noise.py
"""Seeded, deterministic value/gradient-noise fields for displacement.

A field is a callable: points (K, 3) -> values (K,) float32 in ~[-1, 1].
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _resolve_rng(seed) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def perlin(seed=None, freq: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Classic Perlin gradient noise in 3D, seeded via a permutation table."""
    rng = _resolve_rng(seed)
    perm = rng.permutation(256).astype(np.int32)
    perm = np.concatenate([perm, perm])  # doubled to avoid overflow indexing

    # 12 canonical gradient directions.
    grad3 = np.array(
        [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
         [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
         [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]],
        dtype=np.float32,
    )

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def grad(ix, iy, iz, dx, dy, dz):
        h = perm[perm[perm[ix & 255] + (iy & 255)] + (iz & 255)] % 12
        g = grad3[h]
        return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

    def field(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64) * freq
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        xi, yi, zi = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), np.floor(z).astype(np.int32)
        xf, yf, zf = x - xi, y - yi, z - zi
        u, v, w = fade(xf), fade(yf), fade(zf)

        def lerp(a, b, t):
            return a + t * (b - a)

        n000 = grad(xi, yi, zi, xf, yf, zf)
        n100 = grad(xi + 1, yi, zi, xf - 1, yf, zf)
        n010 = grad(xi, yi + 1, zi, xf, yf - 1, zf)
        n110 = grad(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf)
        n001 = grad(xi, yi, zi + 1, xf, yf, zf - 1)
        n101 = grad(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1)
        n011 = grad(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1)
        n111 = grad(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1)

        x00 = lerp(n000, n100, u)
        x10 = lerp(n010, n110, u)
        x01 = lerp(n001, n101, u)
        x11 = lerp(n011, n111, u)
        y0 = lerp(x00, x10, v)
        y1 = lerp(x01, x11, v)
        return lerp(y0, y1, w).astype(np.float32)

    return field


def fbm(seed=None, freq: float = 1.0, octaves: int = 4,
        lacunarity: float = 2.0, gain: float = 0.5) -> Callable[[np.ndarray], np.ndarray]:
    """Fractal Brownian motion: summed octaves of `perlin`."""
    rng = _resolve_rng(seed)
    layers = [(perlin(seed=rng, freq=freq * lacunarity**i), gain**i) for i in range(octaves)]
    norm = sum(a for _, a in layers)

    def field(points: np.ndarray) -> np.ndarray:
        total = np.zeros(len(points), dtype=np.float32)
        for f, amp in layers:
            total += amp * f(points)
        return (total / norm).astype(np.float32)

    return field
```

Restore the `noise` re-export in `__init__.py` (added back now that the module exists):

```python
from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling import noise

__all__ = ["Mesh", "noise"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_noise.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/noise.py src/manifoldx/modeling/__init__.py tests/modeling/test_noise.py
git commit -m "feat(modeling): seeded perlin + fbm noise fields"
```

---

### Task 6: `displace(field, amount, along)` deformer

**Files:**
- Create: `src/manifoldx/modeling/deform.py`
- Modify: `src/manifoldx/modeling/mesh.py` (add `Mesh.displace`)
- Test: `tests/modeling/test_deform.py`

**Interfaces:**
- Consumes: `Mesh` (Task 1), a field callable (Task 5).
- Produces:
  - `deform.displace(mesh, field, amount=1.0, along="normal") -> Mesh` — offsets each vertex by `amount * field(positions)` along its vertex normal (`along="normal"`) or a fixed 3-vector `along=(x,y,z)`. Recomputes normals on the result.
  - `Mesh.displace(field, amount=1.0, along="normal") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_deform.py
import numpy as np
from manifoldx.modeling import Mesh, noise


def test_displace_changes_positions_and_is_deterministic():
    base = Mesh.icosphere(subdivisions=3)
    field = noise.fbm(seed=11, octaves=4)
    a = base.displace(field, amount=0.3)
    b = base.displace(field, amount=0.3)
    assert a.positions.shape == base.positions.shape
    assert not np.allclose(a.positions, base.positions)
    assert np.array_equal(a.positions, b.positions)         # deterministic
    assert a.normals is not None                             # normals recomputed


def test_displace_along_fixed_vector():
    base = Mesh.plane(width=2, depth=2, segments=4)
    const = lambda pts: np.ones(len(pts), dtype=np.float32)  # noqa: E731
    out = base.displace(const, amount=0.5, along=(0, 1, 0))
    assert np.allclose(out.positions[:, 1], base.positions[:, 1] + 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_deform.py -v`
Expected: FAIL — `AttributeError: 'Mesh' object has no attribute 'displace'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/manifoldx/modeling/deform.py
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
```

Add to `Mesh`:

```python
    def displace(self, field, amount: float = 1.0, along="normal") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.displace(self, field, amount, along)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_deform.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/deform.py src/manifoldx/modeling/mesh.py tests/modeling/test_deform.py
git commit -m "feat(modeling): displace deformer (field along normal or vector)"
```

---

### Task 7: `twist(angle, axis)` deformer

**Files:**
- Modify: `src/manifoldx/modeling/deform.py`
- Modify: `src/manifoldx/modeling/mesh.py` (add `Mesh.twist`)
- Test: `tests/modeling/test_deform.py` (append)

**Interfaces:**
- Produces:
  - `deform.twist(mesh, angle, axis="y") -> Mesh` — rotates each vertex about `axis` by `angle * coord_along_axis` (radians per unit length). Recomputes normals.
  - `Mesh.twist(angle, axis="y") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_deform.py
def test_twist_preserves_axis_coordinate_and_count():
    base = Mesh.cylinder(radius=1.0, height=4.0, segments=24)
    out = base.twist(angle=1.0, axis="y")
    assert out.positions.shape == base.positions.shape
    # Twisting about y leaves the y coordinate unchanged.
    assert np.allclose(out.positions[:, 1], base.positions[:, 1])
    # Points off the axis actually move.
    assert not np.allclose(out.positions[:, [0, 2]], base.positions[:, [0, 2]])


def test_twist_zero_angle_is_identity():
    base = Mesh.cylinder(radius=1.0, height=2.0, segments=12)
    out = base.twist(angle=0.0, axis="y")
    assert np.allclose(out.positions, base.positions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_deform.py -k twist -v`
Expected: FAIL — `AttributeError: 'Mesh' object has no attribute 'twist'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/manifoldx/modeling/deform.py

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
```

Add to `Mesh`:

```python
    def twist(self, angle: float, axis: str = "y") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.twist(self, angle, axis)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_deform.py -k twist -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/deform.py src/manifoldx/modeling/mesh.py tests/modeling/test_deform.py
git commit -m "feat(modeling): twist deformer"
```

---

### Task 8: `bend(angle, axis, along)` deformer

**Files:**
- Modify: `src/manifoldx/modeling/deform.py`
- Modify: `src/manifoldx/modeling/mesh.py` (add `Mesh.bend`)
- Test: `tests/modeling/test_deform.py` (append)

**Interfaces:**
- Produces:
  - `deform.bend(mesh, angle, axis="z", along="x") -> Mesh` — bends the mesh in the plane spanned by `along` and the third axis, by `angle` radians across the `along` extent (rotation angle proportional to the `along` coordinate). `axis` is the rotation axis; `along` is the coordinate that drives the bend. Recomputes normals.
  - `Mesh.bend(angle, axis="z", along="x") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_deform.py
def test_bend_zero_angle_identity_and_count():
    base = Mesh.plane(width=4, depth=1, segments=8)
    out = base.bend(angle=0.0, axis="z", along="x")
    assert out.positions.shape == base.positions.shape
    assert np.allclose(out.positions, base.positions)


def test_bend_curves_the_strip():
    base = Mesh.plane(width=4, depth=1, segments=16)  # flat in XZ, y == 0
    out = base.bend(angle=1.5, axis="z", along="x")
    # Bending about z, driven by x, must lift vertices out of the y == 0 plane.
    assert np.abs(out.positions[:, 1]).max() > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_deform.py -k bend -v`
Expected: FAIL — `AttributeError: 'Mesh' object has no attribute 'bend'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/manifoldx/modeling/deform.py

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
```

Add to `Mesh`:

```python
    def bend(self, angle: float, axis: str = "z", along: str = "x") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.bend(self, angle, axis, along)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_deform.py -k bend -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/deform.py src/manifoldx/modeling/mesh.py tests/modeling/test_deform.py
git commit -m "feat(modeling): bend deformer"
```

---

### Task 9: `taper(factor, axis)` deformer

**Files:**
- Modify: `src/manifoldx/modeling/deform.py`
- Modify: `src/manifoldx/modeling/mesh.py` (add `Mesh.taper`)
- Test: `tests/modeling/test_deform.py` (append)

**Interfaces:**
- Produces:
  - `deform.taper(mesh, factor, axis="y") -> Mesh` — scales each vertex's cross-section (the two axes != `axis`) by `1 + factor * coord_along_axis`, normalized so the axis extent maps `factor` across `[-1, 1]` of the normalized coordinate. Recomputes normals.
  - `Mesh.taper(factor, axis="y") -> Mesh`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_deform.py
def test_taper_narrows_one_end():
    base = Mesh.cylinder(radius=1.0, height=2.0, segments=24)
    out = base.taper(factor=0.8, axis="y")
    top = base.positions[:, 1] > 0.9
    bot = base.positions[:, 1] < -0.9
    r_top = np.linalg.norm(out.positions[top][:, [0, 2]], axis=1).mean()
    r_bot = np.linalg.norm(out.positions[bot][:, [0, 2]], axis=1).mean()
    assert r_top > r_bot        # widened at +y, narrowed at -y


def test_taper_zero_is_identity():
    base = Mesh.cylinder(radius=1.0, height=2.0, segments=12)
    out = base.taper(factor=0.0, axis="y")
    assert np.allclose(out.positions, base.positions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_deform.py -k taper -v`
Expected: FAIL — `AttributeError: 'Mesh' object has no attribute 'taper'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/manifoldx/modeling/deform.py

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
```

Add to `Mesh`:

```python
    def taper(self, factor: float, axis: str = "y") -> "Mesh":
        from manifoldx.modeling import deform
        return deform.taper(self, factor, axis)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modeling/test_deform.py -k taper -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/deform.py src/manifoldx/modeling/mesh.py tests/modeling/test_deform.py
git commit -m "feat(modeling): taper deformer"
```

---

### Task 10: Upgrade the asteroid demo + CHANGELOG

**Files:**
- Modify: `examples/modeling_asteroid.py`
- Modify: `tests/modeling/test_asteroid_demo.py` (append the pipeline contract test)
- Modify: `CHANGELOG.md` (`[Unreleased]` entry)

**Interfaces:**
- Consumes: `Mesh.icosphere`, `Mesh.displace`, `Mesh.twist`, `Mesh.taper`, `noise.fbm`.

Grow the seam demo into the actual procedural asteroid by composing the Batch 1 deformers. (Smoothing arrives with Batch 2's `smooth`; not used here.)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_asteroid_demo.py
from manifoldx.modeling import noise


def test_asteroid_pipeline_produces_valid_geometry():
    rock = (
        Mesh.icosphere(subdivisions=4)
        .displace(noise.fbm(seed=7, octaves=5), amount=0.35)
        .twist(angle=0.4, axis="y")
        .taper(factor=0.2, axis="y")
    )
    geo = rock.to_geometry()
    assert geo["indices"].max() < geo["positions"].shape[0]
    assert np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modeling/test_asteroid_demo.py -k pipeline -v`
Expected: PASS if Tasks 5–9 are complete (this guards composition). If any deformer regressed, it fails there.

- [ ] **Step 3: Update the demo pipeline**

Replace the `rock = GeoMesh.icosphere(subdivisions=4)` line in `examples/modeling_asteroid.py` with:

```python
from manifoldx.modeling import noise

rock = (
    GeoMesh.icosphere(subdivisions=4)
    .displace(noise.fbm(seed=7, octaves=5), amount=0.35)
    .twist(angle=0.4, axis="y")
    .taper(factor=0.2, axis="y")
)
```

- [ ] **Step 4: Render the smoke clip + run tests**

Run: `uv run python examples/modeling_asteroid.py --render --duration 3 --fps 30 --output /tmp/asteroid.mp4`
Expected: exits 0, writes a rotating lumpy asteroid (skip + record if no GPU backend).
Run: `uv run pytest tests/modeling/ -v` → all PASS.
Run: `make lint` → clean.

- [ ] **Step 5: Commit + CHANGELOG**

Add under `## [Unreleased]` in `CHANGELOG.md`:

```markdown
### Added
- **Geometric modeling v1 (foundation + deformers).** New `manifoldx.modeling`
  subpackage: immutable numpy `Mesh` value type, `icosphere`/`box`/`plane`/
  `cylinder`/`torus` primitives, seeded `perlin`/`fbm` noise fields, and the
  `displace`/`twist`/`bend`/`taper` deformers, composed fluently and baked to
  the existing geometry dict via `Mesh.to_geometry()`. Demo:
  `examples/modeling_asteroid.py` (procedural asteroid). Design:
  `.knowledge/analysis/2026-07-28-geometric-modeling-v1-design.md`.
```

```bash
git add examples/modeling_asteroid.py tests/modeling/test_asteroid_demo.py CHANGELOG.md
git commit -m "feat(modeling): grow asteroid demo into full deformer pipeline + changelog"
```

---

## Self-Review

**Spec coverage (Batches 0–1 of the design):**
- `Mesh` value type + `with_positions`/`recompute_normals`/`to_geometry`/`from_geometry` → Task 1. ✓
- Primitives (box/plane/icosphere/cylinder/torus) → Tasks 2, 4. ✓
- Determinism via seeded noise → Task 5. ✓
- Deformers twist/bend/taper/displace → Tasks 6–9. ✓
- Naming: `modeling.Mesh` never top-level exported → enforced by `__init__.py` (Tasks 1, 5). ✓
- Integration seam `to_geometry()` only; no renderer/ECS edits → held across all tasks. ✓
- Asteroid demo as end-to-end deliverable → Tasks 3, 10. ✓
- Testing strategy (invariants, determinism, one render smoke) → per-task tests + render step. ✓
- **Intentionally deferred (design's own batch split):** `adjacency`+`smooth`, sculpt (Batch 2); topology (Batch 3); booleans (Batch 4); `ffd`/`bend_along_curve` (Batch 1 tail). Called out in Global Constraints. ✓

**Placeholder scan:** No TBD/TODO; every code + test step carries real content. ✓

**Type consistency:** `Mesh(positions, faces, normals, uvs)` used identically everywhere; deformers are `(mesh, ...) -> Mesh` with matching `Mesh` method wrappers; field callables are `(points (K,3)) -> (K,)` in both `noise.py` and `displace`. `_AXIS` map shared across twist/bend/taper. ✓
