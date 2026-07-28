# Modeling Fields v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** A composable scalar-field algebra (`Field` type + sources + combinators) so developers build their own terrain/deformation fields from reusable primitives.

**Architecture:** `Field` wraps a callable `(points (K,3)) -> (K,) float32` and *is* callable, so `displace(field)` consumes it unchanged. Operators/methods return new `Field`s. Sources are module functions returning `Field`s. `perlin`/`fbm` move from `noise.py` into `fields.py`; `noise.py` becomes a back-compat shim.

**Tech Stack:** Python 3.13+, numpy, `uv`. Design: `.knowledge/analysis/2026-07-28-modeling-fields-v1-design.md`.

## Global Constraints

- Pure numpy, dependency-free. Run tests with `.venv/bin/python -m pytest` (uv run has been resolving inconsistently this session).
- `Field.__call__` returns `(K,) float32`; coerce incoming points to float64 for sampling math.
- Scalars coerce to constant fields inside operators/combinators (`field + 0.5`, `2.0 * field`).
- Back-compat: `from manifoldx.modeling import noise; noise.perlin/fbm(...)` must keep working (now returning `Field`, still callable). Existing modeling tests/examples must pass unchanged.
- Conventional commits `feat(modeling):`. No renderer/ECS changes (coloring is a later sub-project).

## Consumed from current `main`

- `manifoldx.modeling.noise.perlin(seed, freq)` / `fbm(seed, freq, octaves, lacunarity, gain)` — existing gradient/fbm noise (to be moved).
- `manifoldx.modeling.noise._resolve_rng(seed)` — None/int/Generator → `np.random.Generator`.
- `Mesh.plane(width, depth, segments)`, `Mesh.displace(field, amount, along)`.

## File Structure

- Create `src/manifoldx/modeling/fields.py` — `Field` + `_as_field` + `_resolve_rng` + sources.
- Rewrite `src/manifoldx/modeling/noise.py` — thin shim re-exporting `perlin`, `fbm`.
- Modify `src/manifoldx/modeling/__init__.py` — export `fields`, `Field`.
- Create `tests/modeling/test_fields.py`.
- Create `examples/modeling_fields.py`; modify `CHANGELOG.md`.

---

### Task 1: `Field` type — operators + combinators + warp

**Files:** Create `src/manifoldx/modeling/fields.py`; test `tests/modeling/test_fields.py`.

**Interfaces — Produces:**
- `Field(fn)`; `Field.__call__(points) -> (K,) float32`.
- Operators: `+ - * /` (Field⊕Field or Field⊕scalar, incl. reflected), unary `-`.
- Methods: `mix(other, t)`, `minimum(other)`, `maximum(other)`, `clamp(lo, hi)`, `remap(a, b, c, d)`, `abs()`, `power(n)`, `scale(s)`, `bias(b)`, `warp(amount, fx=None, fy=None, fz=None)`.
- `_as_field(x)` — wrap scalars as constant fields (internal).

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_fields.py
import numpy as np
from manifoldx.modeling import fields
from manifoldx.modeling.fields import Field


def _pts():
    return np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)


X = Field(lambda p: p[:, 0])          # the x-coordinate as a field


def test_call_returns_float32():
    out = X(_pts())
    assert out.dtype == np.float32 and out.shape == (3,)
    assert np.allclose(out, [0, 1, 2])


def test_arithmetic_operators():
    assert np.allclose((X + 1.0)(_pts()), [1, 2, 3])
    assert np.allclose((1.0 + X)(_pts()), [1, 2, 3])
    assert np.allclose((2.0 * X)(_pts()), [0, 2, 4])
    assert np.allclose((X - X)(_pts()), [0, 0, 0])
    assert np.allclose((10.0 - X)(_pts()), [10, 9, 8])
    assert np.allclose((X / 2.0)(_pts()), [0, 0.5, 1.0])
    assert np.allclose((-X)(_pts()), [0, -1, -2])


def test_combinators():
    assert np.allclose(X.clamp(0, 1)(_pts()), [0, 1, 1])
    assert np.allclose(X.remap(0, 2, 0, 10)(_pts()), [0, 5, 10])
    assert np.allclose(X.minimum(1.0)(_pts()), [0, 1, 1])
    assert np.allclose(X.maximum(1.0)(_pts()), [1, 1, 2])
    assert np.allclose(X.power(2)(_pts()), [0, 1, 4])
    assert np.allclose(X.scale(3).bias(1)(_pts()), [1, 4, 7])
    ten = Field(lambda p: np.full(len(p), 10.0))
    assert np.allclose(X.mix(ten, 0.5)(_pts()), [5.0, 5.5, 6.0])


def test_warp_shifts_sampling():
    ones = Field(lambda p: np.ones(len(p)))
    warped = X.warp(1.0, fx=ones)      # sample X at x + 1 → values shift up by 1
    assert np.allclose(warped(_pts()), [1, 2, 3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'manifoldx.modeling.fields'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/manifoldx/modeling/fields.py
"""Composable scalar-field algebra: a fluent Field type + noise/pattern sources.

A Field wraps (points (K,3)) -> (K,) float32 and is itself callable, so any
consumer that samples a field (e.g. Mesh.displace) accepts a Field unchanged.
"""

from __future__ import annotations

import numpy as np


def _resolve_rng(seed) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class Field:
    """A composable scalar field over 3-D space."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return np.asarray(self._fn(p), dtype=np.float32)

    # --- arithmetic ---
    def __add__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) + o(p))

    __radd__ = __add__

    def __sub__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) - o(p))

    def __rsub__(self, o):
        o = _as_field(o)
        return Field(lambda p: o(p) - self(p))

    def __mul__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) * o(p))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) / o(p))

    def __rtruediv__(self, o):
        o = _as_field(o)
        return Field(lambda p: o(p) / self(p))

    def __neg__(self):
        return Field(lambda p: -self(p))

    # --- combinators ---
    def mix(self, other, t):
        other, tf = _as_field(other), _as_field(t)
        return Field(lambda p: self(p) * (1.0 - tf(p)) + other(p) * tf(p))

    def minimum(self, other):
        other = _as_field(other)
        return Field(lambda p: np.minimum(self(p), other(p)))

    def maximum(self, other):
        other = _as_field(other)
        return Field(lambda p: np.maximum(self(p), other(p)))

    def clamp(self, lo, hi):
        return Field(lambda p: np.clip(self(p), lo, hi))

    def remap(self, a, b, c, d):
        return Field(lambda p: c + (self(p) - a) * (d - c) / (b - a))

    def abs(self):
        return Field(lambda p: np.abs(self(p)))

    def power(self, n):
        return Field(lambda p: np.power(self(p), n))

    def scale(self, s):
        return self * s

    def bias(self, b):
        return self + b

    def warp(self, amount, fx=None, fy=None, fz=None):
        fx = _as_field(0.0 if fx is None else fx)
        fy = _as_field(0.0 if fy is None else fy)
        fz = _as_field(0.0 if fz is None else fz)

        def fn(p):
            offset = np.stack([fx(p), fy(p), fz(p)], axis=1) * amount
            return self(p + offset)

        return Field(fn)


def _as_field(x) -> Field:
    if isinstance(x, Field):
        return x
    v = float(x)
    return Field(lambda p, v=v: np.full(len(p), v, dtype=np.float32))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/fields.py tests/modeling/test_fields.py
git commit -m "feat(modeling): composable Field type (operators, combinators, warp)"
```

---

### Task 2: Migrate `perlin`/`fbm` into `fields.py`; `noise.py` shim

**Files:** Modify `src/manifoldx/modeling/fields.py`, rewrite `src/manifoldx/modeling/noise.py`; test `tests/modeling/test_fields.py` (append).

**Interfaces — Produces:** `fields.perlin(seed=None, freq=1.0) -> Field`, `fields.fbm(seed=None, freq=1.0, octaves=4, lacunarity=2.0, gain=0.5) -> Field`. `noise.perlin`/`noise.fbm` re-export them.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_fields.py
from manifoldx.modeling import noise


def _grid(k=200):
    return np.random.default_rng(0).uniform(-3, 3, size=(k, 3)).astype(np.float32)


def test_perlin_fbm_are_fields():
    assert isinstance(fields.perlin(seed=1), Field)
    assert isinstance(fields.fbm(seed=1), Field)


def test_perlin_deterministic_and_bounded():
    pts = _grid(500)
    a, b = fields.perlin(seed=7)(pts), fields.perlin(seed=7)(pts)
    assert np.array_equal(a, b)
    assert a.shape == (500,) and a.min() >= -1.5 and a.max() <= 1.5


def test_noise_shim_matches_fields():
    pts = _grid()
    assert np.array_equal(noise.fbm(seed=3, octaves=4)(pts), fields.fbm(seed=3, octaves=4)(pts))
    assert callable(noise.perlin(seed=2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -k "are_fields or shim" -q`
Expected: FAIL — `AttributeError: module 'manifoldx.modeling.fields' has no attribute 'perlin'`.

- [ ] **Step 3: Write minimal implementation**

Move the `perlin` and `fbm` bodies verbatim from the current `noise.py` into `fields.py`, wrapping each returned closure in `Field(...)`. The `perlin` closure `field` and `fbm` closure `field` are unchanged except the outer `return Field(field)`; `fbm`'s internal `perlin(seed=rng, ...)` now yields `Field`s whose `f(points)` call still returns arrays, so the octave loop is unchanged. Both live below `_as_field` in `fields.py`.

Then rewrite `noise.py` entirely:

```python
# src/manifoldx/modeling/noise.py
"""Back-compat shim. Noise sources now live in manifoldx.modeling.fields."""

from manifoldx.modeling.fields import perlin, fbm, _resolve_rng  # noqa: F401

__all__ = ["perlin", "fbm"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py tests/modeling/test_noise.py -q`
Expected: PASS — new tests plus the existing `test_noise.py` (now hitting the shim).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/fields.py src/manifoldx/modeling/noise.py tests/modeling/test_fields.py
git commit -m "feat(modeling): move perlin/fbm into fields (return Field); noise is a shim"
```

---

### Task 3: `ridged` + `billow` sources

**Files:** Modify `fields.py`; test `tests/modeling/test_fields.py` (append).

**Interfaces — Produces:** `fields.ridged(seed=None, freq=1.0, octaves=4, lacunarity=2.0, gain=0.5) -> Field` (range ~[0,1]); `fields.billow(...)` same signature (range ~[0,1]).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_fields.py
def test_ridged_and_billow_range_and_determinism():
    pts = _grid(500)
    for src in (fields.ridged, fields.billow):
        a, b = src(seed=4)(pts), src(seed=4)(pts)
        assert np.array_equal(a, b)
        assert a.shape == (500,)
        assert a.min() >= -1e-4 and a.max() <= 1.0 + 1e-4      # non-negative, bounded
    assert not np.allclose(fields.ridged(seed=1)(pts), fields.ridged(seed=2)(pts))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -k "ridged" -q`
Expected: FAIL — `AttributeError: ... 'ridged'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to fields.py
def ridged(seed=None, freq: float = 1.0, octaves: int = 4,
           lacunarity: float = 2.0, gain: float = 0.5) -> Field:
    rng = _resolve_rng(seed)
    layers = [(perlin(seed=rng, freq=freq * lacunarity**i), gain**i) for i in range(octaves)]
    norm = sum(a for _, a in layers)

    def fn(p):
        total = np.zeros(len(p), dtype=np.float32)
        for f, amp in layers:
            total += amp * (1.0 - np.abs(f(p))) ** 2
        return (total / norm).astype(np.float32)

    return Field(fn)


def billow(seed=None, freq: float = 1.0, octaves: int = 4,
           lacunarity: float = 2.0, gain: float = 0.5) -> Field:
    rng = _resolve_rng(seed)
    layers = [(perlin(seed=rng, freq=freq * lacunarity**i), gain**i) for i in range(octaves)]
    norm = sum(a for _, a in layers)

    def fn(p):
        total = np.zeros(len(p), dtype=np.float32)
        for f, amp in layers:
            total += amp * np.abs(f(p))
        return (total / norm).astype(np.float32)

    return Field(fn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -k "ridged" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/fields.py tests/modeling/test_fields.py
git commit -m "feat(modeling): ridged + billow field sources"
```

---

### Task 4: `worley` cellular source

**Files:** Modify `fields.py`; test `tests/modeling/test_fields.py` (append).

**Interfaces — Produces:** `fields.worley(seed=None, freq=1.0, feature="f1"|"f2f1", metric="euclidean"|"manhattan") -> Field` (range ≥ 0). Feature points: one per integer lattice cell at `cell + hash(cell, seed)`; distance to nearest (`f1`) or `F2 - F1` (`f2f1`). Searches the 3×3×3 neighbour cells.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_fields.py
def test_worley_deterministic_nonnegative():
    pts = _grid(400)
    a, b = fields.worley(seed=5)(pts), fields.worley(seed=5)(pts)
    assert np.array_equal(a, b)
    assert a.shape == (400,) and a.min() >= 0.0
    assert not np.allclose(fields.worley(seed=1)(pts), fields.worley(seed=2)(pts))


def test_worley_f2f1_is_ridge_like():
    # F2 - F1 is ~0 exactly on a cell border and grows toward cell centres.
    w = fields.worley(seed=3, feature="f2f1")
    pts = _grid(400)
    v = w(pts)
    assert v.min() >= 0.0
    assert v.max() > 0.1          # some cellular structure exists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -k worley -q`
Expected: FAIL — `AttributeError: ... 'worley'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to fields.py
def _hash3(cell: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-cell offset in [0, 1)^3 from integer cell + seed."""
    x = cell[:, 0].astype(np.int64)
    y = cell[:, 1].astype(np.int64)
    z = cell[:, 2].astype(np.int64)
    base = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791) ^ (seed * 2654435761)

    def rnd(salt):
        h = (base ^ (salt * 40503)).astype(np.uint64)
        h ^= h >> np.uint64(13)
        h *= np.uint64(0x9E3779B1)
        h ^= h >> np.uint64(15)
        return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)

    return np.stack([rnd(1), rnd(2), rnd(3)], axis=1)


def worley(seed=None, freq: float = 1.0, feature: str = "f1", metric: str = "euclidean") -> Field:
    seed_int = int(_resolve_rng(seed).integers(0, 2**31 - 1))

    def fn(points):
        p = points * freq
        base = np.floor(p).astype(np.int64)
        best1 = np.full(len(p), np.inf)
        best2 = np.full(len(p), np.inf)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = base + np.array([dx, dy, dz], dtype=np.int64)
                    fp = cell.astype(np.float64) + _hash3(cell, seed_int)
                    diff = p - fp
                    if metric == "manhattan":
                        d = np.abs(diff).sum(axis=1)
                    else:
                        d = np.sqrt((diff * diff).sum(axis=1))
                    closer = d < best1
                    best2 = np.where(closer, best1, np.minimum(best2, d))
                    best1 = np.where(closer, d, best1)
        out = (best2 - best1) if feature == "f2f1" else best1
        return out.astype(np.float32)

    return Field(fn)
```

Note: numpy may warn on `uint64 >> python-int`; the `np.uint64(...)` casts on the shift amounts and multipliers avoid it. If a `RuntimeWarning`/overflow surfaces in Step 4, keep all shift/multiply constants wrapped in `np.uint64(...)` (they already are) — overflow *wrapping* on `uint64` is the intended hash behaviour, not a bug.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -k worley -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/fields.py tests/modeling/test_fields.py
git commit -m "feat(modeling): worley cellular field source (f1 / f2f1)"
```

---

### Task 5: `constant` / `coord` / `distance` sources

**Files:** Modify `fields.py`; test `tests/modeling/test_fields.py` (append).

**Interfaces — Produces:** `fields.constant(value) -> Field`; `fields.coord(axis: "x"|"y"|"z") -> Field`; `fields.distance(center=(0,0,0)) -> Field`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/modeling/test_fields.py
def test_constant_coord_distance():
    pts = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 2]], dtype=np.float64)
    assert np.allclose(fields.constant(2.5)(pts), [2.5, 2.5, 2.5])
    assert np.allclose(fields.coord("x")(pts), [0, 3, 0])
    assert np.allclose(fields.coord("z")(pts), [0, 0, 2])
    assert np.allclose(fields.distance()(pts), [0, 5, 2])
    assert np.allclose(fields.distance(center=(3, 4, 0))(pts), [5, 0, np.sqrt(3**2 + 4**2 + 2**2)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -k "constant_coord" -q`
Expected: FAIL — `AttributeError: ... 'constant'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to fields.py
_AXIS = {"x": 0, "y": 1, "z": 2}


def constant(value) -> Field:
    v = float(value)
    return Field(lambda p: np.full(len(p), v, dtype=np.float32))


def coord(axis: str) -> Field:
    i = _AXIS[axis]
    return Field(lambda p: p[:, i].astype(np.float32))


def distance(center=(0, 0, 0)) -> Field:
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    return Field(lambda p: np.linalg.norm(p - c, axis=1).astype(np.float32))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields.py -k "constant_coord" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/modeling/fields.py tests/modeling/test_fields.py
git commit -m "feat(modeling): constant/coord/distance field sources"
```

---

### Task 6: Exports + fields demo + CHANGELOG

**Files:** Modify `src/manifoldx/modeling/__init__.py`, create `examples/modeling_fields.py`, modify `CHANGELOG.md`; test `tests/modeling/test_fields_demo.py`.

**Interfaces — Produces:** `from manifoldx.modeling import fields, Field`. A developer-composed terrain field displacing a plane.

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/test_fields_demo.py
import numpy as np
from manifoldx.modeling import Mesh, fields


def test_composed_terrain_field_displaces_plane():
    terrain = (
        fields.ridged(seed=3, freq=0.4) * 0.7
        + fields.fbm(seed=7, freq=1.2) * 0.15
    ).warp(0.3, fx=fields.fbm(seed=2), fz=fields.fbm(seed=9)).remap(-1, 1, 0, 1)

    base = Mesh.plane(width=10, depth=10, segments=40)
    out = base.displace(terrain, amount=2.0)
    geo = out.to_geometry()
    assert out.positions.shape == base.positions.shape
    assert np.all(np.isfinite(geo["positions"]))
    assert out.positions[:, 1].ptp() > 0.1          # terrain has relief
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields_demo.py -q`
Expected: FAIL — `ImportError: cannot import name 'fields'` (until `__init__` exports it).

- [ ] **Step 3: Wire exports + write the demo**

Update `__init__.py`:

```python
from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.sculpt import Falloff
from manifoldx.modeling.ffd import FFD
from manifoldx.modeling.fields import Field
from manifoldx.modeling import fields, noise

__all__ = ["Mesh", "Falloff", "FFD", "Field", "fields", "noise"]
```

Create `examples/modeling_fields.py`: a developer-composed terrain field (ridged mountains + fbm roughness, domain-warped, remapped, with a `distance`-based island falloff) displacing `Mesh.plane(width, depth, segments=140)`; laid flat (rotate the +Z plane to normal +Y or displace along +Y — the plane's normal is +Y already, so `displace(field, amount, along="normal")` lifts +Y), a raking `DirectionalLight` + `enable_shadows`, a slow `camera.orbit`. Monochrome clay `StandardMaterial` (color arrives in the next sub-project). Follow the lighting/animation pattern in `examples/modeling_topology.py`.

- [ ] **Step 4: Run test + render**

Run: `.venv/bin/python -m pytest tests/modeling/test_fields_demo.py -q` → PASS.
Run: `.venv/bin/python examples/modeling_fields.py --render --duration 4 --fps 24 --output /tmp/fields.mp4` → exits 0 (skip + record if no GPU backend).
Run: `uv run ruff check src/manifoldx/modeling/ examples/modeling_fields.py tests/modeling/` → clean.

- [ ] **Step 5: Commit + CHANGELOG**

Add under `## [Unreleased]` → `### Features` a bullet describing the composable field algebra (`Field` + operators + `mix`/`clamp`/`remap`/`warp` + sources `perlin`/`fbm`/`ridged`/`billow`/`worley`/`constant`/`coord`/`distance`; `noise` now a shim; demo `examples/modeling_fields.py`).

```bash
git add src/manifoldx/modeling/__init__.py examples/modeling_fields.py tests/modeling/test_fields_demo.py CHANGELOG.md
git commit -m "feat(modeling): fields demo (composed terrain) + exports + changelog"
```

---

## Self-Review

**Spec coverage:** `Field` type + operators + combinators + warp (T1); `perlin`/`fbm` → `Field`, `noise` shim (T2); `ridged`/`billow` (T3); `worley` f1/f2f1 (T4); `constant`/`coord`/`distance` (T5); exports + demo (T6). Back-compat verified in T2 + by existing `test_noise.py`. ✓

**Placeholder scan:** every code + test step has real content; the demo body references the concrete pattern in `modeling_topology.py` rather than pseudo-code. ✓

**Type consistency:** every source returns `Field`; `Field.__call__ -> (K,) float32`; `_as_field` coerces scalars everywhere; `_resolve_rng` shared by all seeded sources; `warp(amount, fx, fy, fz)` signature identical in code + test. ✓
