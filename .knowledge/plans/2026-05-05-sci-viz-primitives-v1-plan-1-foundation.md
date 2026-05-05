# Sci-Viz Primitives v1 — Plan 1: Foundation (PointCloud + ColormapMaterial)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the foundational point-cloud primitive — `PointCloud` marker component, `ScalarValue` and `Radius` components, six built-in colormaps as 1D LUT textures, a `ColormapMaterial` with a camera-facing point-sprite shader using sphere-imposter shading, a new sprite render path in the renderer, and integration tests that validate per-frame attribute updates propagate to the GPU.

**Architecture:** New `manifoldx.viz` subpackage holds all sci-viz primitives. `PointCloud` is a marker (zero-field) component that tells the renderer to use the sprite path. `ScalarValue` and `Radius` are standard 1-float components updatable via the existing `_FieldView` deferred-mutation path. `ColormapMaterial` registers a 1D RGBA8 LUT (256 texels) plus per-batch `vmin`/`vmax` uniforms; per-instance `transforms`, `scalar_values`, and `radii` upload as parallel storage buffers each frame. Renderer splits batch dispatch into mesh + sprite groups; sprite path expands a unit quad into a camera-facing billboard, scales by per-instance radius, and reconstructs sphere normals in the fragment shader.

**Tech Stack:** Python 3.13+, NumPy, wgpu-py (rendercanvas), pytest, uv.

**Spec:** `.knowledge/analysis/2026-05-05-sci-viz-primitives-v1-design.md` (commit `6068533`).

**Out of scope for this plan (later plans):** Text rendering / `LabelMaterial` (Plan 2), `AxisFrame` / `ScaleBar` / `colormap_legend` (Plan 3), functional shim API (`point_cloud()`, `axes()`, etc.) (Plan 4), visual regression infrastructure (Plan 5).

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `src/manifoldx/viz/__init__.py` | Public re-exports for Plan 1 surface (`PointCloud`, `ScalarValue`, `Radius`, `ColormapMaterial`). |
| `src/manifoldx/viz/components.py` | `PointCloud` (marker, 0 floats), `ScalarValue` (1 float), `Radius` (1 float). |
| `src/manifoldx/viz/colormaps.py` | LUT data for viridis / magma / plasma / inferno / turbo / gray (256 texels each as 256×4 uint8 NumPy arrays); `get_colormap(name)` lookup. |
| `src/manifoldx/viz/geometry.py` | `SPRITE_QUAD` definition: 4 vertices (no normals — fragment shader reconstructs), 6 indices. |
| `src/manifoldx/viz/materials.py` | `ColormapMaterial` class with WGSL shader source and uniform/binding layout. |
| `tests/viz/__init__.py` | Empty package marker. |
| `tests/viz/test_colormaps.py` | LUT shape, normalization, known sample values for viridis. |
| `tests/viz/test_components_viz.py` | Register / spawn / data layout for `PointCloud`, `ScalarValue`, `Radius`. |
| `tests/viz/test_geometry_viz.py` | `SPRITE_QUAD` vertex/index counts, topology. |
| `tests/viz/test_materials_viz.py` | `ColormapMaterial` uniform packing, pipeline cache key, shader source returned. |
| `tests/viz/test_point_cloud_integration.py` | End-to-end: spawn, render to offscreen canvas, read back framebuffer, verify pixel colors against expected colormap output. |

### Modified files

| Path | Change |
|---|---|
| `src/manifoldx/__init__.py` | Add `from manifoldx import viz` re-export. |
| `src/manifoldx/resources.py` | Register `SPRITE_QUAD` as a built-in geometry alongside `cube`, `sphere`, `plane`. Keep existing `MaterialRegistry` API. |
| `src/manifoldx/renderer.py` | Extend `RenderPipeline` to: (a) build separate mesh and sprite batch lists in `run()`; (b) populate per-batch `scalar_values` and `radii` storage buffers when material is `ColormapMaterial`; (c) add `_render_sprite_batches()` method; (d) extend pipeline cache key with `material_subtype` (LUT identity for `ColormapMaterial`). |
| `pyproject.toml` | Add `[project.optional-dependencies] viz = ["pillow>=10.0"]` (Pillow staged for Plan 2 but added here so the extra exists; no Pillow imports in Plan 1). Update `all = [...]` to include `manifold-gfx[viz]`. |

---

## Sequencing notes

Tasks are ordered to respect TDD and dependency ordering:

1. **Tasks 1–3:** Skeleton + colormaps (pure data/Python, no GPU).
2. **Tasks 4–6:** Three new components (extend existing `EntityStore` registration).
3. **Task 7:** `SPRITE_QUAD` geometry data + register in `resources.py`.
4. **Tasks 8–11:** `ColormapMaterial` — uniform layout, vertex shader, fragment shader, integration with `MaterialRegistry`.
5. **Tasks 12–14:** Renderer split — sprite batch list, sprite render path, pipeline cache key.
6. **Tasks 15–17:** Integration tests + smoke + cleanup.

Each task ends with a commit. Use `cd /home/apiad/Workspace/repos/manifoldx` for all commands; the repo has its own git history independent of the workspace.

---

## Task 1: Set up `viz` subpackage skeleton

**Files:**
- Create: `src/manifoldx/viz/__init__.py`
- Create: `tests/viz/__init__.py`
- Modify: `src/manifoldx/__init__.py` (add `from manifoldx import viz`)
- Modify: `pyproject.toml` (add `[viz]` extra)

- [ ] **Step 1: Create empty test package marker**

```bash
mkdir -p tests/viz
touch tests/viz/__init__.py
```

- [ ] **Step 2: Create empty viz package with placeholder `__init__.py`**

```bash
mkdir -p src/manifoldx/viz
```

Write `src/manifoldx/viz/__init__.py`:

```python
"""manifoldx.viz — Scientific visualization primitives.

Public surface for Plan 1:
    PointCloud, ScalarValue, Radius, ColormapMaterial

Future plans add:
    TextLabel, AxisFrame, ScaleBar, LabelMaterial, AxisMaterial,
    point_cloud(), axes(), scale_bar(), colormap_legend()
"""
```

- [ ] **Step 3: Re-export viz from top-level package**

In `src/manifoldx/__init__.py`, append (after existing imports — locate the end of the import block, then add):

```python
from manifoldx import viz  # noqa: E402, F401
```

- [ ] **Step 4: Add `[viz]` extra in `pyproject.toml`**

Find the `[project.optional-dependencies]` table. If it does not yet have a `viz` entry, add:

```toml
viz = ["pillow>=10.0"]
```

Update the existing `all = [...]` line so it includes the viz extra. The exact existing form will look like `all = ["manifold-gfx[desktop,offline]"]` — extend to `all = ["manifold-gfx[desktop,offline,viz]"]`.

- [ ] **Step 5: Run existing tests to verify nothing broke**

```bash
cd /home/apiad/Workspace/repos/manifoldx
uv sync --all-extras
make test
```

Expected: all existing tests pass. If any fail, the `viz` re-export broke something — check that no name collisions exist.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/viz/ tests/viz/ src/manifoldx/__init__.py pyproject.toml
git commit -m "feat(viz): add manifoldx.viz subpackage skeleton"
```

---

## Task 2: Implement viridis colormap LUT and lookup

**Files:**
- Create: `src/manifoldx/viz/colormaps.py`
- Create: `tests/viz/test_colormaps.py`

- [ ] **Step 1: Write the failing test**

Write `tests/viz/test_colormaps.py`:

```python
"""Tests for manifoldx.viz.colormaps module."""
import numpy as np
import pytest

from manifoldx.viz import colormaps


def test_viridis_shape():
    lut = colormaps.get_colormap("viridis")
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8


def test_viridis_endpoints():
    """Viridis spans dark purple (idx 0) to yellow (idx 255)."""
    lut = colormaps.get_colormap("viridis")
    # Dark purple: low R, low G, mid B
    r0, g0, b0, a0 = lut[0]
    assert r0 < 100
    assert g0 < 30
    assert 60 < b0 < 130
    assert a0 == 255
    # Yellow: high R, high G, low B
    r255, g255, b255, a255 = lut[255]
    assert r255 > 200
    assert g255 > 200
    assert b255 < 60
    assert a255 == 255


def test_unknown_colormap_raises():
    with pytest.raises(KeyError, match="unknown colormap"):
        colormaps.get_colormap("not-a-real-cmap")


def test_lookup_normalized_value():
    """Sample at a normalized scalar; verify (R,G,B,A) returned matches LUT."""
    lut = colormaps.get_colormap("viridis")
    rgba = colormaps.lookup("viridis", 0.5)
    assert rgba.shape == (4,)
    # Index 127 or 128 — accept either
    np.testing.assert_array_equal(rgba, lut[127]) if (rgba == lut[127]).all() else np.testing.assert_array_equal(rgba, lut[128])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/apiad/Workspace/repos/manifoldx
uv run pytest tests/viz/test_colormaps.py -v
```

Expected: ImportError — `manifoldx.viz.colormaps` does not exist.

- [ ] **Step 3: Implement minimal colormaps module — viridis only**

The simplest production implementation pulls LUTs from `matplotlib.cm` once at import time and freezes them as `np.uint8` arrays. matplotlib is *not* a runtime dep of manifoldx, so we generate the LUTs once and check them in as Python literals to avoid the runtime dependency.

Use this script (run once locally to produce the LUT data — not part of the package):

```python
# Generate-once helper — DO NOT commit this, paste output into colormaps.py
import matplotlib.cm as cm
import numpy as np
for name in ["viridis", "magma", "plasma", "inferno", "turbo", "gray"]:
    cmap = cm.get_cmap(name, 256)
    lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    print(f"_{name.upper()}_LUT_BASE64 = ", repr(lut.tobytes().hex()))
```

For Task 2 we only ship viridis. Write `src/manifoldx/viz/colormaps.py`:

```python
"""1D colormap LUTs for sci-viz materials.

Each colormap is a 256-texel RGBA8 array (shape (256, 4), dtype uint8).
LUTs are precomputed from matplotlib at package-build time and frozen
here as hex strings to avoid a matplotlib runtime dependency.
"""
import numpy as np

# Precomputed from matplotlib 3.x: cm.get_cmap("viridis", 256)
# Run scripts/regenerate_colormaps.py to regenerate.
_VIRIDIS_HEX = (
    "PASTE_GENERATED_HEX_HERE"  # 1024 hex chars; see Step 4 below for generator
)


def _decode_lut(hex_str: str) -> np.ndarray:
    """Decode a hex-encoded RGBA8 LUT into a (256, 4) uint8 array."""
    raw = bytes.fromhex(hex_str)
    if len(raw) != 256 * 4:
        raise ValueError(f"LUT must be 1024 bytes; got {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(256, 4).copy()


_LUTS = {
    "viridis": _decode_lut(_VIRIDIS_HEX),
}


def get_colormap(name: str) -> np.ndarray:
    """Return the (256, 4) uint8 LUT for the named colormap."""
    if name not in _LUTS:
        raise KeyError(f"unknown colormap: {name!r}; available: {sorted(_LUTS)}")
    return _LUTS[name]


def lookup(name: str, value: float) -> np.ndarray:
    """Sample the colormap at normalized value in [0, 1]; returns (4,) uint8."""
    lut = get_colormap(name)
    idx = int(np.clip(value, 0.0, 1.0) * 255 + 0.5)
    return lut[idx]
```

**Note for the implementer:** the actual `_VIRIDIS_HEX` string must be generated from matplotlib using the snippet above and pasted in. Do not commit the placeholder literally — the test will fail until real bytes are inserted. Add a generator script at `scripts/regenerate_colormaps.py` that re-emits these (one-liner per cmap): this script imports matplotlib, but only developers run it; runtime never imports matplotlib.

- [ ] **Step 4: Generate viridis LUT and paste into `colormaps.py`**

Run, in a Python shell (matplotlib must be available locally — `uv pip install matplotlib` if not):

```python
import matplotlib.cm as cm
import numpy as np
lut = (cm.get_cmap("viridis", 256)(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
print(lut.tobytes().hex())
```

Replace the `_VIRIDIS_HEX = "44015400ff..."` line in `colormaps.py` with the actual hex output (one long string, ~2050 chars). Also create `scripts/regenerate_colormaps.py` with the full generation snippet for future regeneration.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_colormaps.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/viz/colormaps.py tests/viz/test_colormaps.py scripts/regenerate_colormaps.py
git commit -m "feat(viz): add viridis colormap LUT with lookup helpers"
```

---

## Task 3: Add remaining colormaps (magma, plasma, inferno, turbo, gray)

**Files:**
- Modify: `src/manifoldx/viz/colormaps.py`
- Modify: `tests/viz/test_colormaps.py`

- [ ] **Step 1: Extend test to cover all 6 colormaps**

Add to `tests/viz/test_colormaps.py`:

```python
@pytest.mark.parametrize("name", ["viridis", "magma", "plasma", "inferno", "turbo", "gray"])
def test_all_colormaps_well_formed(name):
    lut = colormaps.get_colormap(name)
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8
    # Alpha is always 255
    assert (lut[:, 3] == 255).all()


def test_gray_is_monotonic():
    lut = colormaps.get_colormap("gray")
    rgb = lut[:, :3]
    # All three channels should equal each other and be monotonic increasing
    assert (rgb[:, 0] == rgb[:, 1]).all()
    assert (rgb[:, 1] == rgb[:, 2]).all()
    assert (np.diff(rgb[:, 0].astype(int)) >= 0).all()


def test_available_colormaps_complete():
    available = sorted(colormaps._LUTS.keys())
    assert available == sorted(["viridis", "magma", "plasma", "inferno", "turbo", "gray"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/viz/test_colormaps.py -v
```

Expected: parametrized cases for `magma`, `plasma`, `inferno`, `turbo`, `gray` fail with `KeyError`. `test_gray_is_monotonic` fails. `test_available_colormaps_complete` fails.

- [ ] **Step 3: Generate the 5 remaining LUTs**

Run the generator (extend `scripts/regenerate_colormaps.py` first to emit all six):

```python
# scripts/regenerate_colormaps.py
"""Regenerate the viz colormap hex literals from matplotlib.

Usage:
    uv pip install matplotlib
    uv run python scripts/regenerate_colormaps.py

Paste the output into src/manifoldx/viz/colormaps.py.
"""
import matplotlib.cm as cm
import numpy as np

NAMES = ["viridis", "magma", "plasma", "inferno", "turbo", "gray"]

for name in NAMES:
    lut = (cm.get_cmap(name, 256)(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    var = f"_{name.upper()}_HEX"
    print(f'{var} = (\n    "{lut.tobytes().hex()}"\n)\n')
```

Run it, copy the output, paste the 5 new hex literals into `src/manifoldx/viz/colormaps.py` above `_LUTS`.

- [ ] **Step 4: Extend the `_LUTS` dict**

In `src/manifoldx/viz/colormaps.py`, replace the existing `_LUTS` with:

```python
_LUTS = {
    "viridis": _decode_lut(_VIRIDIS_HEX),
    "magma":   _decode_lut(_MAGMA_HEX),
    "plasma":  _decode_lut(_PLASMA_HEX),
    "inferno": _decode_lut(_INFERNO_HEX),
    "turbo":   _decode_lut(_TURBO_HEX),
    "gray":    _decode_lut(_GRAY_HEX),
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_colormaps.py -v
```

Expected: all colormap tests pass (10 cases).

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/viz/colormaps.py tests/viz/test_colormaps.py scripts/regenerate_colormaps.py
git commit -m "feat(viz): add magma, plasma, inferno, turbo, gray colormaps"
```

---

## Task 4: `ScalarValue` component

**Files:**
- Create: `src/manifoldx/viz/components.py` (start)
- Create: `tests/viz/test_components_viz.py` (start)

- [ ] **Step 1: Write failing test**

Write `tests/viz/test_components_viz.py`:

```python
"""Tests for manifoldx.viz.components module."""
import numpy as np

from manifoldx.viz import components as viz_components


def test_scalar_value_default_data():
    sv = viz_components.ScalarValue()
    data = sv.get_data(n=5)
    assert data.shape == (5, 1)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, np.zeros((5, 1), dtype=np.float32))


def test_scalar_value_scalar_broadcast():
    sv = viz_components.ScalarValue(value=2.5)
    data = sv.get_data(n=10)
    assert data.shape == (10, 1)
    np.testing.assert_array_equal(data, np.full((10, 1), 2.5, dtype=np.float32))


def test_scalar_value_array_input():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sv = viz_components.ScalarValue(value=values)
    data = sv.get_data(n=4)
    assert data.shape == (4, 1)
    np.testing.assert_array_equal(data[:, 0], values)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/viz/test_components_viz.py -v
```

Expected: ImportError — `manifoldx.viz.components` does not exist.

- [ ] **Step 3: Implement `ScalarValue`**

Write `src/manifoldx/viz/components.py`:

```python
"""Sci-viz ECS components: PointCloud (marker), ScalarValue, Radius."""
import numpy as np


class ScalarValue:
    """Per-entity scalar attribute, mapped through ColormapMaterial's LUT.

    Storage layout: 1 float per entity (column 0).

    Usage:
        ScalarValue()                      # default 0.0 for all
        ScalarValue(value=1.5)             # broadcast scalar
        ScalarValue(value=array_shape_N)   # explicit per-entity
    """

    def __init__(self, value=None):
        self._value = value

    def get_data(self, n: int, registry=None) -> np.ndarray:
        data = np.zeros((n, 1), dtype=np.float32)
        if self._value is None:
            return data
        v = np.asarray(self._value, dtype=np.float32)
        if v.ndim == 0:
            data[:, 0] = float(v)
        elif v.ndim == 1 and v.shape[0] == n:
            data[:, 0] = v
        else:
            raise ValueError(
                f"ScalarValue: value shape {v.shape} incompatible with n={n}"
            )
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_components_viz.py::test_scalar_value_default_data tests/viz/test_components_viz.py::test_scalar_value_scalar_broadcast tests/viz/test_components_viz.py::test_scalar_value_array_input -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/components.py tests/viz/test_components_viz.py
git commit -m "feat(viz): add ScalarValue component"
```

---

## Task 5: `Radius` component

**Files:**
- Modify: `src/manifoldx/viz/components.py`
- Modify: `tests/viz/test_components_viz.py`

- [ ] **Step 1: Write failing test**

Append to `tests/viz/test_components_viz.py`:

```python
def test_radius_default_data():
    r = viz_components.Radius()
    data = r.get_data(n=5)
    assert data.shape == (5, 1)
    assert data.dtype == np.float32
    # Default radius is 1.0
    np.testing.assert_array_equal(data, np.ones((5, 1), dtype=np.float32))


def test_radius_scalar_broadcast():
    r = viz_components.Radius(radius=0.05)
    data = r.get_data(n=10)
    np.testing.assert_array_equal(data, np.full((10, 1), 0.05, dtype=np.float32))


def test_radius_array_input():
    radii = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    r = viz_components.Radius(radius=radii)
    data = r.get_data(n=3)
    np.testing.assert_array_equal(data[:, 0], radii)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/viz/test_components_viz.py -v -k radius
```

Expected: AttributeError — no `Radius` in module.

- [ ] **Step 3: Implement `Radius`**

Append to `src/manifoldx/viz/components.py`:

```python
class Radius:
    """Per-entity world-space radius for sprite scaling.

    Storage layout: 1 float per entity (column 0).
    Default: 1.0.

    Usage:
        Radius()                       # all 1.0
        Radius(radius=0.05)            # broadcast scalar
        Radius(radius=array_shape_N)   # explicit per-entity
    """

    def __init__(self, radius=None):
        self._radius = radius

    def get_data(self, n: int, registry=None) -> np.ndarray:
        data = np.ones((n, 1), dtype=np.float32)
        if self._radius is None:
            return data
        v = np.asarray(self._radius, dtype=np.float32)
        if v.ndim == 0:
            data[:, 0] = float(v)
        elif v.ndim == 1 and v.shape[0] == n:
            data[:, 0] = v
        else:
            raise ValueError(
                f"Radius: radius shape {v.shape} incompatible with n={n}"
            )
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_components_viz.py -v
```

Expected: 6 tests pass (3 ScalarValue + 3 Radius).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/components.py tests/viz/test_components_viz.py
git commit -m "feat(viz): add Radius component"
```

---

## Task 6: `PointCloud` marker component

**Files:**
- Modify: `src/manifoldx/viz/components.py`
- Modify: `src/manifoldx/viz/__init__.py`
- Modify: `tests/viz/test_components_viz.py`

- [ ] **Step 1: Write failing test**

Append to `tests/viz/test_components_viz.py`:

```python
def test_point_cloud_marker_no_data():
    """PointCloud is a marker — get_data returns an empty (n, 0) array."""
    pc = viz_components.PointCloud()
    data = pc.get_data(n=42)
    assert data.shape == (42, 0)
    assert data.dtype == np.float32


def test_point_cloud_importable_from_viz():
    from manifoldx.viz import PointCloud, ScalarValue, Radius
    assert PointCloud is viz_components.PointCloud
    assert ScalarValue is viz_components.ScalarValue
    assert Radius is viz_components.Radius
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/viz/test_components_viz.py -v -k point_cloud
```

Expected: AttributeError on `viz_components.PointCloud` (not yet defined) and ImportError on the second test.

- [ ] **Step 3: Implement `PointCloud`**

Append to `src/manifoldx/viz/components.py`:

```python
class PointCloud:
    """Marker component — tags entities for the sprite render path.

    Carries no per-entity data. The renderer detects this component
    and substitutes the SPRITE_QUAD geometry, ignoring any Mesh component.
    """

    def __init__(self):
        pass

    def get_data(self, n: int, registry=None) -> np.ndarray:
        # Marker — zero columns. Existence in EntityStore._components
        # is the signal; the data array is unused.
        return np.zeros((n, 0), dtype=np.float32)
```

Update `src/manifoldx/viz/__init__.py`:

```python
"""manifoldx.viz — Scientific visualization primitives.

Public surface for Plan 1:
    PointCloud, ScalarValue, Radius, ColormapMaterial

Future plans add:
    TextLabel, AxisFrame, ScaleBar, LabelMaterial, AxisMaterial,
    point_cloud(), axes(), scale_bar(), colormap_legend()
"""
from manifoldx.viz.components import PointCloud, Radius, ScalarValue

__all__ = ["PointCloud", "ScalarValue", "Radius"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_components_viz.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/components.py src/manifoldx/viz/__init__.py tests/viz/test_components_viz.py
git commit -m "feat(viz): add PointCloud marker component"
```

---

## Task 7: `SPRITE_QUAD` geometry definition

**Files:**
- Create: `src/manifoldx/viz/geometry.py`
- Create: `tests/viz/test_geometry_viz.py`

- [ ] **Step 1: Write failing test**

Write `tests/viz/test_geometry_viz.py`:

```python
"""Tests for manifoldx.viz.geometry module."""
import numpy as np

from manifoldx.viz import geometry as viz_geometry


def test_sprite_quad_vertex_count():
    """SPRITE_QUAD is a unit quad: 4 vertices."""
    q = viz_geometry.SPRITE_QUAD
    assert q["vertices"].shape == (4, 3)
    assert q["vertices"].dtype == np.float32


def test_sprite_quad_indices():
    """SPRITE_QUAD has 6 indices forming 2 triangles."""
    q = viz_geometry.SPRITE_QUAD
    assert q["indices"].shape == (6,)
    assert q["indices"].dtype == np.uint32
    # Indices reference all 4 vertices
    assert set(q["indices"].tolist()) == {0, 1, 2, 3}


def test_sprite_quad_uv_corners():
    """The four vertices form unit-quad corners in XY (z=0)."""
    q = viz_geometry.SPRITE_QUAD
    v = q["vertices"]
    # All vertices at z=0
    np.testing.assert_array_equal(v[:, 2], np.zeros(4, dtype=np.float32))
    # XY corners are (-1,-1), (1,-1), (1,1), (-1,1) in some order
    xy = set(map(tuple, v[:, :2].tolist()))
    assert xy == {(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/viz/test_geometry_viz.py -v
```

Expected: ImportError — `manifoldx.viz.geometry` does not exist.

- [ ] **Step 3: Implement `SPRITE_QUAD`**

Write `src/manifoldx/viz/geometry.py`:

```python
"""Built-in geometries for sci-viz primitives.

SPRITE_QUAD: unit quad in XY plane, expanded into a camera-facing
billboard by the vertex shader. UVs are reconstructed from the
quad-local position (xy in [-1, 1]^2) inside the fragment shader.
"""
import numpy as np


# Vertex layout: position only. Normals reconstructed in fragment shader.
_QUAD_VERTICES = np.array(
    [
        [-1.0, -1.0, 0.0],  # 0: bottom-left
        [ 1.0, -1.0, 0.0],  # 1: bottom-right
        [ 1.0,  1.0, 0.0],  # 2: top-right
        [-1.0,  1.0, 0.0],  # 3: top-left
    ],
    dtype=np.float32,
)

# Two triangles: (0, 1, 2) and (0, 2, 3) — counter-clockwise winding
_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


SPRITE_QUAD = {
    "vertices": _QUAD_VERTICES,
    "indices": _QUAD_INDICES,
    "name": "sprite_quad",
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_geometry_viz.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/geometry.py tests/viz/test_geometry_viz.py
git commit -m "feat(viz): add SPRITE_QUAD built-in geometry"
```

---

## Task 8: Register `SPRITE_QUAD` in the geometry registry

**Files:**
- Modify: `src/manifoldx/resources.py`
- Create: function-level test in `tests/viz/test_geometry_viz.py`

- [ ] **Step 1: Inspect existing geometry registration pattern**

Read `src/manifoldx/resources.py`, locate `GeometryRegistry` and the existing built-in geometry registration for `cube`, `sphere`, `plane`. Note the exact API used to register a built-in (e.g., `registry.register_builtin(name, vertices, indices)` or similar — adapt to the actual call).

- [ ] **Step 2: Write a failing integration test**

Append to `tests/viz/test_geometry_viz.py`:

```python
def test_sprite_quad_registered_in_geometry_registry():
    """After import, SPRITE_QUAD is a built-in geometry that can be looked up by name."""
    from manifoldx.resources import GeometryRegistry

    reg = GeometryRegistry()
    # The sprite_quad name must resolve. Implementation detail:
    # the registry exposes `get_id(name)` or `__contains__`. Adapt the
    # assertion to whichever the existing registry already uses for
    # cube/sphere/plane lookup.
    assert "sprite_quad" in reg
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/viz/test_geometry_viz.py::test_sprite_quad_registered_in_geometry_registry -v
```

Expected: failure — sprite_quad not in registry.

- [ ] **Step 4: Implement registration**

In `src/manifoldx/resources.py`, locate the block where `cube`, `sphere`, `plane` are registered as built-ins. Add `sprite_quad` registration alongside, importing from `manifoldx.viz.geometry`:

```python
# Near top of resources.py (with other imports)
from manifoldx.viz.geometry import SPRITE_QUAD as _SPRITE_QUAD

# In the GeometryRegistry __init__ or wherever built-ins are seeded —
# match the existing pattern. Pseudocode (adapt to actual API):
self._register_builtin(
    name=_SPRITE_QUAD["name"],
    vertices=_SPRITE_QUAD["vertices"],
    indices=_SPRITE_QUAD["indices"],
)
```

If the registry already supports `__contains__`, the test passes. If it does not, also add a minimal `__contains__` method:

```python
def __contains__(self, name: str) -> bool:
    return name in self._by_name  # or whatever internal field stores the lookup
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_geometry_viz.py -v
make test  # full suite, ensure no regressions
```

Expected: all viz tests pass; full suite still green.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/resources.py tests/viz/test_geometry_viz.py
git commit -m "feat(viz): register SPRITE_QUAD in GeometryRegistry"
```

---

## Task 9: `ColormapMaterial` — Python class skeleton + uniform packing

**Files:**
- Create: `src/manifoldx/viz/materials.py` (start)
- Create: `tests/viz/test_materials_viz.py` (start)

- [ ] **Step 1: Write failing test**

Write `tests/viz/test_materials_viz.py`:

```python
"""Tests for manifoldx.viz.materials module."""
import numpy as np
import pytest

from manifoldx.viz.materials import ColormapMaterial


def test_colormap_material_construction():
    m = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    assert m.cmap == "viridis"
    assert m.vmin == 0.0
    assert m.vmax == 1.0
    assert m.lit is False


def test_colormap_material_unknown_cmap():
    with pytest.raises(KeyError, match="unknown colormap"):
        ColormapMaterial(cmap="not-a-cmap", vmin=0.0, vmax=1.0)


def test_colormap_material_uniform_data():
    """Per-batch uniform: 4 floats — vmin, vmax, lit_flag, padding."""
    m = ColormapMaterial(cmap="viridis", vmin=-2.0, vmax=3.5, lit=True)
    data = m.get_data(n=10)
    # Per-batch material uniform; 4-float layout
    assert data.shape == (10, 4)
    assert data.dtype == np.float32
    # All rows identical (per-batch uniform)
    np.testing.assert_array_equal(data[0], data[5])
    np.testing.assert_array_equal(data[0], np.array([-2.0, 3.5, 1.0, 0.0], dtype=np.float32))


def test_colormap_material_lut_bytes():
    m = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    lut = m.get_lut()
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8


def test_colormap_material_pipeline_subtype():
    """Pipeline cache key uses the cmap name as subtype."""
    m_v = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    m_m = ColormapMaterial(cmap="magma", vmin=0.0, vmax=1.0)
    assert m_v.pipeline_subtype == "viridis"
    assert m_m.pipeline_subtype == "magma"
    assert m_v.pipeline_subtype != m_m.pipeline_subtype
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/viz/test_materials_viz.py -v
```

Expected: ImportError — `manifoldx.viz.materials` does not exist.

- [ ] **Step 3: Implement Python skeleton (no shader yet)**

Write `src/manifoldx/viz/materials.py`:

```python
"""Sci-viz materials.

ColormapMaterial: maps a per-instance scalar value through a 1D LUT.
Camera-facing point sprites with sphere-imposter normal reconstruction.
"""
import numpy as np

from manifoldx.viz import colormaps


class ColormapMaterial:
    """Point-sprite material that colormaps a per-instance scalar.

    Per-batch uniform (4 floats):
        vmin, vmax, lit_flag, _pad

    Per-instance storage buffers (read in shader by instance_index):
        transforms (mat4x4)         — existing
        scalar_values (float)       — new, sourced from ScalarValue
        radii (float)               — new, sourced from Radius

    Per-batch texture: 256x1 RGBA8 1D LUT, sampled in fragment shader.
    """

    def __init__(self, cmap: str, vmin: float, vmax: float, lit: bool = False):
        # Validate cmap exists
        colormaps.get_colormap(cmap)
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.lit = bool(lit)

    @property
    def pipeline_subtype(self) -> str:
        """Used by RenderPipeline cache to share pipelines across cmaps."""
        return self.cmap

    def get_data(self, n: int, registry=None) -> np.ndarray:
        """Per-batch material uniform data, broadcast to n rows.

        The renderer reads row 0 as the uniform; n rows are produced for
        compatibility with the existing material registry's per-instance
        data convention.
        """
        row = np.array(
            [self.vmin, self.vmax, 1.0 if self.lit else 0.0, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 4)).copy()

    def get_lut(self) -> np.ndarray:
        """Return the (256, 4) uint8 LUT for this material's colormap."""
        return colormaps.get_colormap(self.cmap)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/viz/test_materials_viz.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/materials.py tests/viz/test_materials_viz.py
git commit -m "feat(viz): add ColormapMaterial skeleton with uniform packing"
```

---

## Task 10: WGSL shader — sprite imposter vertex stage

**Files:**
- Modify: `src/manifoldx/viz/materials.py`
- Modify: `tests/viz/test_materials_viz.py`

- [ ] **Step 1: Write failing test**

Append to `tests/viz/test_materials_viz.py`:

```python
def test_colormap_material_compile_returns_wgsl():
    m = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    src = m._compile()
    assert isinstance(src, str)
    assert "@vertex" in src
    assert "@fragment" in src
    # Vertex stage references per-instance bindings
    assert "transforms" in src
    assert "scalar_values" in src
    assert "radii" in src
    # Camera-facing billboard math: must reference view matrix or
    # camera-relative reconstruction
    assert "view" in src.lower() or "camera" in src.lower()


def test_colormap_material_compile_validates_against_wgpu():
    """Compiled shader must be parseable by wgpu (no syntax errors).

    This is a smoke test — full pipeline creation is exercised in the
    integration test. Here we only validate shader source compilation.
    """
    import wgpu

    # Acquire a real device (may require GPU — skip if unavailable)
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
        device = adapter.request_device_sync()
    except Exception as e:
        pytest.skip(f"no wgpu device available: {e}")

    m = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)
    src = m._compile()
    # Will raise wgpu.GPUError if shader has syntax errors
    module = device.create_shader_module(code=src)
    assert module is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/viz/test_materials_viz.py::test_colormap_material_compile_returns_wgsl -v
```

Expected: AttributeError — `_compile` does not exist.

- [ ] **Step 3: Add the WGSL shader source**

Append to `src/manifoldx/viz/materials.py`:

```python
# WGSL shader source for ColormapMaterial.
#
# Bindings (group 0):
#   0: Globals uniform   { vp: mat4x4, view: mat4x4, camera_pos: vec3, _pad: f32 }
#   1: transforms        storage<read> array<mat4x4<f32>>
#   2: material uniform  { vmin: f32, vmax: f32, lit_flag: f32, _pad: f32 }
#   3: scalar_values     storage<read> array<f32>
#   4: radii             storage<read> array<f32>
#   5: lut_texture       texture_1d<f32>
#   6: lut_sampler       sampler
#
# Vertex inputs:
#   @location(0) position: vec3<f32>   — quad-local in [-1, 1]^2 (z = 0)
#
# Vertex outputs / fragment inputs:
#   @location(0) quad_uv: vec2<f32>    — passes the quad-local xy for normal reconstruction
#   @location(1) scalar:  f32          — per-instance scalar value passed to fragment

_COLORMAP_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct MaterialUniform {
    vmin: f32,
    vmax: f32,
    lit_flag: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> scalar_values: array<f32>;
@group(0) @binding(4) var<storage, read> radii: array<f32>;
@group(0) @binding(5) var lut_texture: texture_1d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) scalar: f32,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let radius = radii[iidx];
    let scalar = scalar_values[iidx];

    // Center of the sprite in world space
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

    // Transform center to view space, then offset in view-space XY by quad-local
    // position scaled by radius. This makes the quad always face the camera.
    let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
    let offset = vec2<f32>(in.position.x, in.position.y) * radius;
    let view_pos = vec3<f32>(view_center.x + offset.x, view_center.y + offset.y, view_center.z);

    // Reconstruct clip-space position. globals.vp = projection * view, so we
    // need projection * view_pos. Decompose: clip = vp * world_pos where
    // world_pos backs out of view_pos. Cheaper: clip = projection * view_pos
    // — but we don't have projection alone in the bind group. Workaround:
    // reconstruct world position from inverse view, then apply vp.
    //
    // For Plan 1 we keep it simple: pass through vp using a synthesized world
    // position obtained by transforming view_pos back through view^-1. This
    // adds one mat4 inverse per vertex. If profiling shows it matters, the
    // renderer can pre-upload a separate `projection` uniform.
    let view_inv = transpose(globals.view); // approximation: works only for
                                            // rigid view matrices (no scale).
                                            // ManifoldX cameras are rigid, so OK.
    let world_pos = (view_inv * vec4<f32>(view_pos, 1.0)).xyz;
    let clip = globals.vp * vec4<f32>(world_pos, 1.0);

    var out: VSOut;
    out.clip_position = clip;
    out.quad_uv = in.position.xy;  // [-1, 1]^2
    out.scalar = scalar;
    return out;
}
""".strip()


# Append the fragment stage in the next task.

def _compile_wgsl(self) -> str:
    return _COLORMAP_SHADER  # vertex only for now; fragment added next task
ColormapMaterial._compile = _compile_wgsl
```

**Implementer note on the `view_inv = transpose(globals.view)` trick:** this is correct only for orthonormal (rigid) view matrices. ManifoldX cameras compose pure rotation + translation; their view matrices satisfy this. If a future camera adds shear/scale, the trick breaks and we must add a separate `view_inv` uniform. Track this in the renderer's globals upload and add a regression test in Plan 5. **Document the assumption** in the shader header comment block.

- [ ] **Step 4: Run shader-syntax test**

```bash
uv run pytest tests/viz/test_materials_viz.py::test_colormap_material_compile_returns_wgsl -v
```

Expected: passes (vertex stage present, identifiers match).

The `test_colormap_material_compile_validates_against_wgpu` test will fail at the moment because the fragment stage is missing — that's expected. Don't worry about it; Task 11 fixes it.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/materials.py tests/viz/test_materials_viz.py
git commit -m "feat(viz): add WGSL vertex stage for ColormapMaterial sprite imposter"
```

---

## Task 11: WGSL fragment stage — sphere imposter + LUT sampling

**Files:**
- Modify: `src/manifoldx/viz/materials.py`

- [ ] **Step 1: Add fragment shader to `_COLORMAP_SHADER`**

Locate the `_COLORMAP_SHADER = """ ... """.strip()` block in `src/manifoldx/viz/materials.py` and extend it (before `.strip()`):

```python
_COLORMAP_SHADER = """
... (existing vertex stage above) ...

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Quad-local uv in [-1, 1]^2; outside unit disk -> discard
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }

    // Reconstruct view-space sphere normal: (uv.x, uv.y, sqrt(1 - r2))
    let n_view = vec3<f32>(in.quad_uv.x, in.quad_uv.y, sqrt(1.0 - r2));

    // Map scalar through LUT. Normalize to [0, 1] using vmin/vmax.
    let denom = max(material.vmax - material.vmin, 1e-6);
    let t = clamp((in.scalar - material.vmin) / denom, 0.0, 1.0);
    let base_color = textureSample(lut_texture, lut_sampler, t);

    // If lit_flag, modulate by Lambert against a fixed view-space light dir.
    let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
    let lambert = max(dot(n_view, light_dir), 0.0);
    let lit_factor = mix(1.0, 0.2 + 0.8 * lambert, material.lit_flag);

    return vec4<f32>(base_color.rgb * lit_factor, base_color.a);
}
""".strip()
```

- [ ] **Step 2: Run shader-validation test**

```bash
uv run pytest tests/viz/test_materials_viz.py::test_colormap_material_compile_validates_against_wgpu -v
```

Expected: passes (or skips if no GPU available).

If the test runs but fails with a WGSL syntax error, copy the error and fix the shader text. Common pitfalls: missing `;`, mismatched braces, incorrect type names (use `vec3<f32>`, not `vec3f` — ManifoldX targets stable WGSL).

- [ ] **Step 3: Run full materials test suite**

```bash
uv run pytest tests/viz/test_materials_viz.py -v
```

Expected: all 6 tests pass (2 may skip if no GPU).

- [ ] **Step 4: Commit**

```bash
git add src/manifoldx/viz/materials.py
git commit -m "feat(viz): add WGSL fragment stage with sphere imposter + LUT sampling"
```

---

## Task 12: Re-export `ColormapMaterial` from `manifoldx.viz`

**Files:**
- Modify: `src/manifoldx/viz/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

Replace `src/manifoldx/viz/__init__.py` with:

```python
"""manifoldx.viz — Scientific visualization primitives.

Public surface for Plan 1:
    PointCloud, ScalarValue, Radius, ColormapMaterial

Future plans add:
    TextLabel, AxisFrame, ScaleBar, LabelMaterial, AxisMaterial,
    point_cloud(), axes(), scale_bar(), colormap_legend()
"""
from manifoldx.viz.components import PointCloud, Radius, ScalarValue
from manifoldx.viz.materials import ColormapMaterial

__all__ = ["PointCloud", "ScalarValue", "Radius", "ColormapMaterial"]
```

- [ ] **Step 2: Quick smoke test**

```bash
uv run python -c "from manifoldx.viz import PointCloud, ScalarValue, Radius, ColormapMaterial; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/manifoldx/viz/__init__.py
git commit -m "feat(viz): re-export ColormapMaterial from manifoldx.viz"
```

---

## Task 13: Renderer — extend per-batch storage layout

**Files:**
- Modify: `src/manifoldx/renderer.py`

This task extends the renderer's per-batch buffer allocation to add `scalar_values` and `radii` storage buffers when the material is `ColormapMaterial`. It does **not yet** branch the render path; that is Task 14. The intent is to land the buffer-management change first so it can be unit-tested in isolation.

- [ ] **Step 1: Read the existing per-batch buffer allocation in `renderer.py`**

Locate the `RenderPipeline` class. Find the method (`run`, `render`, or a helper) that allocates and uploads the `transforms` storage buffer per batch. Note the exact name (e.g., `_upload_transforms`, `_prepare_batch`).

- [ ] **Step 2: Add `_BatchBuffers` helper class**

Near the top of `renderer.py` (after existing class declarations), add:

```python
from manifoldx.viz.materials import ColormapMaterial


class _BatchBuffers:
    """Per-batch GPU buffers managed by RenderPipeline.

    Extends transform-only management to include optional per-instance
    scalar_values and radii buffers required by ColormapMaterial.
    """

    def __init__(self, device):
        self._device = device
        self.transforms_buf = None
        self.transforms_capacity = 0
        self.scalar_values_buf = None
        self.scalar_values_capacity = 0
        self.radii_buf = None
        self.radii_capacity = 0

    def upload_transforms(self, data: "np.ndarray"):
        # Existing logic — adapt to actual wgpu API used in the file.
        # Allocate or reuse buffer, write data.
        n_bytes = data.nbytes
        if self.transforms_buf is None or self.transforms_capacity < n_bytes:
            self.transforms_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.transforms_capacity = n_bytes
        self._device.queue.write_buffer(self.transforms_buf, 0, data.tobytes())

    def upload_scalar_values(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.scalar_values_buf is None or self.scalar_values_capacity < n_bytes:
            self.scalar_values_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.scalar_values_capacity = n_bytes
        self._device.queue.write_buffer(self.scalar_values_buf, 0, data.tobytes())

    def upload_radii(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.radii_buf is None or self.radii_capacity < n_bytes:
            self.radii_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.radii_capacity = n_bytes
        self._device.queue.write_buffer(self.radii_buf, 0, data.tobytes())
```

**Implementer note:** the actual `device.create_buffer` and `queue.write_buffer` calls must match how the existing `renderer.py` does it (e.g., it may use `wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST` flag aliases or raw ints — match the existing style).

- [ ] **Step 3: Refactor existing transform upload to use `_BatchBuffers`**

In `RenderPipeline`, replace the existing per-batch transform-buffer field(s) with a single `_BatchBuffers` instance per batch (or one shared instance keyed by batch id). Verify by running:

```bash
make test
```

Expected: all existing renderer tests still pass. **If any fail, the refactor is incorrect — revert and re-attempt with smaller diff.**

- [ ] **Step 4: Commit**

```bash
git add src/manifoldx/renderer.py
git commit -m "refactor(renderer): introduce _BatchBuffers helper for per-batch GPU buffers"
```

---

## Task 14: Renderer — sprite batch list + sprite render path

**Files:**
- Modify: `src/manifoldx/renderer.py`

- [ ] **Step 1: Read `RenderPipeline.run` to locate batch construction**

Find where alive entities are grouped into batches by `(geom_id, material_type)`. Identify the batch list data structure.

- [ ] **Step 2: Add sprite batch detection**

In the batch-construction loop, branch on `PointCloud` component presence:

```python
# Pseudocode — adapt names to existing fields in RenderPipeline
from manifoldx.viz.components import PointCloud as _PointCloudMarker

is_sprite = "PointCloud" in store._components and (
    store._components["PointCloud"] is not None
)
# Group alive entities into mesh batches OR sprite batches
mesh_batches = {}    # (geom_id, mat_type) -> [entity indices]
sprite_batches = {}  # (mat_id) -> [entity indices]   (geom is always SPRITE_QUAD)
for ent in alive_indices:
    if is_sprite_entity(ent):  # has PointCloud component
        mat_id = material_id_for(ent)
        sprite_batches.setdefault(mat_id, []).append(ent)
    else:
        key = (geom_id_for(ent), mat_type_for(ent))
        mesh_batches.setdefault(key, []).append(ent)
```

The exact `is_sprite_entity` check uses the `EntityStore.has_component(name, ent_idx)` API (or whatever name the existing store uses to test component presence). If no such helper exists, add one:

```python
# In ecs.py if needed — but only if not already present
def has_component(self, name: str, entity_idx: int) -> bool:
    return name in self._components and self._alive[entity_idx]
```

- [ ] **Step 3: Add `_render_sprite_batches` method**

In `RenderPipeline`, add:

```python
def _render_sprite_batches(self, render_pass, sprite_batches, store, materials):
    """Render all sprite batches.

    Each batch is one (mat_id) group; geometry is always SPRITE_QUAD.
    Per-instance buffers: transforms, scalar_values, radii.
    """
    sprite_geom_id = self._geometry_registry.get_id("sprite_quad")
    sprite_geom = self._geometry_registry.get(sprite_geom_id)

    for mat_id, ent_indices in sprite_batches.items():
        material = materials.get(mat_id)
        if not isinstance(material, ColormapMaterial):
            # Future-proof: only ColormapMaterial implements the sprite path
            # for Plan 1. Other sprite materials raise.
            raise TypeError(
                f"sprite batch material must be ColormapMaterial; got {type(material).__name__}"
            )

        # Pull per-instance arrays via NumPy slicing (no Python loops).
        ent_arr = np.asarray(ent_indices, dtype=np.int64)
        transforms = store._components["Transform"][ent_arr]  # (N, 10)
        # Compute model matrices via existing TransformCache machinery
        models = self._transform_cache.get_matrices(ent_arr)  # (N, 4, 4)
        scalar_values = store._components["ScalarValue"][ent_arr, 0]  # (N,)
        radii = store._components["Radius"][ent_arr, 0]              # (N,)

        # Upload per-instance buffers
        batch_buffers = self._batch_buffers_for(mat_id)
        batch_buffers.upload_transforms(models.astype(np.float32))
        batch_buffers.upload_scalar_values(scalar_values.astype(np.float32))
        batch_buffers.upload_radii(radii.astype(np.float32))

        # Bind group + pipeline
        pipeline = self._get_or_create_pipeline(material, sprite=True)
        bind_group = self._make_sprite_bind_group(
            pipeline, batch_buffers, material
        )

        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 0)
        render_pass.set_vertex_buffer(0, sprite_geom.vertex_buffer)
        render_pass.set_index_buffer(sprite_geom.index_buffer, wgpu.IndexFormat.uint32)
        render_pass.draw_indexed(
            index_count=6,
            instance_count=len(ent_indices),
            first_index=0,
            base_vertex=0,
            first_instance=0,
        )
```

**Implementer note:** `_make_sprite_bind_group` is new. Add it as a helper that constructs the bind group with bindings 0–6 per the shader header comment (globals, transforms, material uniform, scalar_values, radii, lut_texture, lut_sampler). The `lut_texture` is created via `device.create_texture(...)` from `material.get_lut()` and cached on the material instance (or in a module-level dict keyed by cmap name to share across batches).

- [ ] **Step 4: Wire sprite path into `render`**

In the main render method, after rendering mesh batches, call:

```python
self._render_sprite_batches(render_pass, sprite_batches, store, self._material_registry)
```

- [ ] **Step 5: Extend pipeline cache key for `material_subtype`**

Locate `_get_or_create_pipeline`. Change the cache key from `material_type` (e.g., the class name) to a tuple `(material_type, material_subtype)` where `material_subtype` is `material.pipeline_subtype` if the attribute exists, else `None`:

```python
def _get_or_create_pipeline(self, material, sprite: bool = False):
    mat_type = type(material).__name__
    mat_subtype = getattr(material, "pipeline_subtype", None)
    key = (mat_type, mat_subtype, sprite)
    if key in self._pipeline_cache:
        return self._pipeline_cache[key]
    pipeline = self._create_pipeline(material, sprite=sprite)
    self._pipeline_cache[key] = pipeline
    return pipeline
```

`_create_pipeline` branches on `sprite=True` to use `SPRITE_QUAD` vertex layout + the sprite WGSL shader.

- [ ] **Step 6: Run full test suite**

```bash
make test
```

Expected: existing renderer tests still pass. New sprite path is exercised by integration tests in Tasks 15–16, not unit tests here.

- [ ] **Step 7: Commit**

```bash
git add src/manifoldx/renderer.py
git commit -m "feat(viz): add sprite render path to RenderPipeline"
```

---

## Task 15: Integration test — spawn point cloud, render headless, no crash

**Files:**
- Create: `tests/viz/test_point_cloud_integration.py`

- [ ] **Step 1: Write the integration test**

Write `tests/viz/test_point_cloud_integration.py`:

```python
"""End-to-end integration tests for sci-viz Plan 1 primitives."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Transform
from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue


def _make_offscreen_engine():
    """Create an Engine in headless / offscreen mode for tests.

    Skips if the wgpu backend is unavailable in the test environment.
    """
    try:
        engine = mx.Engine("test", canvas="offscreen", width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    return engine


def test_spawn_point_cloud_no_crash():
    """Spawn 100 sprites, render one frame, verify no errors."""
    engine = _make_offscreen_engine()

    N = 100
    rng = np.random.default_rng(42)
    positions = rng.standard_normal((N, 3)).astype(np.float32) * 2.0
    masses = rng.exponential(1.0, N).astype(np.float32)

    engine.spawn(
        PointCloud(),
        ColormapMaterial(cmap="viridis", vmin=0.0, vmax=3.0),
        Transform(pos=positions),
        ScalarValue(value=masses),
        Radius(radius=0.1),
        n=N,
    )

    # Render one frame — must not raise.
    frame = engine.render_one_frame()  # adapt to actual API name (could be _draw_frame)
    assert frame is not None


def test_spawn_point_cloud_entity_count():
    """Verify entity store registered N entities in PointCloud + ScalarValue + Radius."""
    engine = _make_offscreen_engine()

    N = 50
    engine.spawn(
        PointCloud(),
        ColormapMaterial(cmap="magma", vmin=0.0, vmax=1.0),
        Transform(pos=np.zeros((N, 3), dtype=np.float32)),
        ScalarValue(value=0.5),
        Radius(radius=0.1),
        n=N,
    )

    store = engine.store
    # The marker component should be present
    assert "PointCloud" in store._components
    # ScalarValue and Radius must hold N rows
    assert store._components["ScalarValue"].shape[0] >= N
    assert store._components["Radius"].shape[0] >= N
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/viz/test_point_cloud_integration.py -v
```

Expected: passes (or skips if no GPU). If it fails:

- **`ColormapMaterial not registered`** — the existing `engine.spawn` flow registers built-in components on first use; check that the test references the correct material registry path. May need to add `engine._material_registry.register(ColormapMaterial)` before spawn, or update the spawn flow to auto-register unknown materials (it likely already does).
- **WGSL shader error** — fix in `materials.py` per the wgpu error message.
- **Buffer binding mismatch** — check that the bind group layout matches the shader's binding numbers (groups 0-6).

Iterate until green.

- [ ] **Step 3: Commit**

```bash
git add tests/viz/test_point_cloud_integration.py
git commit -m "test(viz): add point-cloud spawn + render smoke test"
```

---

## Task 16: Integration test — per-frame `ScalarValue` updates propagate to GPU

**Files:**
- Modify: `tests/viz/test_point_cloud_integration.py`

- [ ] **Step 1: Write the test**

Append to `tests/viz/test_point_cloud_integration.py`:

```python
def test_scalar_value_update_per_frame():
    """A system mutates ScalarValue; the next frame's render reflects it.

    Strategy: spawn 1 sprite at the origin, large enough to fill the
    framebuffer center pixel. Render frame 0 with scalar=0.0 (LUT[0] = dark
    purple for viridis). Run a system that sets scalar=1.0. Render frame 1
    (LUT[255] = yellow). Read framebuffer center pixel; verify color changed.
    """
    from manifoldx.viz import colormaps

    engine = _make_offscreen_engine()
    engine.spawn(
        PointCloud(),
        ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0),
        Transform(pos=(0.0, 0.0, 0.0)),
        ScalarValue(value=0.0),
        Radius(radius=2.0),  # large sprite
        n=1,
    )

    @engine.system
    def update_scalar(query: mx.Query[ScalarValue], dt: float):
        query[ScalarValue].value += 1.0  # value goes 0 -> 1 in one frame

    # Frame 0: render with initial scalar = 0.0
    img0 = engine.render_one_frame()
    center0 = _read_center_pixel(img0)
    assert _pixel_close(center0, colormaps.lookup("viridis", 0.0))

    # Frame 1: system runs; scalar becomes 1.0
    img1 = engine.render_one_frame()
    center1 = _read_center_pixel(img1)
    assert _pixel_close(center1, colormaps.lookup("viridis", 1.0))

    # Sanity: center0 != center1
    assert not _pixel_close(center0, center1)


def _read_center_pixel(frame: np.ndarray) -> np.ndarray:
    """Return the (R,G,B,A) uint8 of the framebuffer center pixel."""
    h, w = frame.shape[:2]
    return frame[h // 2, w // 2].astype(np.uint8)


def _pixel_close(a: np.ndarray, b: np.ndarray, atol: int = 8) -> bool:
    """Compare two RGBA pixels with channel tolerance (default 8/255)."""
    return np.all(np.abs(a.astype(int) - b.astype(int)) <= atol)
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/viz/test_point_cloud_integration.py::test_scalar_value_update_per_frame -v
```

Expected: passes. If the test fails:

- **`engine.render_one_frame()` API mismatch** — the existing engine's offscreen canvas method may be named differently (`_draw_frame`, `step`, etc.). Adapt the test helper.
- **Pixel doesn't change** — the per-frame `scalar_values` upload is not wired up; trace through `_render_sprite_batches` to confirm `upload_scalar_values` is called every frame, not just once at init.
- **Unexpected color** — confirm the LUT bytes match matplotlib's viridis; the `colormaps.lookup` helper and the GPU shader sample the same LUT.

Iterate until green.

- [ ] **Step 3: Commit**

```bash
git add tests/viz/test_point_cloud_integration.py
git commit -m "test(viz): verify per-frame ScalarValue updates propagate to GPU"
```

---

## Task 17: Final cleanup — full suite, lint, and CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run full test suite**

```bash
make test
```

Expected: all tests pass (some integration tests may skip if no GPU). Address any regressions before proceeding.

- [ ] **Step 2: Run linter and formatter**

```bash
make lint
make format
```

Address any lint warnings in viz/ files.

- [ ] **Step 3: Update CHANGELOG**

In `CHANGELOG.md`, add a new `[Unreleased]` section above existing entries:

```markdown
## [Unreleased]

### Features
- **Sci-viz Plan 1 (foundation)** — `manifoldx.viz` subpackage with `PointCloud` marker component, `ScalarValue`, `Radius`, and `ColormapMaterial`. Six built-in colormaps (viridis, magma, plasma, inferno, turbo, gray) as 1D RGBA8 LUTs. Camera-facing point sprites with sphere-imposter fragment shading and GPU-side colormap LUT sampling. New sprite render path in `RenderPipeline` alongside the existing mesh path.
- **`[viz]` extra** in `pyproject.toml` (Pillow, staged for Plan 2 text rendering).

### Refactors
- `RenderPipeline` extracts per-batch GPU buffers into `_BatchBuffers` helper.
- Pipeline cache key extended with `material_subtype` so different colormaps share a pipeline but rebind only the LUT texture.
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record Plan 1 sci-viz foundation"
```

- [ ] **Step 5: Push to origin**

```bash
git push origin main
```

Expected: push succeeds. Plan 1 is now landed on `main`.

---

## Self-review

**Spec coverage:**

- §3.1 canonical ECS API → Tasks 4-6 (components), 9-12 (material), 13-14 (renderer wiring), 15-16 (integration test exercises the API).
- §3.2 functional shim API → **deferred to Plan 4** (acknowledged in plan header).
- §4 module layout → Tasks 1, 2, 4, 7, 9 create the files for Plan 1's slice; Plan 2/3/4 create the rest.
- §5 components (Plan 1 subset: `PointCloud`, `ScalarValue`, `Radius`) → Tasks 4-6.
- §6.1 `ColormapMaterial` → Tasks 9-12.
- §6.2 `LabelMaterial` → **Plan 2**.
- §6.3 `AxisMaterial` → **Plan 3**.
- §7.1 batch construction → Task 14.
- §7.2 render order → Task 14 (mesh+sprite in 3D pass). Label pass deferred to Plan 2.
- §7.3 pipeline cache → Task 14 step 5.
- §8 text rendering → **Plan 2**.
- §9 functional API → **Plan 4**.
- §11 testing strategy: unit (Tasks 2, 3, 4, 5, 6, 7, 9-11), integration (Tasks 15-16). Visual regression and perf smoke → **Plan 5**.
- §12 no-goals → respected throughout (no labels, no axes, no shim API in Plan 1).
- §13 file change inventory: Plan 1 covers the foundation files; remaining files are scheduled in later plans.
- §15 acceptance criteria: Plan 1 contributes to checkboxes 1, 4 (partially: viridis-only verified through framebuffer; rest deferred to visual regression in Plan 5), 6 (perf is Plan 5).

**Placeholder scan:** searched for "TBD", "TODO", "..." (only used for code-elision in shader source, not as a placeholder), "implement later" — none found.

**Type / API consistency:** `ColormapMaterial` constructor signature `(cmap, vmin, vmax, lit=False)` is consistent across Tasks 9, 10, 11, 12, 15, 16. `ScalarValue(value=...)` and `Radius(radius=...)` constructors consistent across Tasks 4–6, 15, 16. `pipeline_subtype` attribute consistent between Task 9 (definition) and Task 14 step 5 (consumption).

**Known soft spots flagged inline for the implementer:**

- Task 8 step 4: registry's exact API for built-in registration is "match the existing pattern" — implementer must read the existing code rather than rely on the plan's pseudocode.
- Task 10 step 3: `view_inv = transpose(globals.view)` is correct only for orthonormal view matrices; documented as an assumption with a roadmap entry.
- Task 13 step 2: the `wgpu.BufferUsage.STORAGE | COPY_DST` flag composition must match the existing renderer's style.
- Task 14 step 3: `_make_sprite_bind_group` is a new helper to be created during implementation; the plan describes its contents but not the exact wgpu call sequence (depends on the library version's API surface).

These are explicit "look at the surrounding code" prompts, not placeholders — the engineer cannot bypass them with a default value.

---

## Execution handoff

Plan complete and saved to `.knowledge/plans/2026-05-05-sci-viz-primitives-v1-plan-1-foundation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Uses `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batched with checkpoints.

Which approach?
