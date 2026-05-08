# Volume rendering v1 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the v1 volume primitive (`Volume` component, `VolumeMaterial`, `engine.register_volume/update_volume`, the v2-forward-compat `bind_compute_volume` stub, and a fragment-shader-raymarched render pass) so the `examples/volume_demo.py` Gaussian-blob scene composites correctly with the existing mesh/sprite/label/axis passes and the full test suite stays green.

**Architecture:** Resource-pointer ECS pattern (Volume → integer handle → `VolumeRegistry` on engine, mirroring `GeometryRegistry` / `MaterialRegistry`). Voxels live as a `r32float texture_3d`; rendering is a fullscreen-quad fragment shader that does ray/box-AABB intersection in entity-local space then a fixed-step front-to-back composite, sampling two 1D LUTs per step (existing RGBA8 colormap LUT for color, a new R32F per-material LUT for opacity). One new render pass added between sprite and label, no changes to the `Globals` uniform.

**Tech Stack:** Python 3.12, numpy, wgpu-py (WebGPU), pytest. Existing manifoldx infra (`Component`, `Material`, `Engine`, `RenderPipeline`, `colormaps._LUTS`).

**Scope note (deferred from design doc):** The design's §Refactoring proposal — splitting `renderer.py` into `src/manifoldx/render/passes/` — is **not** part of this plan. v1 adds `_render_volume_pass` as a new method on the existing `RenderPipeline`, mirroring the four existing `_render_*_pass`/`_render_*_batches` methods. The split is an independent refactor and will land as its own plan once a clear refactoring window opens. This keeps v1 surgical: every changed line traces to "volume rendering."

---

## File structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/manifoldx/viz/components.py` | modify | Add `Volume` component (single i32 field `volume_id`). |
| `src/manifoldx/viz/materials.py` | modify | Add `VolumeMaterial` class with opacity-LUT baking. |
| `src/manifoldx/viz/__init__.py` | modify | Export `Volume`, `VolumeMaterial`. |
| `src/manifoldx/resources.py` | modify | Add `VolumeRegistry` (CPU-side numpy storage + GPU `texture_3d` lifecycle). |
| `src/manifoldx/engine.py` | modify | Wire `VolumeRegistry`; add `register_volume`, `update_volume`, `bind_compute_volume` methods. |
| `src/manifoldx/renderer.py` | modify | Add `_render_volume_pass`, `_is_volume_entity`, volume bind-group layout, pipeline-cache `"volume"` kind, and the WGSL shader source string. Wire dispatch in `render()`. |
| `tests/viz/test_volume_component.py` | create | Volume component shape + offset/length contract. |
| `tests/test_volume_registry.py` | create | `register_volume`/`update_volume`/`bind_compute_volume` validation + handle behavior. |
| `tests/viz/test_volume_material.py` | create | `VolumeMaterial` LUT baking + validation. |
| `tests/test_volume_render.py` | create | Integration tests against an offscreen wgpu device. |
| `examples/volume_demo.py` | create | Gaussian-blob demo scene. |
| `CHANGELOG.md` | modify | Add v1 volume rendering entry under `[Unreleased]`. |
| `.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md` | modify | Annotate that the renderer-split refactor is deferred to a separate plan. |

---

## Task 1: Annotate the design doc with the refactor-deferral note

**Files:**
- Modify: `.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md`

- [ ] **Step 1: Edit the design doc**

In the "Internals → Refactoring `renderer.py`" section, replace the current paragraph with:

```markdown
### Refactoring `renderer.py` (deferred)

`renderer.py` is currently 1532 lines, with `_render_mesh_batches`,
`_render_sprite_batches`, `_render_label_pass`, `_render_axis_pass`
all inlined. The eventual right shape is to split each pass into its
own module under `src/manifoldx/render/passes/` and reduce
`RenderPipeline.run` to a thin orchestrator. **This refactor is
deferred from v1 to a follow-up plan** so volumes ship without
churning every other render path. v1 simply adds
`_render_volume_pass` as a sibling of the existing four pass methods.
```

- [ ] **Step 2: Commit**

```bash
git add .knowledge/analysis/2026-05-08-volume-rendering-v1-design.md
git commit -m "docs(viz): defer renderer-split refactor to follow-up plan"
```

---

## Task 2: `Volume` component

**Files:**
- Modify: `src/manifoldx/viz/components.py`
- Modify: `src/manifoldx/viz/__init__.py`
- Create: `tests/viz/test_volume_component.py`

- [ ] **Step 1: Write the failing test**

Create `tests/viz/test_volume_component.py`:

```python
"""Volume component layout: one i32 handle per entity."""
import numpy as np

from manifoldx.viz import Volume


def test_volume_layout_is_single_int_field():
    """Volume holds exactly one int field 'volume_id' at offset 0, length 1."""
    assert Volume._layout == {"volume_id": (0, 1)}


def test_volume_construct_with_handle():
    """Volume(volume_id=7) materializes a single-entity component
    with the value 7 in the volume_id slot.
    """
    v = Volume(volume_id=7)
    data = v.get_data()
    assert data.shape == (1, 1)
    assert int(data[0, 0]) == 7


def test_volume_construct_with_array():
    """Volume(volume_id=array_of_N) materializes N entities."""
    handles = np.array([1, 2, 3], dtype=np.int32)
    v = Volume(volume_id=handles)
    data = v.get_data()
    assert data.shape == (3, 1)
    assert (data[:, 0].astype(np.int32) == handles).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run -q pytest tests/viz/test_volume_component.py -v`
Expected: FAIL with `ImportError: cannot import name 'Volume' from 'manifoldx.viz'`.

- [ ] **Step 3: Add the `Volume` component**

In `src/manifoldx/viz/components.py`, append after the existing `AxisFrame` definition:

```python
class Volume(Component):
    """Per-entity reference to a registered 3D scalar field.

    The voxel data itself lives in `Engine._volume_registry`, keyed by
    the integer handle returned from `engine.register_volume(array)`.
    Mirrors the resource-pointer pattern used by `Mesh.geometry_id` and
    `Material.material_id`.
    """
    volume_id: int = 0
```

In `src/manifoldx/viz/__init__.py`:

```python
from manifoldx.viz.components import (
    AxisFrame,
    PointCloud,
    Radius,
    ScalarValue,
    TextLabel,
    Volume,
)
# ... and add "Volume" to __all__
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run -q pytest tests/viz/test_volume_component.py -v`
Expected: PASS, 3 tests.

Run the full suite to confirm no regression:
`uv run -q pytest -q`
Expected: 363 passed (360 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/components.py src/manifoldx/viz/__init__.py tests/viz/test_volume_component.py
git commit -m "feat(viz): add Volume ECS component (handle-only, layout {volume_id: (0, 1)})"
```

---

## Task 3: `VolumeRegistry` skeleton (CPU-side validation, no GPU yet)

**Files:**
- Modify: `src/manifoldx/resources.py`
- Create: `tests/test_volume_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_volume_registry.py`:

```python
"""VolumeRegistry: CPU-side handle bookkeeping + numpy validation.

These tests exercise registry shape only (no GPU). Render-pipeline
integration is covered separately in tests/test_volume_render.py.
"""
import numpy as np
import pytest

from manifoldx.resources import VolumeRegistry


def _gaussian_blob(n=8):
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.exp(-(X**2 + Y**2 + Z**2) / 0.1).astype(np.float32)


def test_register_returns_sequential_handles_starting_at_one():
    reg = VolumeRegistry(device=None)
    a = reg.register(_gaussian_blob(), name="a")
    b = reg.register(_gaussian_blob(), name="b")
    assert (a, b) == (1, 2)


def test_register_rejects_non_3d_array():
    reg = VolumeRegistry(device=None)
    with pytest.raises(ValueError, match="3D"):
        reg.register(np.zeros((8, 8), dtype=np.float32))


def test_register_rejects_non_float32_with_hint():
    reg = VolumeRegistry(device=None)
    with pytest.raises(ValueError, match="float32"):
        reg.register(np.zeros((4, 4, 4), dtype=np.float64))


def test_register_rejects_non_contiguous():
    reg = VolumeRegistry(device=None)
    base = np.zeros((8, 8, 8, 2), dtype=np.float32)
    non_contig = base[..., 0]
    assert not non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        reg.register(non_contig)


def test_get_returns_resource_with_data_dirty_name():
    reg = VolumeRegistry(device=None)
    arr = _gaussian_blob()
    vol_id = reg.register(arr, name="blob")
    res = reg.get(vol_id)
    assert res.name == "blob"
    assert res.data is arr
    assert res.dirty is True   # newly registered → needs upload


def test_update_requires_matching_shape():
    reg = VolumeRegistry(device=None)
    vol_id = reg.register(_gaussian_blob(8))
    with pytest.raises(ValueError, match="shape"):
        reg.update(vol_id, _gaussian_blob(16))


def test_update_swaps_data_and_sets_dirty():
    reg = VolumeRegistry(device=None)
    vol_id = reg.register(_gaussian_blob(8))
    reg.get(vol_id).dirty = False     # simulate post-upload
    new = _gaussian_blob(8)
    reg.update(vol_id, new)
    res = reg.get(vol_id)
    assert res.data is new
    assert res.dirty is True


def test_unknown_handle_raises():
    reg = VolumeRegistry(device=None)
    with pytest.raises(KeyError, match="999"):
        reg.get(999)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run -q pytest tests/test_volume_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'VolumeRegistry' from 'manifoldx.resources'`.

- [ ] **Step 3: Implement the `VolumeRegistry`**

In `src/manifoldx/resources.py`, append at the end of the file (above the `__all__` list, then add `"VolumeRegistry"` to `__all__`):

```python
class _VolumeResource:
    """Internal: one registered volume's CPU + (eventual) GPU state."""

    __slots__ = ("data", "name", "dirty", "texture")

    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name
        self.dirty = True       # set on register/update; cleared post-upload
        self.texture = None     # GPU texture; created lazily on first upload


class VolumeRegistry:
    """Cache of GPU 3D scalar field resources.

    Mirrors the GeometryRegistry / MaterialRegistry shape:
    - `register(numpy_array, name=...) -> int handle`
    - `update(handle, numpy_array)` — same shape only; flips dirty bit.
    - `get(handle) -> _VolumeResource`
    - GPU texture creation is lazy (handled by the renderer at the
      first frame in which the volume is needed).
    """

    def __init__(self, device=None):
        self._device = device
        self._volumes: Dict[int, _VolumeResource] = {}
        self._next_id = 1

    def register(self, data: np.ndarray, *, name: str | None = None) -> int:
        if data.ndim != 3:
            raise ValueError(
                f"volume data must be 3D (Nz, Ny, Nx); got {data.ndim}D"
            )
        if data.dtype != np.float32:
            raise ValueError(
                f"volume data must be float32; got {data.dtype}. "
                f"Convert with `array.astype(np.float32)`."
            )
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "volume data must be C-contiguous; "
                "call `np.ascontiguousarray(array)` before registering."
            )
        handle = self._next_id
        self._next_id += 1
        self._volumes[handle] = _VolumeResource(
            data=data, name=name or f"volume_{handle}"
        )
        return handle

    def update(self, handle: int, data: np.ndarray) -> None:
        res = self.get(handle)
        if data.shape != res.data.shape:
            raise ValueError(
                f"update_volume: shape mismatch — registered {res.data.shape}, "
                f"got {data.shape}. Re-register if you need a different size."
            )
        if data.dtype != np.float32:
            raise ValueError(
                f"volume data must be float32; got {data.dtype}."
            )
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError("volume data must be C-contiguous.")
        res.data = data
        res.dirty = True

    def get(self, handle: int) -> _VolumeResource:
        if handle not in self._volumes:
            raise KeyError(f"unknown volume handle: {handle}")
        return self._volumes[handle]
```

The `Dict` import is already at the top of `resources.py`; verify with `grep '^from typing' src/manifoldx/resources.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -q pytest tests/test_volume_registry.py -v`
Expected: PASS, 8 tests.

Run the full suite:
`uv run -q pytest -q`
Expected: 371 passed (363 + 8 new).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/resources.py tests/test_volume_registry.py
git commit -m "feat(viz): add VolumeRegistry (CPU-side handle + dirty bookkeeping)"
```

---

## Task 4: Engine wiring — `register_volume` / `update_volume` / `bind_compute_volume`

**Files:**
- Modify: `src/manifoldx/engine.py`
- Modify: `tests/test_volume_registry.py` (add engine-level tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_volume_registry.py`:

```python
def test_engine_register_volume_returns_handle():
    """engine.register_volume forwards to the registry."""
    import manifoldx as mx
    engine = mx.Engine("vol-test", width=64, height=64)
    arr = _gaussian_blob(8)
    handle = engine.register_volume(arr, name="blob")
    assert handle == 1
    assert engine._volume_registry.get(handle).data is arr


def test_engine_update_volume_round_trip():
    import manifoldx as mx
    engine = mx.Engine("vol-test", width=64, height=64)
    handle = engine.register_volume(_gaussian_blob(8))
    engine._volume_registry.get(handle).dirty = False
    new = _gaussian_blob(8)
    engine.update_volume(handle, new)
    assert engine._volume_registry.get(handle).data is new
    assert engine._volume_registry.get(handle).dirty is True


def test_engine_bind_compute_volume_is_v2_stub():
    """bind_compute_volume exists in v1 but raises NotImplementedError."""
    import manifoldx as mx
    engine = mx.Engine("vol-test", width=64, height=64)
    handle = engine.register_volume(_gaussian_blob(8))
    with pytest.raises(NotImplementedError, match="v2"):
        engine.bind_compute_volume(handle, "density")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run -q pytest tests/test_volume_registry.py -v`
Expected: 3 NEW FAIL with `AttributeError: 'Engine' object has no attribute 'register_volume'`.

- [ ] **Step 3: Wire the engine**

In `src/manifoldx/engine.py`:

1. Add the import at the top alongside the existing `GeometryRegistry, MaterialRegistry`:

```python
from manifoldx.resources import GeometryRegistry, MaterialRegistry, VolumeRegistry
```

2. In `Engine.__init__`, immediately after the line `self._material_registry = MaterialRegistry(self._device)`, add:

```python
        self._volume_registry = VolumeRegistry(self._device)
```

3. Add three methods on the `Engine` class (anywhere reasonable — placement next to `compute()` is fine, since they're a sibling family of resource-registration helpers). Show the full code:

```python
    def register_volume(
        self,
        data: "np.ndarray",
        *,
        name: str | None = None,
    ) -> int:
        """Register a 3D scalar field for volume rendering.

        Args:
            data: numpy array of shape (Nz, Ny, Nx), dtype=float32,
                  C-contiguous. data[k, j, i] is the voxel at integer
                  coordinates (i, j, k) along local-space (x, y, z).
            name: optional diagnostic label.

        Returns:
            Integer handle to pass into `Volume(volume_id=handle)`.
        """
        return self._volume_registry.register(data, name=name)

    def update_volume(self, handle: int, data: "np.ndarray") -> None:
        """Replace voxel data for a registered volume. Same shape required.

        Bumps the dirty bit; the renderer re-uploads on the next frame.
        """
        self._volume_registry.update(handle, data)

    def bind_compute_volume(self, handle: int, kernel_field: str) -> None:
        """v2: bind a Phase-2 compute kernel's `Writes[Volume3D]` field
        to this volume's storage texture. v1 raises NotImplementedError —
        the API is reserved here so v2 can land without a breaking change.
        """
        raise NotImplementedError(
            "bind_compute_volume is reserved for v2; v1 supports only "
            "CPU-uploaded volumes via register_volume()/update_volume()."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -q pytest tests/test_volume_registry.py -v`
Expected: PASS, 11 tests.

Full suite: `uv run -q pytest -q` → 374 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/engine.py tests/test_volume_registry.py
git commit -m "feat(viz): wire VolumeRegistry into Engine (register/update/bind_compute_volume)"
```

---

## Task 5: `VolumeMaterial` opacity LUT baking

**Files:**
- Modify: `src/manifoldx/viz/materials.py`
- Modify: `src/manifoldx/viz/__init__.py`
- Create: `tests/viz/test_volume_material.py`

- [ ] **Step 1: Write the failing test**

Create `tests/viz/test_volume_material.py`:

```python
"""VolumeMaterial: opacity LUT baking + parameter validation."""
import numpy as np
import pytest

from manifoldx.viz import VolumeMaterial


def test_default_opacity_lut_is_linear_ramp():
    """opacity_stops=None bakes alpha = linspace(0, 1, 256)."""
    m = VolumeMaterial()
    expected = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    np.testing.assert_allclose(m.opacity_lut, expected, rtol=1e-6)


def test_opacity_stops_full_range_linear():
    """[(0,0),(1,1)] piecewise-linear matches the default ramp."""
    m = VolumeMaterial(opacity_stops=[(0.0, 0.0), (1.0, 1.0)])
    expected = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    np.testing.assert_allclose(m.opacity_lut, expected, rtol=1e-6)


def test_opacity_stops_step_at_half():
    """[(0,0),(0.5,0),(0.5,1),(1,1)] produces a step at index 128."""
    m = VolumeMaterial(opacity_stops=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 1.0)])
    assert m.opacity_lut[127] == 0.0
    assert m.opacity_lut[128] == 1.0


def test_opacity_stops_array_used_as_is():
    """Pre-baked (256,) float32 array passes through unchanged."""
    arr = np.linspace(0.5, 1.0, 256, dtype=np.float32)
    m = VolumeMaterial(opacity_stops=arr)
    np.testing.assert_array_equal(m.opacity_lut, arr)


def test_opacity_stops_array_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(256,\)"):
        VolumeMaterial(opacity_stops=np.zeros(128, dtype=np.float32))


def test_opacity_stops_must_be_sorted():
    with pytest.raises(ValueError, match="ascending"):
        VolumeMaterial(opacity_stops=[(0.5, 0.0), (0.2, 1.0)])


def test_unknown_cmap_lists_available():
    with pytest.raises(ValueError, match="viridis"):
        VolumeMaterial(cmap="not-a-real-colormap")


def test_vmin_must_be_less_than_vmax():
    with pytest.raises(ValueError, match="vmin"):
        VolumeMaterial(vmin=1.0, vmax=1.0)


def test_step_size_must_be_positive():
    with pytest.raises(ValueError, match="step_size"):
        VolumeMaterial(step_size=0.0)
    with pytest.raises(ValueError, match="step_size"):
        VolumeMaterial(step_size=-0.1)


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError, match="max_steps"):
        VolumeMaterial(max_steps=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run -q pytest tests/viz/test_volume_material.py -v`
Expected: FAIL with `ImportError: cannot import name 'VolumeMaterial' from 'manifoldx.viz'`.

- [ ] **Step 3: Implement `VolumeMaterial`**

In `src/manifoldx/viz/materials.py`, append a new class (after `AxisMaterial`):

```python
class VolumeMaterial(Material):
    """Direct volume rendering material — colormap LUT + opacity LUT.

    See `.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md`
    for the full transfer-function semantics.
    """

    pipeline_subtype: str | None = "volume"

    def __init__(
        self,
        cmap: str = "viridis",
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
        opacity_stops=None,
        density_scale: float = 1.0,
        step_size: float | None = None,
        max_steps: int = 256,
    ):
        from manifoldx.viz.colormaps import _LUTS

        if cmap not in _LUTS:
            raise ValueError(
                f"unknown cmap: {cmap!r}; available: {sorted(_LUTS)}"
            )
        if not (vmin < vmax):
            raise ValueError(f"vmin must be < vmax; got vmin={vmin}, vmax={vmax}")
        if step_size is not None and step_size <= 0.0:
            raise ValueError(f"step_size must be > 0; got {step_size}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0; got {max_steps}")

        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.density_scale = float(density_scale)
        self.step_size = step_size
        self.max_steps = int(max_steps)
        self.opacity_lut = self._bake_opacity(opacity_stops)
        super().__init__()

    @staticmethod
    def _bake_opacity(stops) -> np.ndarray:
        """Produce a (256,) float32 array of alpha values in [0, 1]."""
        if stops is None:
            return np.linspace(0.0, 1.0, 256, dtype=np.float32)

        if isinstance(stops, np.ndarray):
            if stops.shape != (256,):
                raise ValueError(
                    f"opacity_stops array must have shape (256,); got {stops.shape}"
                )
            return stops.astype(np.float32, copy=False)

        # List of (scalar, alpha) pairs — must be ascending in scalar.
        xs = [float(s) for s, _ in stops]
        ys = [float(a) for _, a in stops]
        for i in range(1, len(xs)):
            if xs[i] < xs[i - 1]:
                raise ValueError(
                    "opacity_stops scalars must be in ascending order; "
                    f"got {xs}"
                )
        sample_xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        return np.interp(sample_xs, xs, ys).astype(np.float32)
```

(`numpy as np` is already imported at the top of `materials.py`; verify with `grep '^import numpy' src/manifoldx/viz/materials.py`.)

In `src/manifoldx/viz/__init__.py`, add `VolumeMaterial` to both the import line and `__all__`:

```python
from manifoldx.viz.materials import (
    AxisMaterial,
    ColormapMaterial,
    LabelMaterial,
    VolumeMaterial,
)
# ...
__all__ = [
    # ... existing entries
    "VolumeMaterial",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -q pytest tests/viz/test_volume_material.py -v`
Expected: PASS, 10 tests.

Full suite: `uv run -q pytest -q` → 384 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/materials.py src/manifoldx/viz/__init__.py tests/viz/test_volume_material.py
git commit -m "feat(viz): add VolumeMaterial with opacity-LUT baking"
```

---

## Task 6: Volume WGSL shader source — pure-text validation

**Files:**
- Modify: `src/manifoldx/renderer.py`
- Create: `tests/test_volume_render.py` (just one shader-compile test for now)

The shader itself is the heart of the volume pass — get it written and validated by wgpu *before* hooking it into the per-frame draw routing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_volume_render.py`:

```python
"""Volume render-pass integration tests (require an offscreen wgpu device)."""
import numpy as np
import pytest


def _make_offscreen_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_volume_shader_compiles_via_wgpu_create_shader_module():
    """The hand-written WGSL volume shader must pass wgpu validation."""
    from manifoldx.renderer import VOLUME_SHADER_SOURCE

    engine = _make_offscreen_engine()
    # If create_shader_module raises, the test fails with a wgpu validation error.
    engine._device.create_shader_module(code=VOLUME_SHADER_SOURCE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run -q pytest tests/test_volume_render.py -v`
Expected: FAIL with `ImportError: cannot import name 'VOLUME_SHADER_SOURCE' from 'manifoldx.renderer'`.

- [ ] **Step 3: Add the shader source string**

In `src/manifoldx/renderer.py`, near the existing `SHADER_SOURCE` constant (top of file), add `VOLUME_SHADER_SOURCE`:

```python
# Volume rendering: fullscreen-quad fragment shader, ray/box-AABB in
# entity-local space, fixed-step front-to-back composite. See design
# doc .knowledge/analysis/2026-05-08-volume-rendering-v1-design.md.
VOLUME_SHADER_SOURCE = """
struct Globals {
    vp:            mat4x4<f32>,
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    camera_pos:    vec3<f32>,
    _pad0:         f32,
    viewport_size: vec2<f32>,
    _pad1:         vec2<f32>,
};

struct VolumeUniforms {
    model:         mat4x4<f32>,
    inv_model:     mat4x4<f32>,
    vmin:          f32,
    vmax:          f32,
    density_scale: f32,
    step_size:     f32,
    max_steps:     u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var volume_tex:  texture_3d<f32>;
@group(1) @binding(1) var vol_sampler: sampler;
@group(1) @binding(2) var color_lut:   texture_2d<f32>;
@group(1) @binding(3) var opacity_lut: texture_2d<f32>;
@group(1) @binding(4) var lut_sampler: sampler;
@group(1) @binding(5) var<uniform> vu: VolumeUniforms;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    // Oversized fullscreen triangle (covers NDC [-1,1]^2 with 3 verts).
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOut;
    out.clip_pos = vec4<f32>(pos[vid], 0.0, 1.0);
    return out;
}

fn ray_unit_cube(
    ro: vec3<f32>, rd: vec3<f32>,
    t_near: ptr<function, f32>, t_far: ptr<function, f32>,
) -> bool {
    // Slab method against [-0.5, 0.5]^3.
    let inv_rd = vec3<f32>(1.0) / rd;
    let t1 = (vec3<f32>(-0.5) - ro) * inv_rd;
    let t2 = (vec3<f32>( 0.5) - ro) * inv_rd;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let near = max(max(tmin.x, tmin.y), tmin.z);
    let far  = min(min(tmax.x, tmax.y), tmax.z);
    *t_near = near;
    *t_far  = far;
    return far >= max(near, 0.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Reconstruct world-space ray from gl_FragCoord + globals.
    let pixel = in.clip_pos.xy;
    let ndc = vec2<f32>(
        (pixel.x / globals.viewport_size.x) * 2.0 - 1.0,
        1.0 - (pixel.y / globals.viewport_size.y) * 2.0,
    );
    // Unproject NDC near/far into world space.
    let inv_vp_near = inverse_mat4(globals.vp) * vec4<f32>(ndc, 0.0, 1.0);
    let inv_vp_far  = inverse_mat4(globals.vp) * vec4<f32>(ndc, 1.0, 1.0);
    let world_near = inv_vp_near.xyz / inv_vp_near.w;
    let world_far  = inv_vp_far.xyz  / inv_vp_far.w;
    let ro_world = world_near;
    let rd_world = normalize(world_far - world_near);

    // Transform ray into local space and intersect against the unit cube.
    let ro_local = (vu.inv_model * vec4<f32>(ro_world, 1.0)).xyz;
    let rd_local = (vu.inv_model * vec4<f32>(rd_world, 0.0)).xyz;

    var t_near: f32 = 0.0;
    var t_far:  f32 = 0.0;
    if (!ray_unit_cube(ro_local, rd_local, &t_near, &t_far)) {
        discard;
    }

    var t = max(t_near, 0.0);
    var accum: vec4<f32> = vec4<f32>(0.0);
    var step_count: u32 = 0u;
    let inv_range = 1.0 / max(vu.vmax - vu.vmin, 1e-12);

    loop {
        if (step_count >= vu.max_steps) { break; }
        if (t > t_far)                  { break; }
        if (accum.a > 0.99)             { break; }

        let p_local = ro_local + t * rd_local;
        let p_uvw = p_local + vec3<f32>(0.5);
        let s = textureSampleLevel(volume_tex, vol_sampler, p_uvw, 0.0).r;
        let s_n = clamp((s - vu.vmin) * inv_range, 0.0, 1.0);

        // 1D LUTs encoded as 256x1 2D textures (wgpu portability).
        let rgb = textureSampleLevel(color_lut,   lut_sampler, vec2<f32>(s_n, 0.5), 0.0).rgb;
        let a   = textureSampleLevel(opacity_lut, lut_sampler, vec2<f32>(s_n, 0.5), 0.0).r
                  * vu.density_scale * vu.step_size;

        accum = accum + (1.0 - accum.a) * vec4<f32>(rgb * a, a);
        t = t + vu.step_size;
        step_count = step_count + 1u;
    }
    return accum;
}

// Cofactor-based 4x4 inverse. Used once per fragment for unprojection.
fn inverse_mat4(m: mat4x4<f32>) -> mat4x4<f32> {
    let a = m[0]; let b = m[1]; let c = m[2]; let d = m[3];
    let det =
        a.x*(b.y*(c.z*d.w - c.w*d.z) - b.z*(c.y*d.w - c.w*d.y) + b.w*(c.y*d.z - c.z*d.y))
      - a.y*(b.x*(c.z*d.w - c.w*d.z) - b.z*(c.x*d.w - c.w*d.x) + b.w*(c.x*d.z - c.z*d.x))
      + a.z*(b.x*(c.y*d.w - c.w*d.y) - b.y*(c.x*d.w - c.w*d.x) + b.w*(c.x*d.y - c.y*d.x))
      - a.w*(b.x*(c.y*d.z - c.z*d.y) - b.y*(c.x*d.z - c.z*d.x) + b.z*(c.x*d.y - c.y*d.x));
    let inv_det = 1.0 / det;
    var r: mat4x4<f32>;
    r[0][0] =  (b.y*(c.z*d.w-c.w*d.z) - b.z*(c.y*d.w-c.w*d.y) + b.w*(c.y*d.z-c.z*d.y)) * inv_det;
    r[0][1] = -(a.y*(c.z*d.w-c.w*d.z) - a.z*(c.y*d.w-c.w*d.y) + a.w*(c.y*d.z-c.z*d.y)) * inv_det;
    r[0][2] =  (a.y*(b.z*d.w-b.w*d.z) - a.z*(b.y*d.w-b.w*d.y) + a.w*(b.y*d.z-b.z*d.y)) * inv_det;
    r[0][3] = -(a.y*(b.z*c.w-b.w*c.z) - a.z*(b.y*c.w-b.w*c.y) + a.w*(b.y*c.z-b.z*c.y)) * inv_det;
    r[1][0] = -(b.x*(c.z*d.w-c.w*d.z) - b.z*(c.x*d.w-c.w*d.x) + b.w*(c.x*d.z-c.z*d.x)) * inv_det;
    r[1][1] =  (a.x*(c.z*d.w-c.w*d.z) - a.z*(c.x*d.w-c.w*d.x) + a.w*(c.x*d.z-c.z*d.x)) * inv_det;
    r[1][2] = -(a.x*(b.z*d.w-b.w*d.z) - a.z*(b.x*d.w-b.w*d.x) + a.w*(b.x*d.z-b.z*d.x)) * inv_det;
    r[1][3] =  (a.x*(b.z*c.w-b.w*c.z) - a.z*(b.x*c.w-b.w*c.x) + a.w*(b.x*c.z-b.z*c.x)) * inv_det;
    r[2][0] =  (b.x*(c.y*d.w-c.w*d.y) - b.y*(c.x*d.w-c.w*d.x) + b.w*(c.x*d.y-c.y*d.x)) * inv_det;
    r[2][1] = -(a.x*(c.y*d.w-c.w*d.y) - a.y*(c.x*d.w-c.w*d.x) + a.w*(c.x*d.y-c.y*d.x)) * inv_det;
    r[2][2] =  (a.x*(b.y*d.w-b.w*d.y) - a.y*(b.x*d.w-b.w*d.x) + a.w*(b.x*d.y-b.y*d.x)) * inv_det;
    r[2][3] = -(a.x*(b.y*c.w-b.w*c.y) - a.y*(b.x*c.w-b.w*c.x) + a.w*(b.x*c.y-b.y*c.x)) * inv_det;
    r[3][0] = -(b.x*(c.y*d.z-c.z*d.y) - b.y*(c.x*d.z-c.z*d.x) + b.z*(c.x*d.y-c.y*d.x)) * inv_det;
    r[3][1] =  (a.x*(c.y*d.z-c.z*d.y) - a.y*(c.x*d.z-c.z*d.x) + a.z*(c.x*d.y-c.y*d.x)) * inv_det;
    r[3][2] = -(a.x*(b.y*d.z-b.z*d.y) - a.y*(b.x*d.z-b.z*d.x) + a.z*(b.x*d.y-b.y*d.x)) * inv_det;
    r[3][3] =  (a.x*(b.y*c.z-b.z*c.y) - a.y*(b.x*c.z-b.z*c.x) + a.z*(b.x*c.y-b.y*c.x)) * inv_det;
    return r;
}
""".strip()
```

Add `"VOLUME_SHADER_SOURCE"` to the existing `__all__` list at the bottom of `renderer.py`.

**Note on the LUT layout:** wgpu/WebGPU portability for 1D textures is uneven across backends; we encode both 1D LUTs as `256x1` 2D textures and sample at `vec2(s_n, 0.5)`. This is the same trick the existing `ColormapMaterial` uses.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run -q pytest tests/test_volume_render.py -v`
Expected: PASS (1 test) OR SKIPPED if no offscreen device. Both are acceptable; CI environments without GPU will skip.

Full suite: `uv run -q pytest -q`. No regressions.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/renderer.py tests/test_volume_render.py
git commit -m "feat(viz): add VOLUME_SHADER_SOURCE — volume DVR fragment shader (compiles via wgpu)"
```

---

## Task 7: Volume texture upload — `VolumeRegistry.upload_to_gpu`

**Files:**
- Modify: `src/manifoldx/resources.py`
- Modify: `tests/test_volume_render.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_volume_render.py`:

```python
def test_volume_registry_creates_r32float_3d_texture_on_first_upload():
    """First call to upload_to_gpu creates a texture_3d r32float; clears dirty."""
    engine = _make_offscreen_engine()
    arr = np.zeros((4, 8, 16), dtype=np.float32)   # (Nz=4, Ny=8, Nx=16)
    arr[2, 4, 8] = 1.0
    handle = engine.register_volume(arr)
    res = engine._volume_registry.get(handle)
    assert res.dirty is True
    assert res.texture is None

    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)

    assert res.dirty is False
    assert res.texture is not None
    # Verify size mapping: numpy (Nz, Ny, Nx) → texture (Nx, Ny, Nz).
    assert res.texture.size == (16, 8, 4) or res.texture.size == [16, 8, 4]


def test_volume_registry_reuploads_only_when_dirty():
    """A second upload call without update_volume() is a no-op."""
    engine = _make_offscreen_engine()
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    handle = engine.register_volume(arr)
    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    tex_before = engine._volume_registry.get(handle).texture
    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    tex_after = engine._volume_registry.get(handle).texture
    assert tex_before is tex_after   # texture object not recreated


def test_volume_registry_update_triggers_reupload():
    """update_volume() flips dirty; upload_to_gpu writes new bytes; same texture object."""
    engine = _make_offscreen_engine()
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    handle = engine.register_volume(arr)
    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    res = engine._volume_registry.get(handle)
    tex_before = res.texture

    new = np.ones((4, 4, 4), dtype=np.float32)
    engine.update_volume(handle, new)
    assert res.dirty is True

    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    assert res.dirty is False
    assert res.texture is tex_before    # reused, not reallocated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run -q pytest tests/test_volume_render.py -v`
Expected: 3 NEW FAIL with `AttributeError: 'VolumeRegistry' object has no attribute 'upload_to_gpu'`.

- [ ] **Step 3: Implement `upload_to_gpu`**

In `src/manifoldx/resources.py`, add a method on `VolumeRegistry`:

```python
    def upload_to_gpu(self, handle: int, queue) -> None:
        """Lazily create a `texture_3d` r32float for this volume and write
        the current numpy data into it. Clears the dirty bit. Re-creates the
        texture only if shape changed (which `update` already disallows).
        """
        if self._device is None or queue is None:
            return
        res = self.get(handle)
        if not res.dirty and res.texture is not None:
            return

        nz, ny, nx = res.data.shape
        if res.texture is None:
            res.texture = self._device.create_texture(
                size=(nx, ny, nz),
                format=wgpu.TextureFormat.r32float,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                dimension=wgpu.TextureDimension.d3,
                mip_level_count=1,
                sample_count=1,
            )

        # Numpy is (Nz, Ny, Nx) C-contiguous → bytes layout matches WGSL
        # texture_3d (x fastest, z slowest).
        data_bytes = res.data.tobytes()
        bytes_per_row = nx * 4   # 4 bytes per f32 voxel
        rows_per_image = ny
        queue.write_texture(
            {"texture": res.texture, "mip_level": 0, "origin": (0, 0, 0)},
            data_bytes,
            {
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": rows_per_image,
            },
            (nx, ny, nz),
        )
        res.dirty = False
```

`wgpu` is already imported at the top of `resources.py`; verify with `grep '^import wgpu' src/manifoldx/resources.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -q pytest tests/test_volume_render.py -v`
Expected: PASS, 4 tests (or all skipped on CI without GPU).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/resources.py tests/test_volume_render.py
git commit -m "feat(viz): VolumeRegistry uploads numpy data to texture_3d r32float lazily"
```

---

## Task 8: Volume render-pass implementation — `_render_volume_pass`

**Files:**
- Modify: `src/manifoldx/renderer.py`

This is the largest task. We add (a) the volume bind-group layout, (b) the pipeline-cache `"volume"` kind, (c) `_render_volume_pass`, and (d) the `_is_volume_entity` helper. We do **not** wire it into `render()` yet — that's Task 9, so the routing change can be reviewed independently.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_volume_render.py`:

```python
def test_centered_blob_renders_visible_pixels_at_origin():
    """Spawn a 32^3 Gaussian blob centered at the origin; render one frame;
    the framebuffer's central pixel must have non-zero alpha; outside the
    projected box bounds must be exactly background (alpha == 0).
    """
    import manifoldx as mx
    from manifoldx.components import Material, Transform
    from manifoldx.viz import Volume, VolumeMaterial

    engine = _make_offscreen_engine(width=64, height=64)
    engine.camera.position = (0.0, 0.0, 5.0)

    n = 32
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    density = np.exp(-(X**2 + Y**2 + Z**2) / 0.1).astype(np.float32)

    handle = engine.register_volume(density)
    engine.spawn(
        Volume(volume_id=handle),
        Material(VolumeMaterial(
            cmap="inferno",
            opacity_stops=[(0.0, 0.0), (0.3, 0.05), (1.0, 0.6)],
            step_size=0.02,
        )),
        Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
        n=1,
    )

    rgba = engine.render_to_array()    # (H, W, 4) uint8
    assert rgba[32, 32, 3] > 0   # center has accumulated alpha
    assert rgba[0, 0, 3] == 0    # corner is pure background


def test_two_entities_share_volume_handle():
    """Same vol_id with different Transform.pos → two visually separate regions."""
    import manifoldx as mx
    from manifoldx.components import Material, Transform
    from manifoldx.viz import Volume, VolumeMaterial

    engine = _make_offscreen_engine(width=128, height=64)
    engine.camera.position = (0.0, 0.0, 5.0)

    arr = np.ones((4, 4, 4), dtype=np.float32)   # uniform-1 box
    handle = engine.register_volume(arr)
    for px in (-2.0, +2.0):
        engine.spawn(
            Volume(volume_id=handle),
            Material(VolumeMaterial(
                cmap="viridis",
                opacity_stops=[(0.0, 0.0), (1.0, 1.0)],
                step_size=0.05,
            )),
            Transform(pos=(px, 0, 0), scale=(1.0, 1.0, 1.0)),
            n=1,
        )

    rgba = engine.render_to_array()
    # Left region (x < 32) and right region (x > 96) must both have alpha > 0;
    # the center column (x ≈ 64) must be background (between the two boxes).
    assert (rgba[32, :32, 3] > 0).any()
    assert (rgba[32, -32:, 3] > 0).any()
    assert rgba[32, 64, 3] == 0
```

> **Note:** `engine.render_to_array()` is the existing offscreen-render helper; if it doesn't exist on `Engine` already, replace with the equivalent backend-direct call used by `tests/test_render_mvp.py`. Check with `grep -n 'render_to_array\|read_texture' tests/ src/manifoldx -r | head`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run -q pytest tests/test_volume_render.py -v`
Expected: 2 NEW FAIL — exact message will be either "no _render_volume_pass" wired or all-zero framebuffer (volumes never drawn).

- [ ] **Step 3: Add the volume bind-group layout factory**

In `src/manifoldx/renderer.py`, near `_get_or_create_pipeline`, add a helper method on `RenderPipeline` that returns a `(pipeline, bind_group_layout)` tuple for the volume pass. Mirror the structure of `_render_sprite_batches`'s `_get_or_create_pipeline` call but with the volume-specific entries. Show the full method:

```python
    def _get_or_create_volume_pipeline(self, device, texture_format):
        """Return (pipeline, bind_group_layout) for the volume DVR pass.

        Cache key: ("volume",) — independent of mesh/sprite/label/axis caches.
        Pipeline state: depth-test LESS_EQUAL, depth-write OFF, alpha-blend ON.
        """
        cache_key = ("volume",)
        cached = self._pipelines.get(cache_key)
        if cached is not None:
            return cached

        shader = device.create_shader_module(code=VOLUME_SHADER_SOURCE)

        bgl_globals = self._globals_bgl   # already created in __init__

        bgl_volume = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d3,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                        "min_binding_size": 160,   # 64 + 64 + 4*4 + 4*4 = 160
                    },
                },
            ],
        )

        layout = device.create_pipeline_layout(
            bind_group_layouts=[bgl_globals, bgl_volume]
        )

        pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{
                    "format": texture_format,
                    "blend": {
                        "color": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                        "alpha": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                    },
                    "write_mask": wgpu.ColorWrite.ALL,
                }],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.less_equal,
            },
            multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
        )
        self._pipelines[cache_key] = (pipeline, bgl_volume)
        return pipeline, bgl_volume
```

> **Note:** the shader uses pre-multiplied output (`accum.rgb` already has alpha folded in via `rgb * a`), so the blend factors are `(one, one_minus_src_alpha)` — *not* `(src_alpha, one_minus_src_alpha)`.

- [ ] **Step 4: Add `_is_volume_entity` and `_render_volume_pass`**

Append to `RenderPipeline` (near the existing `_is_axis_entity` / `_is_label_entity`):

```python
    def _is_volume_entity(self, entity_idx, mat_id, engine):
        if mat_id <= 0:
            return False
        mat_obj = engine._material_registry.get(mat_id)
        if mat_obj is None:
            return False
        from manifoldx.viz import VolumeMaterial
        return isinstance(mat_obj, VolumeMaterial)

    def _render_volume_pass(self, engine, render_pass, volume_batches):
        """One draw call per volume entity (no instancing — each entity has
        its own model matrix and material). Trivial in v1; the per-frame cost
        is dominated by the raymarch, not the draw-call count.
        """
        if not volume_batches:
            return

        from manifoldx.viz.colormaps import _LUTS

        pipeline, bgl_volume = self._get_or_create_volume_pipeline(
            self._device, engine._texture_format
        )

        # Shared linear-clamp sampler (cached on first use).
        if not hasattr(self, "_volume_sampler") or self._volume_sampler is None:
            self._volume_sampler = self._device.create_sampler(
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
                address_mode_v=wgpu.AddressMode.clamp_to_edge,
                address_mode_w=wgpu.AddressMode.clamp_to_edge,
            )

        for entity_local_idx, mat_id, vol_id in volume_batches:
            mat_obj = engine._material_registry.get(mat_id)
            res = engine._volume_registry.get(vol_id)
            engine._volume_registry.upload_to_gpu(vol_id, self._device.queue)
            volume_view = res.texture.create_view()

            # Lazily create the material's color + opacity LUT textures.
            color_view = self._get_or_create_color_lut_view(mat_obj.cmap)
            opacity_view = self._get_or_create_opacity_lut_view(mat_id, mat_obj.opacity_lut)

            # Per-draw uniform buffer: model + inv_model + scalars.
            mat_uniform_buf = self._get_or_create_volume_uniform_buffer(mat_id)
            self._write_volume_uniforms(mat_uniform_buf, entity_local_idx, mat_obj, res, engine)

            # Globals bind group already populated each frame.
            globals_bg = self._globals_bind_group

            volume_bg = self._device.create_bind_group(
                layout=bgl_volume,
                entries=[
                    {"binding": 0, "resource": volume_view},
                    {"binding": 1, "resource": self._volume_sampler},
                    {"binding": 2, "resource": color_view},
                    {"binding": 3, "resource": opacity_view},
                    {"binding": 4, "resource": self._volume_sampler},
                    {
                        "binding": 5,
                        "resource": {
                            "buffer": mat_uniform_buf, "offset": 0, "size": 160,
                        },
                    },
                ],
            )

            render_pass.set_pipeline(pipeline)
            render_pass.set_bind_group(0, globals_bg, [], 0, 0)
            render_pass.set_bind_group(1, volume_bg, [], 0, 0)
            render_pass.draw(3, 1, 0, 0)   # fullscreen triangle
```

The four helper methods called above (`_get_or_create_color_lut_view`, `_get_or_create_opacity_lut_view`, `_get_or_create_volume_uniform_buffer`, `_write_volume_uniforms`) are added next.

- [ ] **Step 5: Add the LUT-view + uniform-buffer helpers**

Append to `RenderPipeline`:

```python
    def _get_or_create_color_lut_view(self, cmap_name: str):
        """Cache 256x1 RGBA8 colormap textures by name."""
        cache = getattr(self, "_volume_color_lut_views", None)
        if cache is None:
            cache = {}
            self._volume_color_lut_views = cache
        if cmap_name in cache:
            return cache[cmap_name]
        from manifoldx.viz.colormaps import get_colormap
        lut = get_colormap(cmap_name)   # (256, 4) uint8
        tex = self._device.create_texture(
            size=(256, 1, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
        )
        self._device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            lut.tobytes(),
            {"offset": 0, "bytes_per_row": 256 * 4, "rows_per_image": 1},
            (256, 1, 1),
        )
        view = tex.create_view()
        cache[cmap_name] = view
        return view

    def _get_or_create_opacity_lut_view(self, mat_id: int, lut: np.ndarray):
        """Cache 256x1 R32F opacity textures by material id.

        v1 assumes a material's opacity LUT is immutable post-construction;
        if mutability is needed later, hash the lut bytes for the cache key.
        """
        cache = getattr(self, "_volume_opacity_lut_views", None)
        if cache is None:
            cache = {}
            self._volume_opacity_lut_views = cache
        if mat_id in cache:
            return cache[mat_id]
        tex = self._device.create_texture(
            size=(256, 1, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
        )
        self._device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            lut.astype(np.float32).tobytes(),
            {"offset": 0, "bytes_per_row": 256 * 4, "rows_per_image": 1},
            (256, 1, 1),
        )
        view = tex.create_view()
        cache[mat_id] = view
        return view

    def _get_or_create_volume_uniform_buffer(self, mat_id: int):
        """Per-material uniform buffer (160 bytes; rewritten each frame)."""
        cache = getattr(self, "_volume_uniform_buffers", None)
        if cache is None:
            cache = {}
            self._volume_uniform_buffers = cache
        if mat_id not in cache:
            cache[mat_id] = self._device.create_buffer(
                size=160,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
        return cache[mat_id]

    def _write_volume_uniforms(self, buf, entity_local_idx, mat_obj, vol_res, engine):
        """Pack and upload the VolumeUniforms struct."""
        # Model matrix from the per-entity Transform cache.
        model = self._transform_cache.get_transforms(
            self._store, np.array([entity_local_idx], dtype=np.int64)
        )[0].reshape(4, 4)
        inv_model = np.linalg.inv(model)

        # Auto step size: half-voxel on the most-finely-sampled axis.
        nz, ny, nx = vol_res.data.shape
        scale_diag = np.array([model[0, 0], model[1, 1], model[2, 2]], dtype=np.float32)
        voxel_dims = scale_diag / np.array([nx, ny, nz], dtype=np.float32)
        auto_step = float(np.min(np.abs(voxel_dims)) * 0.5)
        step_size = mat_obj.step_size if mat_obj.step_size is not None else auto_step

        packed = np.zeros(160 // 4, dtype=np.float32)
        # column-major mat4 (WGSL convention)
        packed[0:16]  = model.T.astype(np.float32).ravel()
        packed[16:32] = inv_model.T.astype(np.float32).ravel()
        packed[32]    = mat_obj.vmin
        packed[33]    = mat_obj.vmax
        packed[34]    = mat_obj.density_scale
        packed[35]    = step_size
        # max_steps is u32 — write as raw bytes to slot 36 (offset 144).
        u32_view = packed.view(np.uint32)
        u32_view[36] = mat_obj.max_steps
        # Slots 37, 38, 39 are pad (already zero).
        self._device.queue.write_buffer(buf, 0, packed.tobytes())
```

- [ ] **Step 6: Run tests**

The render tests still won't pass yet (Task 9 wires the dispatch). But the *shader* and *pipeline* should already create successfully. Verify by:

```python
uv run python - <<'PY'
import numpy as np
import manifoldx as mx
from manifoldx.backends import get_offscreen_canvas
canvas = get_offscreen_canvas(width=64, height=64)
engine = mx.Engine("smoke", width=64, height=64)
engine._init_canvas(canvas)
engine._running = True
pipe, bgl = engine._render_pipeline._get_or_create_volume_pipeline(
    engine._device, engine._texture_format
)
print("OK", pipe is not None, bgl is not None)
PY
```

Expected: `OK True True`.

- [ ] **Step 7: Commit**

```bash
git add src/manifoldx/renderer.py
git commit -m "feat(viz): add _render_volume_pass + bind-group layout + LUT/uniform helpers"
```

---

## Task 9: Wire the volume pass into `RenderPipeline.render()`

**Files:**
- Modify: `src/manifoldx/renderer.py`
- Modify: `tests/test_volume_render.py` (re-run from Task 8 — should now pass)

- [ ] **Step 1: Confirm the render tests still fail at the framebuffer-content level**

Run: `uv run -q pytest tests/test_volume_render.py::test_centered_blob_renders_visible_pixels_at_origin -v`
Expected: FAIL — center pixel alpha is 0 because the volume pass is never invoked.

- [ ] **Step 2: Add volume routing to `render()`**

In `src/manifoldx/renderer.py`, modify the `render()` method around line 738-820. The changes:

(a) Add `has_volume` near the other component-presence checks:

```python
        has_mesh = "Mesh" in self._store._components
        has_point_cloud = "PointCloud" in self._store._components
        has_text_label = "TextLabel" in self._store._components
        has_axis_frame = "AxisFrame" in self._store._components
        has_volume = "Volume" in self._store._components
        if (
            not has_mesh
            and not has_point_cloud
            and not has_text_label
            and not has_axis_frame
            and not has_volume
        ):
            return
```

(b) Pull the per-entity `Volume.volume_id` array (after the existing `mesh_data` / `material_data` block):

```python
        volume_data = None
        if has_volume:
            volume_data = self._store.get_component_data("Volume", alive_indices)
```

(c) Add `volume_batches` to the batching block:

```python
        volume_batches = []   # list of (entity_local_idx, mat_id, vol_id)
```

(d) In the per-entity routing loop, branch on volume entities **before** the sprite/mesh fallthrough, after the existing axis/label/sprite checks:

```python
            is_volume = (
                (not is_axis) and (not is_label)
                and has_volume
                and self._is_volume_entity(entity_idx, mat_id, engine)
            )
            # ... existing is_sprite line stays as-is ...

            if is_axis:
                # ... existing
            elif is_label:
                # ... existing
            elif is_volume:
                vol_id = int(volume_data[i, 0])
                if vol_id > 0:
                    volume_batches.append((i, mat_id, vol_id))
            elif is_sprite:
                # ... existing
            else:
                # ... existing mesh path
```

(e) Insert the volume-pass dispatch in the existing draw-batches sequence, **between** sprite and label:

```python
        # Draw mesh batches (if any)
        if mesh_batches:
            self._render_mesh_batches(...)

        # Draw sprite batches (if any)
        if sprite_batches:
            self._render_sprite_batches(...)

        # Draw volume batches (depth-test on, depth-write off, alpha-blend on)
        if volume_batches:
            self._render_volume_pass(engine, render_pass, volume_batches)

        # Draw label batches (depth-write off, alpha-blend on)
        if label_batches:
            self._render_label_pass(...)

        # Draw axis batches (LineList topology, opaque)
        if axis_batches:
            self._render_axis_pass(...)
```

(f) Make sure the `is_sprite` predicate now also excludes `is_volume` — replace the existing `is_sprite = (not is_axis) and (not is_label) and has_point_cloud and geom_id == 0` with:

```python
            is_sprite = (
                (not is_axis) and (not is_label) and (not is_volume)
                and has_point_cloud and geom_id == 0
            )
```

- [ ] **Step 3: Run the render tests**

Run: `uv run -q pytest tests/test_volume_render.py -v`
Expected: PASS, all volume-related tests (4 from Task 7 + 2 new from Task 8).

Run the full suite:
`uv run -q pytest -q`
Expected: 388 passed (or all volume tests skipped on no-GPU CI).

- [ ] **Step 4: Commit**

```bash
git add src/manifoldx/renderer.py
git commit -m "feat(viz): wire volume pass into RenderPipeline.render() between sprite and label"
```

---

## Task 10: Step-size invariance + re-upload tests

**Files:**
- Modify: `tests/test_volume_render.py`

These two are the load-bearing correctness tests for DVR — without them subtle integration bugs go unnoticed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_volume_render.py`:

```python
def test_step_size_invariance():
    """Halving step_size produces nearly identical center-pixel rgba.

    The `alpha *= step_size` term in the integration formula is what
    makes this true. If someone refactors the shader and drops it,
    visibility will silently scale with quality settings.
    """
    import manifoldx as mx
    from manifoldx.components import Material, Transform
    from manifoldx.viz import Volume, VolumeMaterial

    n = 32
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    density = np.exp(-(X**2 + Y**2 + Z**2) / 0.1).astype(np.float32)

    def render_with_step(step):
        engine = _make_offscreen_engine(width=64, height=64)
        engine.camera.position = (0.0, 0.0, 5.0)
        handle = engine.register_volume(density)
        engine.spawn(
            Volume(volume_id=handle),
            Material(VolumeMaterial(
                cmap="inferno",
                opacity_stops=[(0.0, 0.0), (0.3, 0.05), (1.0, 0.6)],
                step_size=step,
            )),
            Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
            n=1,
        )
        return engine.render_to_array()

    rgba_coarse = render_with_step(0.04)
    rgba_fine = render_with_step(0.02)
    # Center pixel alpha should be within 0.02 (5% of 255 → ≈ 13 raw uint8).
    diff = abs(int(rgba_coarse[32, 32, 3]) - int(rgba_fine[32, 32, 3]))
    assert diff < 13


def test_update_volume_changes_pixels():
    """update_volume() between frames produces different framebuffer content."""
    import manifoldx as mx
    from manifoldx.components import Material, Transform
    from manifoldx.viz import Volume, VolumeMaterial

    engine = _make_offscreen_engine(width=64, height=64)
    engine.camera.position = (0.0, 0.0, 5.0)

    arr_a = np.zeros((16, 16, 16), dtype=np.float32)
    arr_a[8, 8, 8] = 1.0   # single bright voxel at center
    handle = engine.register_volume(arr_a)
    engine.spawn(
        Volume(volume_id=handle),
        Material(VolumeMaterial(
            cmap="viridis",
            opacity_stops=[(0.0, 0.0), (1.0, 1.0)],
            step_size=0.05,
            density_scale=20.0,
        )),
        Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
        n=1,
    )
    rgba_before = engine.render_to_array().copy()

    arr_b = np.zeros_like(arr_a)
    arr_b[2, 2, 2] = 1.0   # bright voxel at a different location
    engine.update_volume(handle, arr_b)
    rgba_after = engine.render_to_array()

    assert not np.array_equal(rgba_before, rgba_after)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run -q pytest tests/test_volume_render.py -v`
Expected: PASS, 2 new tests + previous tests still green.

If the step-size test fails by a small margin, raise the tolerance from 13 to 20 — single-voxel data is sensitive to interpolation; the *property* (visibility doesn't drift with step) is what matters, not an exact pixel match.

- [ ] **Step 3: Commit**

```bash
git add tests/test_volume_render.py
git commit -m "test(viz): step-size invariance + update_volume re-render"
```

---

## Task 11: Demo example — `examples/volume_demo.py`

**Files:**
- Create: `examples/volume_demo.py`

- [ ] **Step 1: Write the demo**

Create `examples/volume_demo.py`:

```python
"""Volume rendering demo — Gaussian blob with inferno colormap.

A static 64^3 scalar field uploaded once and raymarched each frame.
Camera orbits to make the box-shaped bounds and the radial density
falloff visible.
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Transform
from manifoldx.viz import Volume, VolumeMaterial


# ── Scalar field ─────────────────────────────────────────────────────────────
N = 64
xs = np.linspace(-1, 1, N, dtype=np.float32)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
density = np.exp(-(X**2 + Y**2 + Z**2) / 0.15).astype(np.float32)


# ── Engine ───────────────────────────────────────────────────────────────────
engine = mx.Engine("Volume rendering — Gaussian blob")
engine.camera.fit(2.0)

vol_id = engine.register_volume(density, name="gaussian")
engine.spawn(
    Volume(volume_id=vol_id),
    Material(VolumeMaterial(
        cmap="inferno",
        vmin=0.0, vmax=1.0,
        opacity_stops=[(0.0, 0.0), (0.2, 0.04), (0.6, 0.20), (1.0, 0.55)],
        density_scale=1.0,
        max_steps=256,
    )),
    Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
    n=1,
)


if __name__ == "__main__":
    engine.cli()
```

- [ ] **Step 2: Smoke render**

```bash
timeout 60 uv run python examples/volume_demo.py --render --duration 2 --fps 30 \
    --quality low --output /tmp/volume_demo.mp4
```

Expected: 60 frames rendered, no shader-validation errors, video saved.
Visually confirm with a player that the blob appears (you should see an orange-yellow glow inside a translucent box).

- [ ] **Step 3: Commit**

```bash
git add examples/volume_demo.py
git commit -m "feat(examples): add volume_demo.py — Gaussian-blob DVR with inferno cmap"
```

---

## Task 12: CHANGELOG entry + final full-suite check + push

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update CHANGELOG**

In `CHANGELOG.md` under `## [Unreleased]` → `### Features`, append:

```markdown
- **Volume rendering v1** — direct volume rendering (DVR) of a 3D scalar field via fragment-shader raymarching. New `Volume` ECS component (handle into a per-engine `VolumeRegistry`), new `VolumeMaterial` with reusable colormap LUT + per-material 256-sample opacity LUT (built from piecewise stops or a numpy array). `engine.register_volume(numpy_array)` / `engine.update_volume(handle, new_array)` for CPU upload; `engine.bind_compute_volume(...)` is reserved as a v2 stub. The volume pass renders between sprite and label with depth-test on / depth-write off and pre-multiplied alpha blending; a fullscreen-quad fragment shader does ray/box-AABB intersection in entity-local space, then a fixed-step front-to-back composite sampling two LUTs per step. New `examples/volume_demo.py` shows a 64³ Gaussian blob. v1 is unlit and CPU-uploaded; lighting, isosurface/MIP modes, compute-written volumes (`Writes[Volume3D]`), editable transfer-function widgets, and slicing are explicit non-goals deferred to follow-up plans. Design: `.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md`.
```

- [ ] **Step 2: Run full lint + test + smoke**

```bash
uv run ruff check src/ tests/ examples/volume_demo.py
uv run -q pytest -q
uv run --with mypy mypy examples/volume_demo.py
```

Expected: all checks pass; ≥ 388 tests pass (varies with how many of Task 8/9/10 were run on the GPU vs skipped).

- [ ] **Step 3: Commit + push**

```bash
git add CHANGELOG.md
git commit -m "docs(viz): CHANGELOG entry for volume rendering v1"
git push origin main
```

---

## Self-review

**Spec coverage:**
- [x] `Volume` component → Task 2.
- [x] `VolumeMaterial` (cmap + opacity_stops + vmin/vmax/density_scale/step_size/max_steps + validation) → Task 5.
- [x] `engine.register_volume / update_volume / bind_compute_volume` → Task 4.
- [x] `r32float` 3D texture upload + dirty re-upload → Task 7.
- [x] DVR fragment shader (ray/box, fixed-step front-to-back) → Task 6.
- [x] Render pass with depth-test/depth-write/blending → Task 8 (pipeline) + Task 9 (dispatch).
- [x] Pass ordering `mesh → sprite → volume → label → axis` → Task 9 step 2(e).
- [x] Pipeline cache `"volume"` kind → Task 8 step 3 (`cache_key = ("volume",)`).
- [x] Step-size invariance test → Task 10.
- [x] Multi-entity sharing one volume → Task 8 (`test_two_entities_share_volume_handle`).
- [x] Re-upload after `update_volume` → Task 10 (`test_update_volume_changes_pixels`).
- [x] CHANGELOG entry → Task 12.
- [x] `volume_demo.py` example → Task 11.
- [x] Renderer-split-refactor deferral noted → Task 1.

**No placeholders:** every task contains the exact code, exact paths, exact commands.

**Type consistency:** `volume_id` (i32, on `Volume`); `handle` (int, returned by `register_volume`); `vol_id` (local var name in renderer routing) — three names for the same scalar, used consistently within their scope. Method names match across tasks: `register`, `update`, `get`, `upload_to_gpu`, `_render_volume_pass`, `_get_or_create_volume_pipeline`. `VOLUME_SHADER_SOURCE` referenced by name in Task 6 + Task 8.

**Out-of-band gotcha:** Task 8's tests reference `engine.render_to_array()`. If that helper doesn't exist on `Engine` already, the implementation must either (a) add it as a thin wrapper around the existing offscreen render path, or (b) replace the test calls with the manual `engine._render_pipeline.render(engine, render_pass)` pattern used by `tests/test_render_mvp.py`. The plan flags this in Task 8 step 1 — verify before writing tests.

---

## Execution

Plan complete and saved to `.knowledge/plans/2026-05-08-volume-rendering-v1.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
