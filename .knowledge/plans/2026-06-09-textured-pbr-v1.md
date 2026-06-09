# Textured PBR v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the classic Utah teapot with a textured albedo map under the existing PBR shader. Land the texture-sampling plumbing as the foundation for follow-up plans (normal / metallic-roughness / AO maps).

**Architecture:** Three vertical slices. (1) Make `sphere()` UV-capable and add the full texture pipeline — `TextureRegistry`, `StandardMaterial.albedo_map`, UV-aware geometry interleave, pipeline-cache subtype keying, bind-group texture entries. Locked down end-to-end on a built-in sphere. (2) OBJ loader as a standalone, GPU-free unit. (3) Teapot demo with bundled asset + closeout.

**Tech Stack:** Python 3.13, `wgpu` (existing), `pillow` (promoted from optional to required), `numpy`, `pytest`. No new third-party deps.

**Spec:** `.knowledge/analysis/2026-06-09-textured-pbr-v1-design.md`.

---

## Slice 1 — Texture pipeline end-to-end on built-in sphere

### Task 1: `sphere()` and `plane()` emit UVs

**Files:**
- Modify: `src/manifoldx/resources.py:613-680` (`sphere`, `plane`)
- Test: `tests/test_geometry_uvs.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_uvs.py
import numpy as np
from manifoldx.resources import sphere, plane


def test_sphere_emits_uvs():
    geo = sphere(1.0, segments=8)
    assert "uvs" in geo
    uvs = geo["uvs"]
    assert uvs.dtype == np.float32
    assert uvs.shape == (geo["positions"].shape[0], 2)
    assert uvs.min() >= 0.0
    assert uvs.max() <= 1.0


def test_plane_emits_uvs():
    geo = plane(2.0, 2.0)
    assert "uvs" in geo
    uvs = geo["uvs"]
    assert uvs.shape == (4, 2)
    # Corner UVs should be at (0,0), (1,0), (1,1), (0,1) in the vertex order
    # used by plane() — positions (-w,-h), (w,-h), (w,h), (-w,h).
    expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    np.testing.assert_allclose(uvs, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_geometry_uvs.py -v`
Expected: 2 FAIL with `assert "uvs" in geo`.

- [ ] **Step 3: Implement UV emission**

In `src/manifoldx/resources.py`, modify `sphere()` to track UVs in the lat/lon loop:

```python
def sphere(radius: float, segments: int = 32) -> dict:
    """
    Create UV sphere geometry with normals and UVs.

    Normals point outward (normalized position for unit sphere).
    UVs use standard (u = lon/lon_lines, v = lat/lat_lines) spherical mapping.
    Winding order is CCW when viewed from outside.
    """
    lat_lines = segments
    lon_lines = segments * 2

    positions = []
    normals = []
    uvs = []
    for lat in range(lat_lines + 1):
        theta = lat * np.pi / lat_lines
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        v = lat / lat_lines

        for lon in range(lon_lines + 1):
            phi = lon * 2 * np.pi / lon_lines
            nx = sin_theta * np.cos(phi)
            ny = cos_theta
            nz = sin_theta * np.sin(phi)
            normals.append([nx, ny, nz])
            positions.append([nx * radius, ny * radius, nz * radius])
            uvs.append([lon / lon_lines, v])

    positions = np.array(positions, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    indices = []
    for lat in range(lat_lines):
        for lon in range(lon_lines):
            first = lat * (lon_lines + 1) + lon
            second = first + lon_lines + 1
            indices.extend([first, first + 1, second])
            indices.extend([second, first + 1, second + 1])

    indices = np.array(indices, dtype=np.uint32)

    return {"positions": positions, "normals": normals, "uvs": uvs, "indices": indices}
```

And modify `plane()`:

```python
def plane(width: float, height: float) -> dict:
    """Create plane geometry with normals facing +Z and UVs in [0,1]²."""
    w, h = width / 2, height / 2

    positions = np.array(
        [[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]],
        dtype=np.float32,
    )
    normals = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        dtype=np.float32,
    )
    uvs = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    return {"positions": positions, "normals": normals, "uvs": uvs, "indices": indices}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_geometry_uvs.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Run full test suite — nothing regressed**

Run: `uv run pytest -q`
Expected: All previous tests still pass. (`cube()` keeps no UVs in this task — it's edited in slice 3.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_geometry_uvs.py src/manifoldx/resources.py
git commit -m "feat(geometry): sphere() and plane() emit UVs"
```

---

### Task 2: `GeometryRegistry.create_buffers` interleaves UVs when present

**Files:**
- Modify: `src/manifoldx/resources.py:345-403` (`GeometryRegistry.create_buffers`)
- Test: `tests/test_geometry_uvs.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_uvs.py — append
def test_create_buffers_records_stride_when_uvs_present():
    """Geometry with UVs gets stride=32 (pos+normal+uv); without UVs, stride=24."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    from manifoldx.resources import GeometryRegistry, sphere

    reg = GeometryRegistry(device=device)
    geo = sphere(1.0, segments=4)
    geom_id = reg.register(geo)
    buffers = reg.create_buffers(geom_id, geo, device.queue)
    assert buffers["stride"] == 32
    assert buffers["has_uvs"] is True
    assert buffers["has_normals"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_geometry_uvs.py::test_create_buffers_records_stride_when_uvs_present -v`
Expected: FAIL on `assert buffers["stride"] == 32` (currently 24).

- [ ] **Step 3: Extend `create_buffers` to interleave UVs**

In `src/manifoldx/resources.py`, replace the body of `create_buffers` that builds the interleaved buffer (lines ~363-388):

```python
        if positions is not None:
            has_normals = "normals" in geometry_obj
            has_uvs = "uvs" in geometry_obj

            if has_normals and has_uvs:
                normals = geometry_obj["normals"].astype(np.float32)
                uvs = geometry_obj["uvs"].astype(np.float32)
                n_verts = len(positions)
                interleaved = np.zeros((n_verts, 8), dtype=np.float32)
                interleaved[:, 0:3] = positions
                interleaved[:, 3:6] = normals
                interleaved[:, 6:8] = uvs
                data = interleaved.tobytes()
                buffers["stride"] = 8 * 4  # pos(3) + normal(3) + uv(2)
                buffers["has_normals"] = True
                buffers["has_uvs"] = True
            elif has_normals:
                normals = geometry_obj["normals"].astype(np.float32)
                n_verts = len(positions)
                interleaved = np.zeros((n_verts, 6), dtype=np.float32)
                interleaved[:, 0:3] = positions
                interleaved[:, 3:6] = normals
                data = interleaved.tobytes()
                buffers["stride"] = 6 * 4
                buffers["has_normals"] = True
                buffers["has_uvs"] = False
            else:
                data = positions.tobytes()
                buffers["stride"] = 3 * 4
                buffers["has_normals"] = False
                buffers["has_uvs"] = False

            vertex_buffer = self._device.create_buffer(
                size=len(data),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )
            queue.write_buffer(vertex_buffer, 0, data)
            buffers["vertex_buffer"] = vertex_buffer
            buffers["vertex_count"] = len(positions)
```

- [ ] **Step 4: Run the new test and the full suite**

Run: `uv run pytest tests/test_geometry_uvs.py -v && uv run pytest -q`
Expected: new test PASS, full suite PASS (existing geometries without UVs still produce stride-24 buffers).

- [ ] **Step 5: Commit**

```bash
git add tests/test_geometry_uvs.py src/manifoldx/resources.py
git commit -m "feat(geometry): UV-aware interleave in GeometryRegistry.create_buffers"
```

---

### Task 3: `TextureHandle` dataclass + `TextureRegistry`

**Files:**
- Create: `src/manifoldx/textures.py`
- Test: `tests/test_textures.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_textures.py
import io
import numpy as np
import pytest
from PIL import Image


def _make_tiny_png(w=2, h=2, rgba=(255, 0, 0, 255)):
    img = Image.new("RGBA", (w, h), rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_texture_handle_fields():
    from manifoldx.textures import TextureHandle
    h = TextureHandle(id=1, texture=None, view=None, sampler=None, size=(2, 2))
    assert h.id == 1
    assert h.size == (2, 2)


def test_load_texture_requires_engine_device(tmp_path):
    """load_texture(engine, path) requires engine._device to be initialized."""
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.textures import load_texture

    engine = mx.Engine("test")
    # Force device init by attaching a canvas — the engine ctor doesn't allocate it lazily.
    engine.attach_canvas(canvas)

    png_path = tmp_path / "red.png"
    png_path.write_bytes(_make_tiny_png(rgba=(255, 0, 0, 255)))

    handle = load_texture(engine, png_path)
    assert handle.size == (2, 2)
    assert handle.texture is not None
    assert handle.view is not None
    assert handle.sampler is not None
```

> **Note for the executor:** before running the test, check what the actual engine attach-canvas API is called — `engine.attach_canvas`, `engine.bind_canvas`, or another name. Grep `src/manifoldx/engine.py` for the public method that drives device init from a canvas. Adjust the test call to match (the API itself is not what we're testing here). If the engine does device init in its constructor, drop the `attach_canvas` line entirely.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_textures.py -v`
Expected: FAIL with `ModuleNotFoundError: manifoldx.textures`.

- [ ] **Step 3: Implement `textures.py`**

```python
# src/manifoldx/textures.py
"""Texture loading and GPU upload for the mesh-PBR path.

A TextureHandle is the public reference users hand to materials
(e.g. StandardMaterial(albedo_map=handle)). The engine's
TextureRegistry owns the underlying GPU resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import wgpu


class TextureSizeError(ValueError):
    """Image is larger than the device's max_texture_dimension_2d limit."""


@dataclass
class TextureHandle:
    id: int
    texture: Any            # wgpu.Texture
    view: Any               # wgpu.TextureView
    sampler: Any            # wgpu.Sampler
    size: tuple[int, int]   # (width, height) in pixels


class TextureRegistry:
    """Owns GPU texture + sampler resources for the engine lifetime."""

    def __init__(self) -> None:
        self._handles: Dict[int, TextureHandle] = {}
        self._next_id = 1

    def add(self, handle: TextureHandle) -> None:
        self._handles[handle.id] = handle

    def alloc_id(self) -> int:
        new_id = self._next_id
        self._next_id += 1
        return new_id


def load_texture(engine, path: str | Path) -> TextureHandle:
    """Decode an image file with Pillow, upload to GPU as Rgba8UnormSrgb.

    Args:
        engine: the manifoldx Engine (must have a device initialized).
        path: filesystem path to a PNG / JPEG / any Pillow-supported format.

    Returns:
        A TextureHandle the caller passes to material constructors.

    Raises:
        FileNotFoundError: path doesn't exist.
        TextureSizeError: image exceeds device.limits.max_texture_dimension_2d.
        RuntimeError: engine has no device (canvas not attached yet).
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    img = Image.open(p).convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)   # (H, W, 4)
    h, w = arr.shape[:2]

    device = getattr(engine, "_device", None)
    if device is None:
        raise RuntimeError(
            "engine has no device yet; attach a canvas before load_texture(...)"
        )

    max_dim = device.limits.get("max-texture-dimension-2d", 8192)
    if w > max_dim or h > max_dim:
        raise TextureSizeError(
            f"image is {w}x{h}, device max_texture_dimension_2d is {max_dim}"
        )

    texture = device.create_texture(
        size=(w, h, 1),
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        mip_level_count=1,
        sample_count=1,
    )
    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        arr.tobytes(),
        {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
        (w, h, 1),
    )
    view = texture.create_view()
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
        address_mode_u=wgpu.AddressMode.repeat,
        address_mode_v=wgpu.AddressMode.repeat,
    )

    registry = engine._texture_registry
    handle = TextureHandle(
        id=registry.alloc_id(), texture=texture, view=view, sampler=sampler,
        size=(w, h),
    )
    registry.add(handle)
    return handle
```

- [ ] **Step 4: Wire `TextureRegistry` into the engine**

In `src/manifoldx/engine.py`, find the `Engine.__init__` (grep for `class Engine` then the `def __init__`). Add one line that initializes the registry alongside the other registries:

```python
from manifoldx.textures import TextureRegistry  # at module top with other imports
...
# In __init__, after the other registry inits (volume registry, material registry, etc.):
self._texture_registry = TextureRegistry()
```

> **Note for the executor:** grep `src/manifoldx/engine.py` for `_volume_registry` or `_material_registry` to find the correct line to add this near. Match the surrounding style.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_textures.py -v`
Expected: PASS on both tests. (`test_load_texture_requires_engine_device` may skip on machines without an offscreen canvas backend — that's fine.)

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: All tests pass; nothing regressed.

- [ ] **Step 7: Commit**

```bash
git add src/manifoldx/textures.py src/manifoldx/engine.py tests/test_textures.py
git commit -m "feat(textures): TextureHandle + TextureRegistry + load_texture"
```

---

### Task 4: Mesh pipeline cache keys on `material_subtype`

The current mesh cache key (`src/manifoldx/renderer.py:324`) is `(geometry_id, material_type)` — it does NOT include `pipeline_subtype`. Without this fix a scalar `StandardMaterial` and a textured `StandardMaterial` would collide in the cache.

**Files:**
- Modify: `src/manifoldx/renderer.py:302-324` (`_get_or_create_pipeline`)
- Modify: `src/manifoldx/render/passes/mesh.py` — any place that recomputes the same key
- Test: `tests/test_pipeline_cache_keys.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_cache_keys.py
import pytest


def test_mesh_cache_key_includes_subtype():
    """Scalar StandardMaterial and textured StandardMaterial must produce
    different pipeline-cache keys for the same geometry."""
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    engine = mx.Engine("test")
    engine.attach_canvas(canvas)

    rp = engine._renderer
    rp._initialize_gpu_resources(engine._device, engine._texture_format)

    # Build two materials: one scalar, one will be textured later in this slice.
    # For now we simulate "different subtype" by monkey-patching the second
    # material's pipeline_subtype attribute, since the StandardMaterial code
    # change for albedo_map lands in a later task.
    from manifoldx.resources import StandardMaterial, sphere

    geo = sphere(1.0, segments=4)
    geom_id = engine._geometry_registry.register(geo)

    mat_scalar = StandardMaterial(color=(1, 0, 0))
    mat_textured = StandardMaterial(color=(1, 0, 0))
    mat_textured.pipeline_subtype = "textured"

    rp._get_or_create_pipeline(
        engine._device, engine._texture_format, geom_id, mat_scalar,
        engine._material_registry,
    )
    rp._get_or_create_pipeline(
        engine._device, engine._texture_format, geom_id, mat_textured,
        engine._material_registry,
    )

    keys = list(rp._pipelines.keys())
    # The two materials should produce two distinct cache entries.
    assert len(keys) >= 2
    scalar_keys = [k for k in keys if "textured" not in str(k)]
    textured_keys = [k for k in keys if "textured" in str(k)]
    assert len(scalar_keys) >= 1
    assert len(textured_keys) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_cache_keys.py -v`
Expected: FAIL with `assert len(textured_keys) >= 1` — both materials collide in one key under the current code.

- [ ] **Step 3: Include `material_subtype` in the mesh cache key**

In `src/manifoldx/renderer.py`, modify `_get_or_create_pipeline` at line ~323:

```python
        if line:
            key = (geometry_id, material_type, material_subtype, "line")
        elif label:
            key = (geometry_id, material_type, material_subtype, "label")
        elif sprite:
            key = (geometry_id, material_type, material_subtype, True)
        else:
            key = (geometry_id, material_type, material_subtype)
```

Then in `src/manifoldx/render/passes/mesh.py`, audit any place that constructs `(geom_id, mat_type)` to look up pipelines / bind groups / material buffers. Replace those tuples with `(geom_id, mat_type, material_subtype)` where they're used as cache keys. Specifically the lines around `bkey = (geom_id, mat_type)` need to become `bkey = (geom_id, mat_type, getattr(mat_obj, "pipeline_subtype", None))`.

> **Note for the executor:** grep `src/manifoldx/render/passes/mesh.py` for `(geom_id, mat_type)` and `bkey` to find every callsite. They MUST all agree.

- [ ] **Step 4: Run the new test and the full suite**

Run: `uv run pytest tests/test_pipeline_cache_keys.py -v && uv run pytest -q`
Expected: new test PASS. Existing PBR demo path (scalar `StandardMaterial`, `BasicMaterial`, etc.) all produce subtype=None and still cache-hit correctly. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/renderer.py src/manifoldx/render/passes/mesh.py tests/test_pipeline_cache_keys.py
git commit -m "fix(renderer): mesh pipeline cache key includes material_subtype"
```

---

### Task 5: `StandardMaterial.albedo_map` kwarg + textured WGSL variant

**Files:**
- Modify: `src/manifoldx/resources.py` — `StandardMaterial`, `Material` base
- Test: `tests/test_standard_material_textured.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_standard_material_textured.py
import pytest
from manifoldx.resources import StandardMaterial
from manifoldx.textures import TextureHandle


def _fake_handle():
    return TextureHandle(id=1, texture=object(), view=object(),
                         sampler=object(), size=(2, 2))


def test_scalar_subtype_is_none():
    m = StandardMaterial(color=(1, 0, 0))
    assert m.pipeline_subtype is None
    assert "@binding(4)" not in m._compile()


def test_textured_subtype_and_shader():
    m = StandardMaterial(color=(1, 0, 0), albedo_map=_fake_handle())
    assert m.pipeline_subtype == "textured"
    src = m._compile()
    assert "@binding(4)" in src
    assert "@binding(5)" in src
    assert "textureSample" in src
    assert "@location(2) uv" in src


def test_textured_get_texture_bindings():
    handle = _fake_handle()
    m = StandardMaterial(color=(1, 0, 0), albedo_map=handle)
    bindings = m.get_texture_bindings()
    assert bindings == {4: handle}


def test_scalar_get_texture_bindings_empty():
    m = StandardMaterial(color=(1, 0, 0))
    assert m.get_texture_bindings() == {}


def test_albedo_map_type_error():
    with pytest.raises(TypeError, match="TextureHandle"):
        StandardMaterial(color=(1, 0, 0), albedo_map="not-a-handle")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_standard_material_textured.py -v`
Expected: 5 FAIL — `albedo_map` kwarg doesn't exist yet.

- [ ] **Step 3: Add `get_texture_bindings` to the `Material` base class**

In `src/manifoldx/resources.py`, in the `Material(ABC)` class (around line 17), add:

```python
    def get_texture_bindings(self) -> Dict[int, "TextureHandle"]:
        """Return {binding_index: TextureHandle} for sampler+texture entries.

        Default: no textures. Override on materials that bind 2D textures.
        Sampler is attached at `binding_index`; texture view at `binding_index + 1`.
        """
        return {}
```

- [ ] **Step 4: Extend `StandardMaterial`**

In `src/manifoldx/resources.py`, modify the `StandardMaterial` class. Change `__init__`:

```python
class StandardMaterial(Material):
    """PBR material with GGX BRDF. Optional 2D albedo map."""

    binding_slot = 1

    def __init__(self, color, roughness=0.5, metallic=0.0, ao=1.0,
                 albedo_map=None):
        from manifoldx.textures import TextureHandle

        if albedo_map is not None and not isinstance(albedo_map, TextureHandle):
            raise TypeError(
                f"albedo_map expects a TextureHandle from load_texture(...); "
                f"got {type(albedo_map).__name__}. "
                f"Did you forget to call load_texture(engine, path) first?"
            )
        self.color = color
        self.roughness = roughness
        self.metallic = metallic
        self.ao = ao
        self.albedo_map = albedo_map

    @property
    def pipeline_subtype(self):
        return "textured" if self.albedo_map is not None else None

    def get_texture_bindings(self):
        if self.albedo_map is None:
            return {}
        return {4: self.albedo_map}
```

- [ ] **Step 5: Emit a textured WGSL variant from `_compile()`**

Replace the existing `StandardMaterial._compile()` with:

```python
    @classmethod
    def _compile(cls, textured: bool = False) -> str:
        # _STANDARDMATERIAL_SHADER is the scalar variant (unchanged).
        if not textured:
            return _STANDARDMATERIAL_SHADER
        return _STANDARDMATERIAL_TEXTURED_SHADER
```

And add the textured shader source. It's a copy of `_STANDARDMATERIAL_SHADER` with five edits:

```python
_STANDARDMATERIAL_TEXTURED_SHADER = _STANDARDMATERIAL_SHADER \
    .replace(
        # 1. Add the sampler + texture bindings after the lights binding.
        "@group(0) @binding(3) var<uniform> light_data: LightData;",
        "@group(0) @binding(3) var<uniform> light_data: LightData;\n"
        "@group(0) @binding(4) var albedo_sampler: sampler;\n"
        "@group(0) @binding(5) var albedo_tex: texture_2d<f32>;",
    ) \
    .replace(
        # 2. Add UV to vertex input.
        "struct VertexInput {\n"
        "    @location(0) position: vec3<f32>,\n"
        "    @location(1) normal: vec3<f32>,\n"
        "    @builtin(instance_index) instance: u32,\n"
        "};",
        "struct VertexInput {\n"
        "    @location(0) position: vec3<f32>,\n"
        "    @location(1) normal: vec3<f32>,\n"
        "    @location(2) uv: vec2<f32>,\n"
        "    @builtin(instance_index) instance: u32,\n"
        "};",
    ) \
    .replace(
        # 3. Carry UV through to fragment.
        "struct VertexOutput {\n"
        "    @builtin(position) position: vec4<f32>,\n"
        "    @location(0) world_normal: vec3<f32>,\n"
        "    @location(1) world_pos: vec3<f32>,\n"
        "};",
        "struct VertexOutput {\n"
        "    @builtin(position) position: vec4<f32>,\n"
        "    @location(0) world_normal: vec3<f32>,\n"
        "    @location(1) world_pos: vec3<f32>,\n"
        "    @location(2) uv: vec2<f32>,\n"
        "};",
    ) \
    .replace(
        # 4. Pass UV through in vs_main.
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    return out;",
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    out.uv = in.uv;\n"
        "    return out;",
    ) \
    .replace(
        # 5. Sample albedo from the texture in fs_main, before F0 mix.
        "    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);",
        "    let sampled_albedo = textureSample(albedo_tex, albedo_sampler, in.uv).rgb;\n"
        "    let F0 = mix(vec3<f32>(0.04), sampled_albedo, material.metallic);",
    ) \
    .replace(
        # 6. Use sampled albedo in the lighting call and the ambient term.
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                       material.albedo, material.metallic,\n"
        "                                       material.roughness, light);",
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                       sampled_albedo, material.metallic,\n"
        "                                       material.roughness, light);",
    ) \
    .replace(
        "    let ambient = vec3<f32>(0.03) * material.albedo * material.ao;",
        "    let ambient = vec3<f32>(0.03) * sampled_albedo * material.ao;",
    )
```

Then update `StandardMaterial._compile` to look at the instance's `albedo_map` attribute via the renderer's call site — wait, `_compile` is a classmethod. We need it to be instance-aware so the renderer compiles the right variant. Change the renderer to call `material._compile(textured=material.albedo_map is not None)`.

Actually the cleanest: make `_compile` accept the instance via a wrapper. Revise:

```python
    @classmethod
    def _compile(cls, textured: bool = False) -> str:
        return _STANDARDMATERIAL_TEXTURED_SHADER if textured else _STANDARDMATERIAL_SHADER
```

The renderer is the only call site for `_compile` (verify via grep). When invoked it gets the material instance — change the call site to forward the right flag. Find the call sites:

```
grep -n "_compile()" src/manifoldx/renderer.py src/manifoldx/render/passes/mesh.py
```

For each call site that takes a `StandardMaterial` (or any material), change `material._compile()` to `material._compile(**_compile_kwargs(material))` where:

```python
# In renderer.py at the top of the module:
def _compile_kwargs(material):
    """Per-material kwargs to pass into _compile()."""
    if hasattr(material, "albedo_map") and material.albedo_map is not None:
        return {"textured": True}
    return {}
```

> **Note for the executor:** if the call sites pass plain `material._compile()` everywhere, the minimal touch is to update only those sites where the material is a `StandardMaterial` instance. Other materials' `_compile` classmethods don't accept the kwarg — pass via the helper above so unknown materials get `{}` and remain untouched.

- [ ] **Step 6: Run the unit tests**

Run: `uv run pytest tests/test_standard_material_textured.py -v`
Expected: 5 PASS.

- [ ] **Step 7: Run the full suite — scalar PBR demo path untouched**

Run: `uv run pytest -q`
Expected: All tests pass; nothing in the existing PBR / `pbr_demo.py` smoke-test paths regressed.

- [ ] **Step 8: Commit**

```bash
git add src/manifoldx/resources.py src/manifoldx/renderer.py tests/test_standard_material_textured.py
git commit -m "feat(materials): StandardMaterial.albedo_map + textured WGSL variant"
```

---

### Task 6: Textured-subtype vertex layout + bind-group layout in pipeline cache

The pipeline cache now keys on `material_subtype`, but a textured cache miss still builds the old 4-binding layout with stride-24 vertex buffer. Both layouts need a "textured" branch.

**Files:**
- Modify: `src/manifoldx/renderer.py` — `_get_or_create_pipeline` mesh path (lines ~620-730)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_cache_keys.py — append
def test_textured_pipeline_has_six_bindings_and_stride_32():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        import pytest; pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.resources import StandardMaterial, sphere
    from manifoldx.textures import load_texture
    import io
    from PIL import Image
    from pathlib import Path
    import tempfile

    engine = mx.Engine("test")
    engine.attach_canvas(canvas)
    rp = engine._renderer
    rp._initialize_gpu_resources(engine._device, engine._texture_format)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(f, format="PNG")
        png_path = Path(f.name)

    tex = load_texture(engine, png_path)

    geo = sphere(1.0, segments=4)
    geom_id = engine._geometry_registry.register(geo)
    engine._geometry_registry.create_buffers(geom_id, geo, engine._device.queue)

    mat = StandardMaterial(color=(1, 1, 1), albedo_map=tex)
    pipeline, bgl = rp._get_or_create_pipeline(
        engine._device, engine._texture_format, geom_id, mat,
        engine._material_registry,
    )
    assert pipeline is not None
    assert bgl is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_cache_keys.py::test_textured_pipeline_has_six_bindings_and_stride_32 -v`
Expected: FAIL — pipeline creation probably raises a wgpu validation error because the shader's `@binding(4)` / `@binding(5)` / `@location(2)` aren't in the layout.

- [ ] **Step 3: Add the textured branch to `_get_or_create_pipeline`**

In `src/manifoldx/renderer.py`, in the mesh branch (after the scalar `else`-branch around line ~640), detect `material_subtype == "textured"` and build the extended layout.

Find this block:

```python
            # StandardMaterial: 4 bindings (globals, transforms, material, lights)
            bind_group_entries = [
                ...4 entries...
            ]
```

And replace it with:

```python
            # StandardMaterial: 4 bindings; +2 more if textured.
            bind_group_entries = [
                {"binding": 0,
                 "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1,
                 "visibility": wgpu.ShaderStage.VERTEX,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2,
                 "visibility": wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 3,
                 "visibility": wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
            if material_subtype == "textured":
                bind_group_entries.extend([
                    {"binding": 4,
                     "visibility": wgpu.ShaderStage.FRAGMENT,
                     "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                    {"binding": 5,
                     "visibility": wgpu.ShaderStage.FRAGMENT,
                     "texture": {
                         "sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": wgpu.TextureViewDimension.d2,
                     }},
                ])
```

And in the same function, find the vertex-buffer layout block (`"array_stride": 6 * 4`) and replace it with:

```python
            if material_subtype == "textured":
                vertex_buffers = [{
                    "array_stride": 8 * 4,  # pos(3) + normal(3) + uv(2)
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {"format": wgpu.VertexFormat.float32x3,
                         "offset": 0, "shader_location": 0},
                        {"format": wgpu.VertexFormat.float32x3,
                         "offset": 3 * 4, "shader_location": 1},
                        {"format": wgpu.VertexFormat.float32x2,
                         "offset": 6 * 4, "shader_location": 2},
                    ],
                }]
            else:
                vertex_buffers = [{
                    "array_stride": 6 * 4,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {"format": wgpu.VertexFormat.float32x3,
                         "offset": 0, "shader_location": 0},
                        {"format": wgpu.VertexFormat.float32x3,
                         "offset": 3 * 4, "shader_location": 1},
                    ],
                }]
            ...
            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": vertex_buffers,
                },
                ...
            )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_pipeline_cache_keys.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/renderer.py tests/test_pipeline_cache_keys.py
git commit -m "feat(renderer): textured-subtype vertex layout + bind-group entries"
```

---

### Task 7: Mesh-pass bind-group builder appends texture entries

The pipeline now declares bindings 4 and 5, but the per-frame mesh pass doesn't yet attach a sampler / view to those bindings. The draw call would fail validation.

**Files:**
- Modify: `src/manifoldx/render/passes/mesh.py` — bind-group construction (lines ~75-100)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_textured_material.py
import io
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


def _make_solid_png(path, rgba):
    Image.new("RGBA", (4, 4), rgba).save(path, format="PNG")


def test_textured_sphere_renders():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    from manifoldx.resources import StandardMaterial, PointLight, sphere
    from manifoldx.textures import load_texture

    engine = mx.Engine("textured-smoke")
    engine.attach_canvas(canvas)

    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / "blue.png"
        _make_solid_png(png_path, (40, 70, 220, 255))   # vivid blue

        tex = load_texture(engine, png_path)
        sphere_geo = sphere(1.0, segments=16)
        mat = StandardMaterial(color=(1, 1, 1), roughness=0.4, metallic=0.0,
                               albedo_map=tex)

        engine.spawn(Transform(position=(0, 0, 0)),
                     Mesh(sphere_geo), Material(mat))

        # Place one bright light + camera (mirror the pbr_demo idiom).
        light = PointLight(position=(2, 3, 3), color=(1, 1, 1), intensity=25.0)
        engine.add_light(light)
        engine._camera.position = np.array([0, 0, 3], dtype=np.float32)
        engine._camera.target = np.array([0, 0, 0], dtype=np.float32)

        engine.render(duration=0.1, fps=30, output=None)
        # If render() throws, the test fails. Successful single-frame render
        # is the success criterion for the smoke test.
```

> **Note for the executor:** the exact API for adding a light and rendering a single frame may differ — grep `examples/pbr_demo.py` and `src/manifoldx/engine.py` for the actual names (`engine.add_light` vs `engine.lights.append`, `engine.render(duration=...)` vs `engine.render_video(...)`). Adjust the test to match. The test's only job is to assert "render does not raise" with a textured StandardMaterial in the scene.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_textured_material.py -v`
Expected: FAIL with a wgpu validation error — the bind group is missing entries 4 and 5.

- [ ] **Step 3: Append texture entries to the bind group in the mesh pass**

In `src/manifoldx/render/passes/mesh.py`, find the `bind_group_entries = [...]` list construction (around line 75-100). After the final existing entry (lights), insert:

```python
        # Append texture bindings declared by the material (sampler at N, view at N+1).
        texture_bindings = mat_obj.get_texture_bindings()
        for binding, handle in texture_bindings.items():
            bind_group_entries.append({
                "binding": binding,
                "resource": handle.sampler,
            })
            bind_group_entries.append({
                "binding": binding + 1,
                "resource": handle.view,
            })
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_textured_material.py -v`
Expected: PASS — textured sphere renders.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/render/passes/mesh.py tests/test_textured_material.py
git commit -m "feat(renderer): mesh pass attaches material texture bindings"
```

---

### Task 8: `MaterialGeometryMismatchError` for textured-mat-on-no-UV-geo

A textured `StandardMaterial` attached to a geometry without UVs would silently render garbage (the vertex layout would mismatch the buffer stride). Catch it loudly at pipeline-cache time.

**Files:**
- Modify: `src/manifoldx/renderer.py` — `_get_or_create_pipeline` mesh textured branch
- Test: `tests/test_textured_material.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_textured_material.py — append
def test_textured_material_on_no_uv_geometry_raises():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    from manifoldx.resources import StandardMaterial, cube
    from manifoldx.textures import load_texture
    from manifoldx.renderer import MaterialGeometryMismatchError

    engine = mx.Engine("mismatch-test")
    engine.attach_canvas(canvas)

    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / "w.png"
        _make_solid_png(png_path, (255, 255, 255, 255))
        tex = load_texture(engine, png_path)

        # cube() has no UVs in v1.
        engine.spawn(
            Transform(position=(0, 0, 0)),
            Mesh(cube(1, 1, 1)),
            Material(StandardMaterial(color=(1, 0, 0), albedo_map=tex)),
        )

        with pytest.raises(MaterialGeometryMismatchError, match="UVs"):
            engine.render(duration=0.05, fps=30, output=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_textured_material.py::test_textured_material_on_no_uv_geometry_raises -v`
Expected: FAIL — either no exception (silent wrong render) or a generic wgpu validation error.

- [ ] **Step 3: Add the error class and the check**

In `src/manifoldx/renderer.py`, near the top of the module, add:

```python
class MaterialGeometryMismatchError(ValueError):
    """A material requires geometry attributes (e.g. UVs) the geometry doesn't have."""
```

In `_get_or_create_pipeline`, in the mesh textured branch (just after the `if material_subtype == "textured":` decision), inspect the geometry registry for the bound geom and raise if it has no UVs:

```python
            if material_subtype == "textured":
                gpu_buffers = registry.geometry_registry.get_gpu_buffers(geometry_id) \
                    if hasattr(registry, "geometry_registry") else None
                # Fall back to the engine's geometry registry through the bound store.
                # Most importantly: check the geometry's own dict for "uvs".
                geom_obj = self._geometry_registry.get(geometry_id) \
                    if hasattr(self, "_geometry_registry") else None
                if geom_obj is not None and "uvs" not in geom_obj:
                    name = geom_obj.get("name", f"id={geometry_id}")
                    raise MaterialGeometryMismatchError(
                        f"StandardMaterial(albedo_map=...) requires geometry with "
                        f"UVs; geometry '{name}' has none"
                    )
```

> **Note for the executor:** the renderer has access to the engine's geometry registry via a pre-existing back-reference; grep `_geometry_registry` or `engine._geometry_registry` in `renderer.py` to find the canonical accessor. Adjust the snippet to use whatever the existing code uses. The check itself is just: "does the geometry dict have `"uvs"`? if not, raise."

- [ ] **Step 4: Run the new test**

Run: `uv run pytest tests/test_textured_material.py -v`
Expected: both tests pass.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/renderer.py tests/test_textured_material.py
git commit -m "feat(renderer): MaterialGeometryMismatchError for textured-mat without UVs"
```

---

## Slice 2 — OBJ loader

### Task 9: `assets/obj.py` — parse a single-triangle OBJ

**Files:**
- Create: `src/manifoldx/assets/__init__.py`
- Create: `src/manifoldx/assets/obj.py`
- Test: `tests/test_obj_loader.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obj_loader.py
import numpy as np
import pytest


SINGLE_TRIANGLE = """\
# tiny test OBJ
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.0 1.0
f 1/1/1 2/2/1 3/3/1
"""


def test_parses_single_triangle(tmp_path):
    from manifoldx.assets.obj import load_obj

    obj_path = tmp_path / "tri.obj"
    obj_path.write_text(SINGLE_TRIANGLE)

    geo = load_obj(obj_path)
    assert set(geo.keys()) >= {"positions", "normals", "uvs", "indices", "name"}
    assert geo["positions"].shape == (3, 3)
    assert geo["normals"].shape == (3, 3)
    assert geo["uvs"].shape == (3, 2)
    assert geo["indices"].shape == (3,)
    assert geo["indices"].dtype == np.uint32
    assert geo["positions"].dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_obj_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: manifoldx.assets`.

- [ ] **Step 3: Implement minimal loader**

```python
# src/manifoldx/assets/__init__.py
from manifoldx.assets.obj import load_obj, ObjParseError

__all__ = ["load_obj", "ObjParseError"]
```

```python
# src/manifoldx/assets/obj.py
"""Minimal Wavefront OBJ parser.

Supports the four face-line forms (1-indexed):
    f v ...
    f v/vt ...
    f v/vt/vn ...
    f v//vn ...

All face lines in a file must use the same form. Polygon faces with >3
vertices are fan-triangulated. Materials (`mtllib`, `usemtl`) and
grouping directives (`o`, `g`, `s`) are silently ignored — v1 carries
material info through Python kwargs, not MTL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ObjParseError(ValueError):
    """The OBJ file is malformed or uses a feature v1 doesn't support."""


def load_obj(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()

    raw_positions: list[list[float]] = []
    raw_normals: list[list[float]] = []
    raw_uvs: list[list[float]] = []
    face_triples: list[tuple] = []  # list of (pi, ti|None, ni|None) per triangle vertex
    face_form: Optional[str] = None  # one of "v", "v/vt", "v/vt/vn", "v//vn"

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, *rest = line.split()
        if head == "v":
            raw_positions.append([float(x) for x in rest[:3]])
        elif head == "vn":
            raw_normals.append([float(x) for x in rest[:3]])
        elif head == "vt":
            raw_uvs.append([float(x) for x in rest[:2]])
        elif head == "f":
            tokens = rest
            parsed, this_form = _parse_face(lineno, tokens)
            if face_form is None:
                face_form = this_form
            elif face_form != this_form:
                raise ObjParseError(
                    f"line {lineno}: face-line form changed from "
                    f"'{face_form}' to '{this_form}'; pick one"
                )
            # Fan-triangulate.
            for i in range(1, len(parsed) - 1):
                face_triples.append(parsed[0])
                face_triples.append(parsed[i])
                face_triples.append(parsed[i + 1])
        # Silently ignored: o, g, s, mtllib, usemtl.

    return _build_geometry(p.stem, raw_positions, raw_normals, raw_uvs,
                           face_triples, face_form)


def _parse_face(lineno: int, tokens: list[str]) -> tuple[list[tuple], str]:
    """Parse one face line's tokens into a list of (pi, ti|None, ni|None)
    triples and detect the face-line form."""
    if len(tokens) < 3:
        raise ObjParseError(
            f"line {lineno}: face needs at least 3 vertices, got {len(tokens)}"
        )

    parsed = []
    forms = set()
    for tok in tokens:
        if "//" in tok:
            # v//vn
            pi_s, ni_s = tok.split("//")
            pi = _idx(lineno, pi_s)
            ni = _idx(lineno, ni_s)
            parsed.append((pi, None, ni))
            forms.add("v//vn")
        elif "/" in tok:
            parts = tok.split("/")
            if len(parts) == 2:
                # v/vt
                pi = _idx(lineno, parts[0])
                ti = _idx(lineno, parts[1])
                parsed.append((pi, ti, None))
                forms.add("v/vt")
            elif len(parts) == 3:
                # v/vt/vn
                pi = _idx(lineno, parts[0])
                ti = _idx(lineno, parts[1])
                ni = _idx(lineno, parts[2])
                parsed.append((pi, ti, ni))
                forms.add("v/vt/vn")
            else:
                raise ObjParseError(f"line {lineno}: malformed face token '{tok}'")
        else:
            # v
            pi = _idx(lineno, tok)
            parsed.append((pi, None, None))
            forms.add("v")

    if len(forms) > 1:
        raise ObjParseError(
            f"line {lineno}: mixed face-line forms within a single face: {sorted(forms)}"
        )
    (this_form,) = forms
    return parsed, this_form


def _idx(lineno: int, s: str) -> int:
    """Parse a 1-indexed OBJ index. Negative (relative) indices unsupported."""
    n = int(s)
    if n < 0:
        raise ObjParseError(
            f"line {lineno}: negative face indices not supported in v1; "
            f"re-export with absolute indices"
        )
    return n - 1  # convert to 0-indexed


def _build_geometry(name, raw_positions, raw_normals, raw_uvs,
                    face_triples, face_form):
    has_uv = face_form in ("v/vt", "v/vt/vn")
    has_normal = face_form in ("v/vt/vn", "v//vn")

    dedup: dict[tuple, int] = {}
    positions_out: list[list[float]] = []
    normals_out: list[list[float]] = []
    uvs_out: list[list[float]] = []
    indices_out: list[int] = []

    for (pi, ti, ni) in face_triples:
        key = (pi, ti, ni)
        if key in dedup:
            indices_out.append(dedup[key])
            continue
        vi = len(positions_out)
        dedup[key] = vi
        positions_out.append(raw_positions[pi])
        if has_normal and ni is not None:
            normals_out.append(raw_normals[ni])
        if has_uv and ti is not None:
            uvs_out.append(raw_uvs[ti])
        indices_out.append(vi)

    geo = {
        "name": name,
        "positions": np.asarray(positions_out, dtype=np.float32),
        "indices": np.asarray(indices_out, dtype=np.uint32),
    }
    if has_normal:
        geo["normals"] = np.asarray(normals_out, dtype=np.float32)
    if has_uv:
        geo["uvs"] = np.asarray(uvs_out, dtype=np.float32)
    return geo
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_obj_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/assets/ tests/test_obj_loader.py
git commit -m "feat(assets): load_obj — single-triangle parse"
```

---

### Task 10: Quad fan-triangulation

**Files:**
- Test: `tests/test_obj_loader.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obj_loader.py — append
QUAD = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1/1 2/2/1 3/3/1 4/4/1
"""


def test_quad_fan_triangulates(tmp_path):
    from manifoldx.assets.obj import load_obj
    obj_path = tmp_path / "quad.obj"
    obj_path.write_text(QUAD)

    geo = load_obj(obj_path)
    # Quad → 2 triangles → 6 indices.
    assert geo["indices"].shape == (6,)
    # 4 unique (pos, normal, uv) triples → 4 vertices.
    assert geo["positions"].shape == (4, 3)
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `uv run pytest tests/test_obj_loader.py::test_quad_fan_triangulates -v`
Expected: PASS — the loop in Task 9 already fan-triangulates.

> If it fails: the fan-triangulation in `load_obj` is wrong; fix it before continuing.

- [ ] **Step 3: Commit (test-only, locks behavior in)**

```bash
git add tests/test_obj_loader.py
git commit -m "test(obj): quad face fan-triangulates to 2 triangles"
```

---

### Task 11: Face-form validation tests

**Files:**
- Test: `tests/test_obj_loader.py` (extend)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_obj_loader.py — append
import pytest


def test_mixed_face_forms_raise(tmp_path):
    from manifoldx.assets.obj import load_obj, ObjParseError
    src = "\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "vn 0 0 1",
        "vt 0 0", "vt 1 0", "vt 0 1",
        "f 1/1/1 2/2/1 3/3/1",
        "f 1//1 2//1 3//1",  # different form on second face
    ])
    obj_path = tmp_path / "mixed.obj"
    obj_path.write_text(src)

    with pytest.raises(ObjParseError, match="form changed"):
        load_obj(obj_path)


def test_negative_indices_raise(tmp_path):
    from manifoldx.assets.obj import load_obj, ObjParseError
    src = "\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "f -3 -2 -1",
    ])
    obj_path = tmp_path / "neg.obj"
    obj_path.write_text(src)

    with pytest.raises(ObjParseError, match="negative"):
        load_obj(obj_path)


def test_v_double_slash_n_form_no_uvs(tmp_path):
    from manifoldx.assets.obj import load_obj
    src = "\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "vn 0 0 1",
        "f 1//1 2//1 3//1",
    ])
    obj_path = tmp_path / "no_uvs.obj"
    obj_path.write_text(src)

    geo = load_obj(obj_path)
    assert "uvs" not in geo
    assert "normals" in geo


def test_ignored_directives(tmp_path):
    from manifoldx.assets.obj import load_obj
    src = "\n".join([
        "mtllib teapot.mtl",
        "o Teapot",
        "g body",
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "vn 0 0 1",
        "vt 0 0", "vt 1 0", "vt 0 1",
        "usemtl porcelain",
        "s 1",
        "f 1/1/1 2/2/1 3/3/1",
    ])
    obj_path = tmp_path / "noisy.obj"
    obj_path.write_text(src)

    geo = load_obj(obj_path)
    assert geo["positions"].shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_obj_loader.py -v`
Expected: all PASS — Task 9's parser already implements these rules.

> If a test fails: fix the parser, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_obj_loader.py
git commit -m "test(obj): face-form validation + negative-index + ignored-directive coverage"
```

---

### Task 12: Top-level `manifoldx.load_obj` export

**Files:**
- Modify: `src/manifoldx/__init__.py`
- Test: `tests/test_obj_loader.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obj_loader.py — append
def test_top_level_export():
    import manifoldx as mx
    assert hasattr(mx, "load_obj")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_obj_loader.py::test_top_level_export -v`
Expected: FAIL — `mx.load_obj` doesn't exist.

- [ ] **Step 3: Add the export**

In `src/manifoldx/__init__.py`, find the existing exports block and add:

```python
from manifoldx.assets.obj import load_obj
```

And add `"load_obj"` to `__all__` if one exists in that file.

- [ ] **Step 4: Run the test and the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/__init__.py tests/test_obj_loader.py
git commit -m "feat(api): top-level manifoldx.load_obj export"
```

---

## Slice 3 — Teapot demo, asset prep, closeout

### Task 13: Bundle teapot asset

**Files:**
- Create: `examples/assets/teapot/teapot.obj`
- Create: `examples/assets/teapot/README.md`

- [ ] **Step 1: Source the OBJ**

Pick one of:

- **Preferred:** Casey Russell's UV-mapped Newell teapot OBJ from a permissive mirror (e.g. the McGuire computer-graphics archive at `https://casual-effects.com/data/`, which redistributes a UV-mapped teapot under a permissive license — verify the exact license on the page before downloading).
- **Fallback:** the bare Newell teapot OBJ from any public domain mirror, then run a one-time UV-unwrap with the contingency script in Task 14.

Download to `examples/assets/teapot/teapot.obj`. Confirm with a quick parser smoke:

```bash
uv run python -c "from manifoldx import load_obj; g = load_obj('examples/assets/teapot/teapot.obj'); print(g['positions'].shape, 'uvs:', 'uvs' in g)"
```

If `uvs: True`, proceed to step 3. If `uvs: False`, do Task 14 (UV unwrap contingency) before continuing.

- [ ] **Step 2: Write the README**

```markdown
# Teapot Asset

## teapot.obj

The classic Utah teapot (Martin Newell, 1975), with auto-generated UV
coordinates suitable for albedo-map sampling.

- Source: <fill in the actual URL you downloaded from>
- License: <fill in the license stated on that page>
- File: `teapot.obj`, ~<size> KB, <vert_count> verts, <tri_count> tris

## teapot_albedo.png

Procedurally generated by `scripts/gen_teapot_albedo.py`. Regenerate with:

```
uv run python scripts/gen_teapot_albedo.py
```

- Size: 512×512
- License: CC0 (procedural, no upstream)
```

Save as `examples/assets/teapot/README.md`. Fill in the placeholders from the actual download.

- [ ] **Step 3: Commit**

```bash
git add examples/assets/teapot/teapot.obj examples/assets/teapot/README.md
git commit -m "chore(assets): bundle Utah teapot OBJ for textured-PBR demo"
```

---

### Task 14: UV unwrap contingency (only if Task 13 step 1 yields no UVs)

**Skip this task if `teapot.obj` already had UVs.**

**Files:**
- Create: `scripts/prep_teapot_uvs.py`

- [ ] **Step 1: Write the spherical-projection script**

```python
# scripts/prep_teapot_uvs.py
"""One-shot: read a UV-less Newell teapot OBJ, assign spherical-projection
UVs (u = atan2(z, x) / 2pi + 0.5, v = asin(y / radius) / pi + 0.5), and
write the result back as a UV-mapped OBJ.

Spherical projection on a teapot is not seamless (the spout and handle
distort), but it's good enough for a "PBR works" demo. Better UV unwraps
belong in Blender, not in this repo.
"""

import sys
import numpy as np
from pathlib import Path

from manifoldx import load_obj


def main(in_path, out_path):
    geo = load_obj(in_path)
    pos = geo["positions"]
    radii = np.linalg.norm(pos, axis=1, keepdims=True)
    radii[radii == 0] = 1.0
    nrm = pos / radii
    u = np.arctan2(nrm[:, 2], nrm[:, 0]) / (2 * np.pi) + 0.5
    v = np.arcsin(np.clip(nrm[:, 1], -1, 1)) / np.pi + 0.5
    uvs = np.stack([u, v], axis=1).astype(np.float32)

    lines = []
    for p in pos:
        lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    for uv in uvs:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    if "normals" in geo:
        for n in geo["normals"]:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

    tris = geo["indices"].reshape(-1, 3)
    has_n = "normals" in geo
    for tri in tris:
        a, b, c = (int(i) + 1 for i in tri)
        if has_n:
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
        else:
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")

    Path(out_path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    in_p = sys.argv[1] if len(sys.argv) > 1 else "examples/assets/teapot/teapot_raw.obj"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "examples/assets/teapot/teapot.obj"
    main(in_p, out_p)
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/prep_teapot_uvs.py examples/assets/teapot/teapot_raw.obj examples/assets/teapot/teapot.obj
uv run python -c "from manifoldx import load_obj; g = load_obj('examples/assets/teapot/teapot.obj'); print(g['positions'].shape, 'uvs:', 'uvs' in g)"
```

Expected: `uvs: True` on the second command.

- [ ] **Step 3: Commit**

```bash
git add scripts/prep_teapot_uvs.py examples/assets/teapot/teapot.obj
git rm -f examples/assets/teapot/teapot_raw.obj 2>/dev/null || true
git commit -m "chore(assets): UV-unwrap script + generated UV-mapped teapot"
```

---

### Task 15: Procedural albedo PNG generator

**Files:**
- Create: `scripts/gen_teapot_albedo.py`
- Create: `examples/assets/teapot/teapot_albedo.png` (generated)

- [ ] **Step 1: Write the generator**

```python
# scripts/gen_teapot_albedo.py
"""Generate a 512x512 "blue-and-white china" albedo texture for the
teapot demo. Cobalt-blue floral spots on a cream background, tileable.

Reproducible: deterministic seed.
"""

import numpy as np
from PIL import Image
from pathlib import Path


SIZE = 512
SEED = 20260609
OUT_PATH = Path("examples/assets/teapot/teapot_albedo.png")


def main():
    rng = np.random.default_rng(SEED)

    # Background: warm cream (#f4ead8).
    img = np.full((SIZE, SIZE, 4), [244, 234, 216, 255], dtype=np.uint8)

    # Sprinkle ~24 cobalt-blue (#1d3a8a) radial-falloff "flowers".
    n_flowers = 24
    flower_color = np.array([29, 58, 138], dtype=np.float32)
    xs = rng.uniform(0, SIZE, size=n_flowers)
    ys = rng.uniform(0, SIZE, size=n_flowers)
    radii = rng.uniform(20, 55, size=n_flowers)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for cx, cy, r in zip(xs, ys, radii):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        falloff = np.clip(1.0 - d2 / (r * r), 0.0, 1.0) ** 1.5
        # 6-fold petal modulation.
        ang = np.arctan2(yy - cy, xx - cx)
        petals = 0.6 + 0.4 * np.cos(6 * ang)
        mask = (falloff * petals)[..., None]
        img_f = img[..., :3].astype(np.float32)
        img_f = img_f * (1 - mask) + flower_color * mask
        img[..., :3] = np.clip(img_f, 0, 255).astype(np.uint8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/gen_teapot_albedo.py
```

Expected: `wrote examples/assets/teapot/teapot_albedo.png`. Inspect the PNG to confirm cream + blue flowers.

- [ ] **Step 3: Commit**

```bash
git add scripts/gen_teapot_albedo.py examples/assets/teapot/teapot_albedo.png
git commit -m "chore(assets): procedural blue-and-white china albedo for the teapot"
```

---

### Task 16: `examples/teapot_demo.py`

**Files:**
- Create: `examples/teapot_demo.py`

- [ ] **Step 1: Write the demo**

```python
# examples/teapot_demo.py
"""Textured PBR teapot demo.

Renders the classic Utah teapot with a "blue-and-white china" albedo
map under the existing GGX-PBR shader, lit by four orbiting point
lights (same rig as pbr_demo.py).

Usage:
    uv run python examples/teapot_demo.py
    uv run python examples/teapot_demo.py --render --duration 4 --fps 30 \
        --output /tmp/teapot.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, PointLight
from manifoldx.textures import load_texture


def main():
    engine = mx.Engine("Textured PBR Teapot")

    teapot_geo = mx.load_obj("examples/assets/teapot/teapot.obj")
    albedo = load_texture(engine, "examples/assets/teapot/teapot_albedo.png")

    mat = StandardMaterial(
        color=(1.0, 1.0, 1.0),   # ignored in textured variant
        roughness=0.35,
        metallic=0.0,
        ao=1.0,
        albedo_map=albedo,
    )

    engine.spawn(
        Transform(position=(0.0, -0.5, 0.0), scale=(1.5, 1.5, 1.5)),
        Mesh(teapot_geo),
        Material(mat),
    )

    # Four lights in a square above the teapot.
    for x, z in [(2, 2), (-2, 2), (2, -2), (-2, -2)]:
        engine.add_light(PointLight(position=(x, 3, z),
                                     color=(1, 1, 1),
                                     intensity=18.0))

    engine._camera.position = np.array([0, 1.2, 4.5], dtype=np.float32)
    engine._camera.target = np.array([0, 0, 0], dtype=np.float32)

    @engine.on("frame")
    def orbit(dt, elapsed, frame):
        radius = 4.5
        engine._camera.position = np.array([
            radius * np.cos(elapsed * 0.5),
            1.2,
            radius * np.sin(elapsed * 0.5),
        ], dtype=np.float32)

    engine.run()


if __name__ == "__main__":
    main()
```

> **Note for the executor:** the precise add-light / on-frame / camera-set APIs are likely slightly different. Mirror the patterns from `examples/pbr_demo.py` and `examples/event_dolly.py` exactly. Don't invent new API calls.

- [ ] **Step 2: Run interactively**

```bash
uv run python examples/teapot_demo.py
```

Expected: a window opens with a porcelain-patterned teapot, lit by four lights, camera orbiting. Close the window after eyeballing.

- [ ] **Step 3: Run headless smoke render**

```bash
uv run python examples/teapot_demo.py --render --duration 2 --fps 30 --output /tmp/teapot.mp4
```

Expected: `/tmp/teapot.mp4` exists and contains 60 frames of teapot. Open it.

> If `--render`/`--duration`/`--fps`/`--output` aren't built into `Engine.run()`, follow the pattern from existing examples (search `examples/` for `--render`) to add them or to wrap with an off-screen render path.

- [ ] **Step 4: Commit**

```bash
git add examples/teapot_demo.py
git commit -m "feat(examples): teapot_demo — textured-PBR Utah teapot"
```

---

### Task 17: Integration test — load + render the real teapot

**Files:**
- Test: `tests/test_textured_material.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_textured_material.py — append
def test_teapot_demo_assets_render():
    """Loading the actual teapot OBJ + albedo PNG renders one frame."""
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=256, height=256)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    from manifoldx.resources import StandardMaterial, PointLight
    from manifoldx.textures import load_texture

    engine = mx.Engine("teapot-test")
    engine.attach_canvas(canvas)

    teapot_geo = mx.load_obj("examples/assets/teapot/teapot.obj")
    assert "uvs" in teapot_geo, "teapot asset is missing UVs"

    albedo = load_texture(engine, "examples/assets/teapot/teapot_albedo.png")
    mat = StandardMaterial(color=(1, 1, 1), roughness=0.35, metallic=0.0,
                           albedo_map=albedo)
    engine.spawn(Transform(), Mesh(teapot_geo), Material(mat))
    engine.add_light(PointLight(position=(2, 3, 3), color=(1, 1, 1), intensity=20.0))

    engine.render(duration=0.1, fps=30, output=None)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_textured_material.py::test_teapot_demo_assets_render -v`
Expected: PASS — the previous tasks have wired up everything needed.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_textured_material.py
git commit -m "test(integration): real teapot asset loads and renders"
```

---

### Task 18: Promote Pillow to a required dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current state**

```bash
grep -n "pillow\|Pillow" pyproject.toml
```

Note where Pillow is currently declared (likely in `[project.optional-dependencies]` under a `viz` key).

- [ ] **Step 2: Promote to required**

In `pyproject.toml`, move the `pillow>=N` line from `[project.optional-dependencies] viz = [...]` into the main `[project] dependencies = [...]` array. Keep the version pin. If `viz` becomes empty as a result, leave the empty list rather than deleting the key — other code may reference it.

- [ ] **Step 3: Regenerate the lock**

```bash
uv lock
```

- [ ] **Step 4: Verify a fresh install includes Pillow without the `viz` extra**

```bash
uv sync
uv run python -c "from PIL import Image; print('pillow', Image.__version__ if hasattr(Image, '__version__') else 'ok')"
```

Expected: prints `pillow ok` or a version string.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): promote pillow from [viz] extra to required"
```

---

### Task 19: CHANGELOG + closeout

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md` (Sub-projects in flight section)

- [ ] **Step 1: Add `[Unreleased]` features entry**

In `CHANGELOG.md`, under the existing `## [Unreleased] / ### Features` block, append:

```markdown
- **Textured PBR v1** — first slice of texture support in the core mesh-PBR path. New `manifoldx.textures` module exposes `load_texture(engine, path) -> TextureHandle`, backed by a per-engine `TextureRegistry`. `StandardMaterial` accepts a new `albedo_map: TextureHandle | None` kwarg; when set, the fragment shader samples `textureSample(albedo_tex, albedo_sampler, uv).rgb` instead of using the scalar `material.albedo`. Geometry dicts now carry an optional `"uvs"` key; `sphere()` and `plane()` emit UVs by default. `GeometryRegistry.create_buffers` interleaves `[pos, normal, uv]` (stride 32 B) when UVs are present, else falls back to the existing `[pos, normal]` (stride 24 B). The mesh pipeline cache now keys on `material_subtype`, so scalar and textured `StandardMaterial` instances coexist in one scene without collision. Trying to attach a textured `StandardMaterial` to a geometry without UVs raises `MaterialGeometryMismatchError` at pipeline-cache time. New `manifoldx.assets.obj.load_obj` (re-exported as `manifoldx.load_obj`) parses Wavefront OBJ files with all four face-line forms (`f v`, `f v/vt`, `f v/vt/vn`, `f v//vn`); fan-triangulates polygon faces; rejects mixed forms and negative indices. First bundled binary asset: `examples/assets/teapot/teapot.obj` (UV-mapped Utah teapot) + a procedural `teapot_albedo.png` regenerable via `scripts/gen_teapot_albedo.py`. New demo `examples/teapot_demo.py`. Plan: `.knowledge/plans/2026-06-09-textured-pbr-v1.md`. Design: `.knowledge/analysis/2026-06-09-textured-pbr-v1-design.md`.
```

Non-goals for v1 (documented in the design): normal maps, metallic-roughness maps, AO maps, environment / IBL, mipmaps, HDR / EXR, hot-reload.

- [ ] **Step 2: Update AGENTS.md sub-projects in flight**

In `AGENTS.md`, find the "Sub-projects in flight" section. Append:

```markdown
- **Textured PBR v1** — landed. `StandardMaterial` accepts an optional `albedo_map: TextureHandle` and samples it in the fragment shader; `manifoldx.textures.load_texture` decodes images via Pillow and uploads them as `Rgba8UnormSrgb`. `manifoldx.load_obj` parses Wavefront OBJ files. First bundled asset: a UV-mapped Utah teapot. Demo at `examples/teapot_demo.py`. Plans 2+ (normal / metallic-roughness / AO maps, IBL, mipmaps) not yet specced.
```

- [ ] **Step 3: Run the full suite one last time**

Run: `uv run pytest -q && uv run python examples/teapot_demo.py --render --duration 2 --fps 30 --output /tmp/teapot.mp4`
Expected: tests green + MP4 written.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md AGENTS.md
git commit -m "docs: changelog + AGENTS.md — textured PBR v1 closeout"
```

- [ ] **Step 5: Push**

```bash
git push origin main
```

---

## Self-Review Notes

**Spec coverage:**
- §Architecture/OBJ parser → Tasks 9–12.
- §Architecture/textures + TextureRegistry → Task 3.
- §Architecture/extended StandardMaterial → Task 5.
- §Architecture/GeometryRegistry UV-aware interleave → Task 2.
- §Architecture/pipeline cache subtype keying → Task 4 + Task 6.
- §Architecture/mesh-pass bind-group append → Task 7.
- §Architecture/demo → Tasks 13–16.
- §Architecture/asset → Tasks 13–15.
- §Error handling → covered: OBJ parsing (Tasks 9–11), texture size (Task 3), material type-error (Task 5), `MaterialGeometryMismatchError` (Task 8). Pillow's own UnidentifiedImageError is propagated by default — no explicit test, fine.
- §Testing → unit (Tasks 9–11), integration (Tasks 7, 8, 17), smoke render (Task 16).

**Placeholder scan:** No TBDs or "implement later" in the plan body. Two "Note for the executor" callouts flag places where the actual API needs grep-confirmation (Task 3 attach-canvas, Task 5 `_compile` call site, Task 8 geometry-registry accessor, Task 16 example boilerplate) — these are honest pointers to verify-then-write, not placeholders.

**Type consistency:**
- `TextureHandle` fields: `id, texture, view, sampler, size` — consistent across Tasks 3, 5, 7, 8.
- `StandardMaterial.albedo_map` and `pipeline_subtype == "textured"` consistent across Tasks 4, 5, 6, 7, 8.
- `MaterialGeometryMismatchError` defined in Task 8, referenced only in Task 8.
- `ObjParseError` defined in Task 9, referenced in Task 11.
- Geometry dict keys (`positions`, `normals`, `uvs`, `indices`, `name`) consistent across Tasks 1, 2, 9, 13, 17.
