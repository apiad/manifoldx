# Shadow Mapping v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add directional-sun lighting + hard shadow mapping to the `StandardMaterial` PBR path, so a `manifoldx` scene casts real shadows from a single sun.

**Architecture:** A new engine-level "sun" (`DirectionalLight`) feeds a directional term into the `StandardMaterial` shader (which today has none). A depth-only shadow pass, encoded before the main render pass, rasterizes all mesh geometry from the sun's POV into an offscreen `depth24plus` shadow map. The `StandardMaterial` fragment shader projects each `world_pos` into light space and samples the shadow map with a comparison sampler, attenuating only the sun term. Sun params + `light_view_proj` + shadow params ride in the existing `Globals` uniform (grown 240 → 352 bytes).

**Tech Stack:** Python 3.13+, `wgpu`, numpy, `uv`. WGSL shaders as Python strings in `src/manifoldx/resources.py`.

## Global Constraints

- Python 3.13+, all invocations via `uv run`.
- Conventional commits: `feat(shadows):`, `test(shadows):`, `fix(shadows):`, `docs(shadows):`.
- GPU tests gate on `get_offscreen_canvas` and `pytest.skip` when the backend is unavailable, so the suite stays runnable headless. Pattern to copy: existing IBL/PBR tests in `tests/`.
- `make test` = full suite (offscreen wgpu backend). `make lint` = ruff. `make format` = ruff format.
- Only `StandardMaterial` receives sun + shadows. Sprite/label/volume/axis/GUI paths and `BasicMaterial`/`ColormapMaterial`/`VolumeMaterial` are untouched.
- `Globals` uniform is exactly **352 bytes** after this work (was 240). Every binding-0 buffer `size` must move 240 → 352 in lockstep (`renderer.py:_ensure_pipeline`, `renderer.py` globals packing, `render/passes/mesh.py` bind group).
- New `Globals` layout appended after the existing 240 bytes: `light_view_proj` mat4 @240; `sun_direction` vec3 + pad @304; `sun_color` vec3 + `sun_intensity` f32 @320; `shadow_enabled` u32 + `shadow_bias` f32 + `shadow_map_size` f32 + pad @336. Offsets 304/320 == `DirectionalLight.get_data()`'s 32-byte block.

---

## File structure

- **Modify** `src/manifoldx/resources.py` — grow the `Globals` WGSL struct in `_STANDARDMATERIAL_SHADER`; add the `calculateSun` directional term + shadow sampling to `fs_main`; the textured variant inherits via its existing `.replace(...)` chain.
- **Modify** `src/manifoldx/engine.py` — add `set_sun(light)`, `enable_shadows(...)`, state `_sun` / `_shadow_config`; hook the shadow pass into `_draw_frame` before the main render pass; create/resize the shadow map.
- **Create** `src/manifoldx/shadow.py` — pure-numpy `compute_light_view_proj(...)` (ortho + look-at). Re-export helper name from package root not required.
- **Create** `src/manifoldx/render/passes/shadow.py` — `render_shadow_map(rp, engine, command_encoder)` depth-only pass.
- **Modify** `src/manifoldx/renderer.py` — grow `_globals_buffer` to 352 + globals packing (sun + `light_view_proj` + shadow params); add shadow GPU resources (shadow map, comparison sampler, depth-only pipeline, placeholder bind group, `_shadow_batch_buffers`); add group-2 shadow bind-group layout to the `StandardMaterial` pipeline; a `_sun_light_view_proj(engine)` helper; a `_collect_mesh_instances(engine)` helper.
- **Modify** `src/manifoldx/render/passes/mesh.py` — bind group 0 `size` 240 → 352; bind group 2 (shadow map active-or-placeholder).
- **Create** `examples/shadow_demo.py` — plane + sphere under a sun, orbiting camera.
- **Modify** `CHANGELOG.md` — `[Unreleased]` entry.
- **Create** `tests/test_shadow.py` — pure-numpy matrix + config tests. **Create** `tests/test_shadow_render.py` — GPU-gated render tests.

---

## Task 1: Directional sun in StandardMaterial (no shadows yet)

Grow `Globals` to its final 352-byte shape (one resize, not two), add engine `set_sun`, pack the sun, and add the `calculateSun` term to the PBR shader. Shadow fields exist but stay zero/identity. Deliverable: directional lighting works — a sphere lit only by a sun shows a Lambert gradient.

**Files:**
- Modify: `src/manifoldx/resources.py` (Globals struct ~128-139; `calculateSun` fn + `fs_main` ~242-284)
- Modify: `src/manifoldx/engine.py` (add `_sun` init near line 118; `set_sun` near `set_lights` ~298)
- Modify: `src/manifoldx/renderer.py` (buffer size line 288; globals packing 1086-1108)
- Modify: `src/manifoldx/render/passes/mesh.py` (binding-0 `size` line 96)
- Test: `tests/test_shadow.py`, `tests/test_shadow_render.py`

**Interfaces:**
- Produces: `engine.set_sun(light: DirectionalLight) -> None` storing `engine._sun`. `Globals` is 352 bytes with sun packed at offset 304 (`sun.get_data()` bytes), `light_view_proj` = identity @240, shadow fields = 0 @336.
- Consumes: existing `DirectionalLight` (`resources.py:911`) with `.get_data()` → 8 float32 `[dir(3), pad, color(3), intensity]`.

- [ ] **Step 1: Write the failing test (engine stores sun; globals is 352B)**

Create `tests/test_shadow_render.py`:

```python
import numpy as np
import pytest

from manifoldx import Engine
from manifoldx.resources import DirectionalLight, StandardMaterial, sphere


def _offscreen_engine():
    try:
        eng = Engine(width=64, height=64, offscreen=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"offscreen wgpu unavailable: {exc}")
    return eng


def test_set_sun_stores_light():
    eng = _offscreen_engine()
    sun = DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -1.0, -0.3))
    eng.set_sun(sun)
    assert eng._sun is sun


def test_globals_buffer_is_352_bytes():
    eng = _offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, -1, 0)))
    s = eng.add_geometry(sphere())
    eng.spawn(mesh=s, material=StandardMaterial(color="#ffffff", roughness=0.5, metallic=0.0))
    eng.render_frame()  # one headless frame
    assert eng._render_pipeline._globals_buffer.size == 352
```

> NOTE (execution): confirm the real offscreen-engine constructor + one-frame-render API before running — grep `tests/` for the exact `Engine(...)` kwargs and the headless render call (`render_frame`, `render`, or `_draw_frame`). Copy the working pattern from an existing GPU test (e.g. the IBL/PBR render test) verbatim into `_offscreen_engine()` and the frame call. Do not invent kwargs.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_shadow_render.py -v`
Expected: FAIL — `Engine` has no `set_sun` / `_sun`.

- [ ] **Step 3: Add sun state + `set_sun` to the engine**

In `src/manifoldx/engine.py`, near the external-lights init (~line 118, after `self._lights = []`):

```python
        # Directional sun (separate from the point-light array). Consumed by
        # StandardMaterial as a directional term; also the shadow caster.
        self._sun = None
```

Near `set_lights` (~line 298):

```python
    def set_sun(self, light):
        """Set the single directional sun (a DirectionalLight).

        The sun is engine-level state (like the IBL environment), distinct
        from the point-light array. StandardMaterial adds a directional term
        for it; enable_shadows() makes it cast.
        """
        self._sun = light
```

- [ ] **Step 4: Grow the `Globals` struct in the shader**

In `src/manifoldx/resources.py`, replace the `Globals` struct inside `_STANDARDMATERIAL_SHADER` (lines 128-139) with:

```wgsl
struct Globals {
    vp:              mat4x4<f32>,   // offset   0
    view:            mat4x4<f32>,   // offset  64
    proj:            mat4x4<f32>,   // offset 128
    camera_pos:      vec3<f32>,     // offset 192
    _pad0:           f32,           // offset 204
    viewport_size:   vec2<f32>,     // offset 208
    _pad1:           vec2<f32>,     // offset 216
    ibl_intensity:   f32,           // offset 224
    ibl_enabled:     u32,           // offset 228
    _pad_ibl:        vec2<f32>,     // offset 232
    light_view_proj: mat4x4<f32>,   // offset 240
    sun_direction:   vec3<f32>,     // offset 304
    _pad_sun0:       f32,           // offset 316
    sun_color:       vec3<f32>,     // offset 320
    sun_intensity:   f32,           // offset 332
    shadow_enabled:  u32,           // offset 336
    shadow_bias:     f32,           // offset 340
    shadow_map_size: f32,           // offset 344
    _pad_shadow:     f32,           // offset 348
};
```

- [ ] **Step 5: Add the `calculateSun` directional term + call it in `fs_main`**

In `src/manifoldx/resources.py`, add this function right after `calculatePointLight` (after line 218):

```wgsl
fn calculateSun(N: vec3<f32>, V: vec3<f32>, F0: vec3<f32>, albedo: vec3<f32>,
                metallic: f32, roughness: f32) -> vec3<f32> {
    // Directional light: L points toward the light = -direction. No distance falloff.
    let L = normalize(-globals.sun_direction);
    let H = normalize(V + L);
    let radiance = globals.sun_color * globals.sun_intensity;
    let NDF = distributionGGX(N, H, roughness);
    let G   = geometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let kD  = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}
```

In `fs_main`, after the point-light loop (after line 257, before the `ambient` line), add:

```wgsl
    if globals.sun_intensity > 0.0 {
        Lo += calculateSun(N, V, F0, material.albedo, material.metallic, material.roughness);
    }
```

- [ ] **Step 6: Grow the globals buffer to 352 bytes**

In `src/manifoldx/renderer.py`, line 288, change `size=240` → `size=352`. Update the comment above it (285-286) to note the appended sun/shadow block totals 352.

- [ ] **Step 7: Pack the sun + identity light_view_proj into globals**

In `src/manifoldx/renderer.py`, in the globals packing block (rewrite lines 1087-1108). Change `np.zeros(240, ...)` → `np.zeros(352, ...)` and after the existing IBL write (before `write_buffer`), append:

```python
        # light_view_proj @240 — identity until shadows populate it (Task 3).
        import numpy as _np  # (module already imports numpy as np at top; use np)
        lvp = self._sun_light_view_proj(engine)  # (4,4) float32 row-major math matrix
        globals_data[240:304] = np.frombuffer(
            lvp.T.astype(np.float32).tobytes(), dtype=np.uint8
        )

        # sun_direction+pad @304, sun_color+intensity @320 — 32 bytes from get_data().
        sun = getattr(engine, "_sun", None)
        if sun is not None:
            globals_data[304:336] = np.frombuffer(
                sun.get_data().astype(np.float32).tobytes(), dtype=np.uint8
            )

        # shadow_enabled/bias/map_size @336 — zero until Task 3/4 enable them.
```

Add the helper method on `RenderPipeline` (near the globals packing method):

```python
    def _sun_light_view_proj(self, engine):
        """Light-space view-projection for the sun (identity until shadows are on)."""
        import numpy as np
        cfg = getattr(engine, "_shadow_config", None)
        sun = getattr(engine, "_sun", None)
        if cfg is None or sun is None:
            return np.eye(4, dtype=np.float32)
        from manifoldx.shadow import compute_light_view_proj
        return compute_light_view_proj(
            direction=np.asarray(sun.direction, dtype=np.float32),
            target=np.asarray(cfg["target"], dtype=np.float32),
            extent=cfg["extent"], near=cfg["near"], far=cfg["far"],
        )
```

> NOTE: `_shadow_config` does not exist until Task 3; `getattr(..., None)` keeps Task 1 green (helper returns identity).

- [ ] **Step 8: Bump the mesh-pass binding-0 size**

In `src/manifoldx/render/passes/mesh.py`, line 96, change `"size": 240` → `"size": 352`.

- [ ] **Step 9: Run tests + a lit-sphere visual check**

Add to `tests/test_shadow_render.py`:

```python
def test_sun_lights_sphere_with_gradient():
    eng = _offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-1, -1, -1)))
    s = eng.add_geometry(sphere())
    eng.spawn(mesh=s, material=StandardMaterial(color="#ffffff", roughness=0.5, metallic=0.0))
    img = eng.render_to_array()  # HxWx4 uint8 — confirm real API name during execution
    lum = img[..., :3].mean(axis=2)
    lit = lum[lum > 8]  # ignore background
    assert lit.size > 0
    assert lit.max() - lit.min() > 20  # a gradient, not flat
```

Run: `uv run pytest tests/test_shadow_render.py -v`
Expected: PASS (all three). If `render_to_array` isn't the real name, grep tests for the headless pixel-readback helper and use it.

- [ ] **Step 10: Lint + commit**

```bash
uv run ruff format src/manifoldx tests examples
uv run ruff check src/manifoldx tests examples
git add src/manifoldx/resources.py src/manifoldx/engine.py src/manifoldx/renderer.py src/manifoldx/render/passes/mesh.py tests/test_shadow_render.py
git commit -m "feat(shadows): directional sun term in StandardMaterial + Globals 240->352B"
```

---

## Task 2: Light-space matrix math (pure numpy)

`compute_light_view_proj` — ortho projection × look-at from the sun. No GPU. This is the math the shadow pass and the globals packing both consume.

**Files:**
- Create: `src/manifoldx/shadow.py`
- Test: `tests/test_shadow.py`

**Interfaces:**
- Produces: `compute_light_view_proj(direction, target, extent, near, far, back_distance=None) -> np.ndarray` — a `(4,4)` float32 row-major matrix such that `M @ [x,y,z,1]` yields clip coords; wgpu NDC has `z ∈ [0,1]`, `x,y ∈ [-1,1]`. `back_distance` defaults to `far * 0.5`.
- Consumes: nothing (pure numpy).

- [ ] **Step 1: Write the failing test**

Create `tests/test_shadow.py`:

```python
import numpy as np

from manifoldx.shadow import compute_light_view_proj


def _project(M, p):
    v = M @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
    return v[:3] / v[3]


def test_target_maps_near_ndc_center():
    M = compute_light_view_proj(
        direction=(0, -1, 0), target=(0, 0, 0), extent=10.0, near=0.1, far=50.0
    )
    ndc = _project(M, (0, 0, 0))
    assert abs(ndc[0]) < 1e-4 and abs(ndc[1]) < 1e-4
    assert 0.0 <= ndc[2] <= 1.0


def test_point_inside_extent_is_in_ndc_range():
    M = compute_light_view_proj(
        direction=(0, -1, 0), target=(0, 0, 0), extent=10.0, near=0.1, far=50.0
    )
    ndc = _project(M, (5.0, 0.0, -5.0))
    assert -1.0 <= ndc[0] <= 1.0
    assert -1.0 <= ndc[1] <= 1.0
    assert 0.0 <= ndc[2] <= 1.0


def test_point_outside_extent_is_out_of_range():
    M = compute_light_view_proj(
        direction=(0, -1, 0), target=(0, 0, 0), extent=10.0, near=0.1, far=50.0
    )
    ndc = _project(M, (50.0, 0.0, 0.0))
    assert abs(ndc[0]) > 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_shadow.py -v`
Expected: FAIL — no module `manifoldx.shadow`.

- [ ] **Step 3: Implement the module**

Create `src/manifoldx/shadow.py`:

```python
"""Light-space matrices for shadow mapping (pure numpy, no GPU)."""

import numpy as np


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(np.dot(f, up)) > 0.999:  # degenerate: sun straight down/up
        up = np.array([0.0, 0.0, 1.0])
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def _ortho(extent, near, far):
    """Symmetric ortho box [-extent,extent]^2, depth near..far, wgpu z in [0,1]."""
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = 1.0 / extent
    M[1, 1] = 1.0 / extent
    M[2, 2] = -1.0 / (far - near)
    M[2, 3] = -near / (far - near)
    M[3, 3] = 1.0
    return M


def compute_light_view_proj(direction, target, extent, near, far, back_distance=None):
    """Ortho light-space view-projection for a directional sun.

    Returns a (4,4) float32 matrix M with M @ [x,y,z,1] -> clip coords
    (wgpu convention: x,y in [-1,1], z in [0,1]).
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    target = np.asarray(target, dtype=np.float64)
    if back_distance is None:
        back_distance = far * 0.5
    eye = target - direction * back_distance
    view = _look_at(eye, target, up=np.array([0.0, 1.0, 0.0]))
    proj = _ortho(extent, near, far)
    return (proj @ view).astype(np.float32)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_shadow.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/manifoldx/shadow.py tests/test_shadow.py
uv run ruff check src/manifoldx/shadow.py tests/test_shadow.py
git add src/manifoldx/shadow.py tests/test_shadow.py
git commit -m "feat(shadows): pure-numpy light-space view-projection matrix"
```

---

## Task 3: Shadow map resources + depth-only shadow pass

Add `engine.enable_shadows(...)`, the shadow GPU resources, and the depth-only pass that renders mesh geometry from the sun's POV. After this task the shadow map is written each frame (not yet sampled by the material). `light_view_proj` now flows into globals via the Task 1 helper (which stops returning identity once `_shadow_config` + `_sun` exist).

**Files:**
- Modify: `src/manifoldx/engine.py` (`enable_shadows`, `_shadow_config` init; shadow-pass hook in `_draw_frame` ~before line 598)
- Create: `src/manifoldx/render/passes/shadow.py`
- Modify: `src/manifoldx/renderer.py` (shadow GPU resources in `_ensure_pipeline`; `_collect_mesh_instances`; `_shadow_batch_buffers`; a `_SHADOW_SHADER`; shadow pipeline/layout lazy-create)
- Test: `tests/test_shadow_render.py`

**Interfaces:**
- Produces: `engine.enable_shadows(target=(0,0,0), extent=10.0, resolution=2048, near=0.1, far=50.0, bias=0.005) -> None` storing `engine._shadow_config` (dict with those keys). `rp._shadow_map` (wgpu texture, `depth24plus`, resolution²), `rp._shadow_map_view`, `rp.render_shadow_map(engine, command_encoder)`.
- Consumes: `compute_light_view_proj` (Task 2); `rp._sun_light_view_proj` (Task 1); `Globals` binding-0 buffer (for `light_view_proj` in the shadow vertex shader).

- [ ] **Step 1: Write the failing test (config + shadow map allocation)**

Add to `tests/test_shadow_render.py`:

```python
def test_enable_shadows_stores_config():
    eng = _offscreen_engine()
    eng.enable_shadows(target=(0, 0, 0), extent=8.0, resolution=1024, bias=0.004)
    cfg = eng._shadow_config
    assert cfg["extent"] == 8.0 and cfg["resolution"] == 1024 and cfg["bias"] == 0.004


def test_shadow_map_allocated_when_enabled():
    eng = _offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, -1, 0)))
    eng.enable_shadows(target=(0, 0, 0), extent=10.0, resolution=512)
    from manifoldx.resources import plane
    p = eng.add_geometry(plane())
    eng.spawn(mesh=p, material=StandardMaterial(color="#ffffff", roughness=0.8, metallic=0.0))
    eng.render_frame()
    tex = eng._render_pipeline._shadow_map
    assert tex is not None
    assert tex.size[0] == 512 and tex.size[1] == 512
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_shadow_render.py -k shadow_map_allocated -v`
Expected: FAIL — no `enable_shadows` / `_shadow_map`.

- [ ] **Step 3: Add `enable_shadows` to the engine**

In `src/manifoldx/engine.py`, near `set_sun` (Task 1):

```python
    def enable_shadows(self, target=(0.0, 0.0, 0.0), extent=10.0, resolution=2048,
                       near=0.1, far=50.0, bias=0.005):
        """Enable directional shadow mapping for the sun set via set_sun()."""
        self._shadow_config = {
            "target": tuple(target), "extent": float(extent),
            "resolution": int(resolution), "near": float(near),
            "far": float(far), "bias": float(bias),
        }
```

And init `self._shadow_config = None` near `self._sun = None` (Task 1).

- [ ] **Step 4: Add the depth-only shadow shader + resources in the renderer**

In `src/manifoldx/renderer.py`, add a module-level shader string (near the other shaders):

```python
_SHADOW_SHADER = """
struct Globals {
    vp:              mat4x4<f32>,
    view:            mat4x4<f32>,
    proj:            mat4x4<f32>,
    camera_pos:      vec3<f32>,
    _pad0:           f32,
    viewport_size:   vec2<f32>,
    _pad1:           vec2<f32>,
    ibl_intensity:   f32,
    ibl_enabled:     u32,
    _pad_ibl:        vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction:   vec3<f32>,
    _pad_sun0:       f32,
    sun_color:       vec3<f32>,
    sun_intensity:   f32,
    shadow_enabled:  u32,
    shadow_bias:     f32,
    shadow_map_size: f32,
    _pad_shadow:     f32,
};
struct Transforms { models: array<mat4x4<f32>>, };
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>,
           @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
    let model = transforms.models[instance];
    let world = (model * vec4<f32>(position, 1.0)).xyz;
    return globals.light_view_proj * vec4<f32>(world, 1.0);
}
"""
```

In `_ensure_pipeline` (after the IBL block, before `self._initialized = True`), add:

```python
        # Shadow-mapping resources.
        self._shadow_map = None
        self._shadow_map_view = None
        self._shadow_map_size = 0
        self._shadow_batch_buffers = _BatchBuffers(device)
        self._shadow_sampler = device.create_sampler(compare=wgpu.CompareFunction.less)
        self._shadow_pipeline = None
        self._shadow_pipeline_layout = None
        # 1x1 placeholder depth texture so the StandardMaterial pipeline's
        # group(2) always has something to bind when shadows are off.
        placeholder = device.create_texture(
            size=(1, 1, 1), format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._shadow_placeholder_view = placeholder.create_view()
        self._shadow_bind_group_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.depth,
                         "view_dimension": wgpu.TextureViewDimension.d2}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.comparison}},
        ])
```

- [ ] **Step 5: Add `_collect_mesh_instances` + `render_shadow_map` to the renderer**

In `src/manifoldx/renderer.py`, add:

```python
    def _collect_mesh_instances(self, engine):
        """geom_id -> np.ndarray of column-major model matrices (N,16), mesh entities only."""
        import numpy as np
        store = self._store
        alive = store.get_alive_indices()
        if len(alive) == 0:
            return {}
        self._transform_cache.mark_dirty(alive)
        models = self._transform_cache.get_transforms(store, alive)  # (N,16) row-major
        mesh_data = store.get_component_data("Mesh", alive) if "Mesh" in store._components else None
        if mesh_data is None:
            return {}
        out = {}
        for i in range(len(alive)):
            geom_id = int(mesh_data[i, 0])
            if geom_id == 0:
                continue
            out.setdefault(geom_id, []).append(models[i])
        return {g: np.asarray(v, dtype=np.float32) for g, v in out.items()}
```

> NOTE (execution): mirror the *exact* alive-index + mesh-routing calls used in `_render_scene_passes` (lines 984-1078) — `get_alive_indices`, `get_component_data("Mesh", ...)`, and the `geom_id == 0` skip. If `_render_scene_passes` guards sprites/labels/axis specially, this simpler mesh-only collector is acceptable for v1 (demo is plane+sphere); note the limitation in the shadow module docstring.

Then the pass entry point delegates to `render/passes/shadow.py`:

```python
    def render_shadow_map(self, engine, command_encoder):
        from manifoldx.render.passes import shadow as _shadow_pass
        _shadow_pass.render_shadow_map(self, engine, command_encoder)
```

- [ ] **Step 6: Implement `render/passes/shadow.py`**

Create `src/manifoldx/render/passes/shadow.py`:

```python
"""Depth-only shadow pass — renders mesh geometry from the sun's POV."""

import numpy as np
import wgpu


def _ensure_shadow_map(rp, resolution):
    if rp._shadow_map is not None and rp._shadow_map_size == resolution:
        return
    rp._shadow_map = rp._device.create_texture(
        size=(resolution, resolution, 1),
        format=wgpu.TextureFormat.depth24plus,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    rp._shadow_map_view = rp._shadow_map.create_view()
    rp._shadow_map_size = resolution


def _ensure_pipeline(rp):
    if rp._shadow_pipeline is not None:
        return
    from manifoldx.renderer import _SHADOW_SHADER
    module = rp._device.create_shader_module(code=_SHADOW_SHADER)
    layout = rp._device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX,
         "buffer": {"type": wgpu.BufferBindingType.uniform}},
        {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
    ])
    rp._shadow_bind_layout_vs = layout
    rp._shadow_pipeline_layout = rp._device.create_pipeline_layout(bind_group_layouts=[layout])
    rp._shadow_pipeline = rp._device.create_render_pipeline(
        layout=rp._shadow_pipeline_layout,
        vertex={"module": module, "entry_point": "vs_main",
                "buffers": [{"array_stride": 6 * 4,  # pos(3)+normal(3); UV-bearing geoms use 8*4
                             "step_mode": wgpu.VertexStepMode.vertex,
                             "attributes": [{"format": wgpu.VertexFormat.float32x3,
                                             "offset": 0, "shader_location": 0}]}]},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                   "front_face": wgpu.FrontFace.ccw, "cull_mode": wgpu.CullMode.back},
        depth_stencil={"format": wgpu.TextureFormat.depth24plus,
                       "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less},
        fragment=None,
    )


def render_shadow_map(rp, engine, command_encoder):
    cfg = getattr(engine, "_shadow_config", None)
    sun = getattr(engine, "_sun", None)
    if cfg is None or sun is None or not rp._initialized:
        return
    _ensure_shadow_map(rp, cfg["resolution"])
    _ensure_pipeline(rp)

    instances = rp._collect_mesh_instances(engine)
    if not instances:
        return

    # Upload all model matrices (column-major) once, record per-geometry offsets.
    all_mats, draw_info, offset = [], {}, 0
    for geom_id, mats in instances.items():
        mats_t = mats.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        draw_info[geom_id] = (offset, len(mats))
        all_mats.append(mats_t)
        offset += len(mats)
    rp._shadow_batch_buffers.upload_transforms(np.concatenate(all_mats).astype(np.float32))

    bind_group = rp._device.create_bind_group(
        layout=rp._shadow_bind_layout_vs,
        entries=[
            {"binding": 0, "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 352}},
            {"binding": 1, "resource": {"buffer": rp._shadow_batch_buffers.transforms_buf,
                                        "offset": 0, "size": rp._shadow_batch_buffers.transforms_capacity}},
        ],
    )

    shadow_pass = command_encoder.begin_render_pass(
        color_attachments=[],
        depth_stencil_attachment={
            "view": rp._shadow_map_view,
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        },
    )
    shadow_pass.set_pipeline(rp._shadow_pipeline)
    shadow_pass.set_bind_group(0, bind_group)
    for geom_id, (first_instance, count) in draw_info.items():
        gpu = engine._geometry_registry.get_gpu_buffers(geom_id)
        if gpu is None:
            geom = engine._geometry_registry.get(geom_id)
            gpu = engine._geometry_registry.create_buffers(geom_id, geom, rp._device.queue) if geom else None
        if gpu is None:
            continue
        shadow_pass.set_vertex_buffer(0, gpu["vertex_buffer"])
        shadow_pass.set_index_buffer(gpu["index_buffer"], wgpu.IndexFormat.uint32)
        shadow_pass.draw_indexed(gpu["index_count"], count, first_index=0,
                                 base_vertex=0, first_instance=first_instance)
    shadow_pass.end()
```

> NOTE (execution): the shadow pipeline's vertex `array_stride` must equal each drawn geometry's actual stride (24B for pos+normal, 32B when the geometry carries UVs — `sphere()`/`plane()` emit UVs, so 32B). Read the real stride from `engine._geometry_registry.get_gpu_buffers(geom_id)["stride"]` and, if geometries with differing strides coexist, key one shadow pipeline per stride. For the plane+sphere demo both are UV-bearing (stride 32) — start with `8 * 4` and confirm against `create_buffers`' interleave logic (`renderer.py` GeometryRegistry). Only position (location 0) is read; normal/UV bytes are skipped by stride.

- [ ] **Step 7: Hook the shadow pass into `_draw_frame`**

In `src/manifoldx/engine.py`, in `_draw_frame`, right after `command_encoder = self._device.create_command_encoder()` (line 595) and before the main `begin_render_pass` (line 598):

```python
        # Shadow pass — depth-only, renders the scene from the sun's POV into
        # the shadow map, which the main mesh pass then samples. No-op unless
        # both a sun and enable_shadows() are set.
        self._render_pipeline.render_shadow_map(self, command_encoder)
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_shadow_render.py -k "shadow" -v`
Expected: PASS (`enable_shadows_stores_config`, `shadow_map_allocated_when_enabled`). Existing Task 1 tests still pass.

- [ ] **Step 9: Lint + commit**

```bash
uv run ruff format src/manifoldx examples tests
uv run ruff check src/manifoldx examples tests
git add src/manifoldx/engine.py src/manifoldx/renderer.py src/manifoldx/render/passes/shadow.py tests/test_shadow_render.py
git commit -m "feat(shadows): depth-only shadow pass + light-space matrix wired to globals"
```

---

## Task 4: Sample the shadow map in StandardMaterial

Add group(2) to the `StandardMaterial` pipeline (shadow map + comparison sampler, placeholder when off), sample it in `fs_main`, attenuate the sun term, and populate `shadow_enabled`/`shadow_bias`/`shadow_map_size` in globals. Deliverable: real cast shadows.

**Files:**
- Modify: `src/manifoldx/resources.py` (`fs_main` shadow bindings + sampling)
- Modify: `src/manifoldx/renderer.py` (add group(2) layout to StandardMaterial pipeline; pack shadow params in globals)
- Modify: `src/manifoldx/render/passes/mesh.py` (bind group 2: active or placeholder)
- Test: `tests/test_shadow_render.py`

**Interfaces:**
- Consumes: `rp._shadow_map_view`, `rp._shadow_placeholder_view`, `rp._shadow_sampler`, `rp._shadow_bind_group_layout` (Task 3); `Globals.shadow_enabled/shadow_bias/shadow_map_size` (Task 1 struct).
- Produces: a shadow factor multiplying the sun term; `shadow_enabled=1` in globals when sun+config present.

- [ ] **Step 1: Write the failing visual test**

Add to `tests/test_shadow_render.py`:

```python
def _plane_sphere_scene(shadows):
    from manifoldx.resources import plane
    eng = _offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.3)))
    if shadows:
        eng.enable_shadows(target=(0, 0, 0), extent=6.0, resolution=1024, bias=0.004)
    pl = eng.add_geometry(plane())
    eng.spawn(mesh=pl, material=StandardMaterial(color="#ffffff", roughness=0.9, metallic=0.0),
              position=(0, 0, 0), scale=(6, 1, 6))
    sp = eng.add_geometry(sphere())
    eng.spawn(mesh=sp, material=StandardMaterial(color="#ffffff", roughness=0.5, metallic=0.0),
              position=(0, 2, 0))
    # camera looking down at the plane — confirm real camera API during execution
    eng.camera.set_position((0, 8, 8)); eng.camera.look_at((0, 0, 0))
    return eng.render_to_array()


def test_sphere_casts_shadow_on_plane():
    lit = _plane_sphere_scene(shadows=False)
    shadowed = _plane_sphere_scene(shadows=True)
    # The plane region directly under the sphere should darken with shadows on.
    # Sample a patch of the plane near image center-bottom (execution: tune coords).
    h, w, _ = shadowed.shape
    patch = (slice(int(h * 0.55), int(h * 0.75)), slice(int(w * 0.4), int(w * 0.6)))
    lit_lum = lit[patch][..., :3].mean()
    shadow_lum = shadowed[patch][..., :3].mean()
    assert shadow_lum < lit_lum - 15
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_shadow_render.py -k casts_shadow -v`
Expected: FAIL — shadow not sampled yet, patches roughly equal.

- [ ] **Step 3: Add shadow bindings + sampling to the shader**

In `src/manifoldx/resources.py`, add group(2) bindings after the IBL group(1) block (after line 173):

```wgsl
@group(2) @binding(0) var shadow_map:     texture_depth_2d;
@group(2) @binding(1) var shadow_sampler: sampler_comparison;
```

Add a helper (after `calculateSun`):

```wgsl
fn sunShadow(world_pos: vec3<f32>) -> f32 {
    if globals.shadow_enabled == 0u { return 1.0; }
    let lp = globals.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 {
        return 1.0;  // outside the sun's frustum — unshadowed
    }
    return textureSampleCompareLevel(shadow_map, shadow_sampler, uv, ndc.z - globals.shadow_bias);
}
```

Change the sun call in `fs_main` (Task 1's block) to attenuate by the shadow factor:

```wgsl
    if globals.sun_intensity > 0.0 {
        let shadow = sunShadow(in.world_pos);
        Lo += calculateSun(N, V, F0, material.albedo, material.metallic, material.roughness) * shadow;
    }
```

- [ ] **Step 4: Add group(2) to the StandardMaterial pipeline layout**

In `src/manifoldx/renderer.py`, in `_get_or_create_pipeline`, where `needs_lights` builds `all_layouts` (lines 875-880), change to append the shadow layout:

```python
            if needs_lights:
                # group(1) = IBL textures, group(2) = shadow map + comparison sampler
                all_layouts = [bind_group_layout, self._ibl_bind_group_layout,
                               self._shadow_bind_group_layout]
            else:
                all_layouts = [bind_group_layout]
```

- [ ] **Step 5: Pack shadow params into globals**

In `src/manifoldx/renderer.py` globals packing (Task 1 block), replace the `# shadow_enabled ... zero` comment with:

```python
        cfg = getattr(engine, "_shadow_config", None)
        if cfg is not None and getattr(engine, "_sun", None) is not None:
            globals_data[336:340] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.uint8)
            globals_data[340:344] = np.frombuffer(np.float32(cfg["bias"]).tobytes(), dtype=np.uint8)
            globals_data[344:348] = np.frombuffer(
                np.float32(cfg["resolution"]).tobytes(), dtype=np.uint8)
```

- [ ] **Step 6: Bind group 2 in the mesh pass**

In `src/manifoldx/render/passes/mesh.py`, after the IBL `set_bind_group(1, ibl_bg)` block (after line 159), add:

```python
            shadow_on = (
                getattr(engine, "_shadow_config", None) is not None
                and getattr(engine, "_sun", None) is not None
                and rp._shadow_map_view is not None
            )
            shadow_view = rp._shadow_map_view if shadow_on else rp._shadow_placeholder_view
            shadow_bg = rp._device.create_bind_group(
                layout=rp._shadow_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": shadow_view},
                    {"binding": 1, "resource": rp._shadow_sampler},
                ],
            )
            render_pass.set_bind_group(2, shadow_bg)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_shadow_render.py -v`
Expected: PASS including `test_sphere_casts_shadow_on_plane`. If the shadow is offset/inverted, the fix is almost always the `light_view_proj` transpose in globals packing (Task 1 Step 7: try storing `lvp` without `.T`) or the UV Y-flip in `sunShadow`. Adjust `bias` if acne/peter-panning appears.

- [ ] **Step 8: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: PASS. The Globals 240→352 change touches every StandardMaterial render; existing IBL/PBR tests must stay green.

- [ ] **Step 9: Lint + commit**

```bash
uv run ruff format src/manifoldx tests
uv run ruff check src/manifoldx tests
git add src/manifoldx/resources.py src/manifoldx/renderer.py src/manifoldx/render/passes/mesh.py tests/test_shadow_render.py
git commit -m "feat(shadows): sample shadow map in StandardMaterial, attenuate sun term"
```

---

## Task 5: Demo + CHANGELOG

The `examples/shadow_demo.py` Alex wants to see, plus the changelog entry.

**Files:**
- Create: `examples/shadow_demo.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `engine.set_sun`, `engine.enable_shadows`, `StandardMaterial`, `sphere`, `plane`, `DirectionalLight` — all above.

- [ ] **Step 1: Write the demo**

Create `examples/shadow_demo.py` (model on `examples/pbr_demo.py` / `examples/teapot_demo.py` for the exact engine bootstrap, camera, and `--render` MP4 arg handling — copy their `if __name__` scaffold verbatim):

```python
"""Directional shadow mapping demo — a sphere casts a shadow on the ground."""

import numpy as np

from manifoldx import Engine
from manifoldx.resources import DirectionalLight, StandardMaterial, sphere, plane

engine = Engine(width=1024, height=768, title="Shadow Mapping")

engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.5, direction=(-0.4, -1.0, -0.35)))
engine.enable_shadows(target=(0, 0, 0), extent=8.0, resolution=2048, near=0.1, far=40.0, bias=0.004)

ground = engine.add_geometry(plane())
engine.spawn(mesh=ground, material=StandardMaterial(color="#cccccc", roughness=0.95, metallic=0.0),
             scale=(10, 1, 10))

ball = engine.add_geometry(sphere())
engine.spawn(mesh=ball, material=StandardMaterial(color="#e05a3a", roughness=0.4, metallic=0.1),
             position=(0, 2.0, 0))

# Orbiting camera (copy the exact frame-callback + camera API from pbr_demo.py).
@engine.on("frame")
def orbit(payload):
    t = payload["elapsed"]
    engine.camera.set_position((8 * np.cos(t * 0.4), 6.0, 8 * np.sin(t * 0.4)))
    engine.camera.look_at((0, 1, 0))

if __name__ == "__main__":
    engine.run()  # copy pbr_demo.py's __main__ incl. --render/--duration/--fps/--output
```

- [ ] **Step 2: Smoke-render the demo to MP4**

Run: `uv run python examples/shadow_demo.py --render --duration 2 --fps 30 --output /tmp/shadow_demo.mp4`
Expected: an MP4 where the sphere casts a moving shadow on the ground plane. Inspect a frame to confirm the shadow is present and roughly under the sphere.

- [ ] **Step 3: CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Features`, add:

```markdown
- **Shadow mapping v1 (directional)** — `StandardMaterial` now casts and receives hard shadows from a single directional sun. New `engine.set_sun(DirectionalLight(...))` adds a real directional term to the PBR shader (previously `DirectionalLight` was silently packed into the point-light array and unused), and `engine.enable_shadows(target, extent, resolution, near, far, bias)` makes the sun cast. A depth-only shadow pass (`src/manifoldx/render/passes/shadow.py`) rasterizes all mesh geometry from the sun's POV into a `depth24plus` shadow map, which the fragment shader samples via a comparison sampler (`textureSampleCompareLevel`, hardware 2×2 PCF) to attenuate the sun term. Light-space matrix math lives in the pure-numpy `manifoldx.shadow` module. The `Globals` uniform grew 240 → 352 bytes (`light_view_proj` + sun + shadow params). Demo at `examples/shadow_demo.py`. Point/spot shadows, larger-kernel PCF, and auto-fit frustum are follow-ups. Design: `.knowledge/analysis/2026-07-17-shadow-mapping-v1-design.md`. Plan: `.knowledge/plans/2026-07-17-shadow-mapping-v1-plan.md`.
```

- [ ] **Step 4: Commit**

```bash
git add examples/shadow_demo.py CHANGELOG.md
git commit -m "docs(shadows): shadow_demo example + CHANGELOG entry for shadow mapping v1"
```

---

## Self-review notes

- **Spec coverage:** sun term (Task 1), light-space math (Task 2), shadow pass + resources (Task 3), sampling + bind group 2 (Task 4), demo + changelog (Task 5). Globals 352B, group-2 layout, comparison sampler, bias, frustum-bounds guard, `StandardMaterial`-only scope — all covered.
- **Execution flags (honest unknowns to resolve against real code, not guesses):** (1) the exact offscreen `Engine(...)` constructor + headless render/readback API — copy from an existing GPU test; (2) camera API method names (`set_position`/`look_at`) — copy from `pbr_demo.py`; (3) the `light_view_proj` transpose convention (store `.T` vs raw) — validated by the Task 4 visual test, flip if the shadow is misplaced; (4) geometry vertex stride for the shadow pipeline (24 vs 32) — read from `get_gpu_buffers(...)["stride"]`. These are marked inline as NOTEs; none change the architecture.
- **Type consistency:** `_shadow_config` dict keys (`target/extent/resolution/near/far/bias`) are identical across `enable_shadows`, `_sun_light_view_proj`, and the globals packing. `compute_light_view_proj` signature matches both call sites.
```
