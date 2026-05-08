# Volume rendering v1 — design

**Status:** Design — approved through brainstorming on 2026-05-08.

**Goal:** Add a `Volume` ECS primitive plus a `VolumeMaterial` and a dedicated
render pass that performs **direct volume rendering (DVR)** of a 3D scalar
field via fragment-shader raymarching. One canonical sci-viz mode (DVR), one
shader path, one material, no GUI dependencies. Composes with the existing
sci-viz primitives (axes, labels, scale bars, colormap legends) in a single
frame.

**Non-goal:** A general-purpose volumetric renderer. v1 is scoped to be the
*minimum* viable production-grade DVR: unlit, fixed-step, single-channel
f32 scalar volume, CPU-uploaded. Lighting (gradient shading), isosurface
mode, MIP, multi-channel volumes, slicing, picking, empty-space skipping,
streaming/zarr ingestion, and editable transfer-function widgets are
explicit non-goals for v1.

---

## Locked decisions (from brainstorming)

1. **Render mode.** DVR only. Front-to-back over compositing with early ray
   termination at `accum.a > 0.99`. Isosurface and MIP are out — they would
   add shader branches with no current customer; transfer-function presets
   can approximate both well enough for v1.

2. **Primary data source.** "Both — same primitive." A single `Volume`
   component is the entry point for both numpy-uploaded static volumes and
   (in v2) compute-kernel-written volumes. v1 implements only the CPU-upload
   path; the registry/component shape and the `r32float` texture format are
   chosen so that v2 can swap a compute storage texture in without changing
   the entity-side API.

3. **Transfer function.** Two parallel 1D LUTs sampled per ray step.
   - **Color LUT** — reuse the six existing 1D RGBA8 colormap textures
     (viridis, magma, plasma, inferno, turbo, gray) already shipped by
     `manifoldx.viz.colormaps`. Material picks one by name.
   - **Opacity LUT** — a new 1D R32F texture, 256 samples, baked at
     material-init time from `opacity_stops` (a list of `(scalar, alpha)`
     stops in `[0, 1]`, piecewise-linear). Default ramp = `linspace(0, 1)`.

4. **Lighting.** Unlit. No gradient sampling, no normals, no light loop.
   This is the standard sci-viz default and is what most papers and tools
   show; lighting introduces visual bias on top of the transfer function
   that v1 deliberately avoids.

5. **Compute integration.** v1 wires only the CPU-upload path
   (`engine.register_volume(numpy_array)` → upload to `texture_3d`).
   Compute kernels writing volumes (`Writes[Volume3D]`) is a v2 plan.
   Forward-compat hook: `engine.bind_compute_volume(vol_id)` exists in
   v1 as a stub that raises `NotImplementedError("v2")`. Storage format
   (`r32float`) is chosen specifically to be compatible with WGSL
   `texture_storage_3d<r32float, write>` so the same texture object can
   be re-bound in v2 without re-allocation.

6. **Proxy geometry.** **Fullscreen quad with per-fragment ray/box-AABB
   intersection** in the fragment shader, not a back-face cube draw. Two
   reasons: (a) handles the camera-inside-volume case naturally (no need
   for a special "front-face fallback" pipeline state), and (b) avoids
   the depth-buffer artefacts that arise when cube faces clip against the
   near plane. Cost is one ray/box-intersection per pixel, which is
   trivial relative to the raymarch loop itself.

7. **Bounding box.** Volume occupies `[-0.5, +0.5]^3` in entity local
   space; world-space box dimensions are derived from `Transform.scale`.
   Translation and rotation come from `Transform`. No separate
   `bounds=...` argument on `register_volume` — Transform is the single
   place to control where the volume sits in world space, matching how
   `Mesh` + `Transform` works elsewhere.

8. **Render-pass ordering.** A 5th render pass added after sprite and
   before label, giving:
   ```
   mesh → sprite → volume → label → axis
   ```
   Volumes are translucent: depth-test ON, depth-write OFF, standard
   over-blending. Labels and axes overlay correctly; opaque mesh and
   sprite occlusion is preserved by the depth test.

9. **Success criterion.** A new `examples/volume_demo.py` renders a
   synthetic 64³ Gaussian-blob volume with viridis + a sloped opacity
   ramp at 30+ fps in `--render --quality low`, byte-for-byte
   deterministic across runs, and the full 360-test suite stays green.

---

## User-facing surface

### Imports

```python
from manifoldx.viz import Volume, VolumeMaterial
```

### Reference kernel — the v1 success bar

```python
import numpy as np
import manifoldx as mx
from manifoldx.viz import Volume, VolumeMaterial
from manifoldx.components import Material, Transform

# 1. Build/load a 3D scalar field as a contiguous float32 numpy array.
N = 64
xs = np.linspace(-1, 1, N, dtype=np.float32)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
density = np.exp(-(X**2 + Y**2 + Z**2) / 0.1).astype(np.float32)

# 2. Register with the engine; receive an integer handle.
engine = mx.Engine("Volume demo")
vol_id = engine.register_volume(density, name="gaussian_blob")

# 3. Spawn an entity that renders it.
engine.spawn(
    Volume(volume_id=vol_id),
    Material(VolumeMaterial(
        cmap="inferno",
        vmin=0.0, vmax=1.0,
        opacity_stops=[(0.0, 0.0), (0.3, 0.05), (1.0, 0.6)],
        density_scale=1.0,
    )),
    Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
    n=1,
)

engine.cli()
```

### `Volume` component

```python
class Volume(Component):
    """Reference to a registered 3D scalar field by integer handle.

    The voxel data itself lives in the engine-scoped volume registry,
    not on the entity. This matches the resource-pointer pattern used
    by Mesh (geometry handle) and Material (material handle).
    """
    volume_id: ScalarValue   # i32 handle
```

### `VolumeMaterial`

```python
class VolumeMaterial(Material):
    cmap: str = "viridis"
    vmin: float = 0.0
    vmax: float = 1.0
    opacity_stops: Sequence[tuple[float, float]] | np.ndarray | None = None
    density_scale: float = 1.0          # Global alpha multiplier
    step_size: float | None = None      # World-space ray step; None → auto
    max_steps: int = 256                # Per-ray cap; controls quality vs cost
```

- **`cmap`** — must be one of the names in `manifoldx.viz.colormaps`.
  Unknown names raise with the available-names list.
- **`vmin` / `vmax`** — the scalar normalization range. Voxel values are
  clamped to `[vmin, vmax]` then linearly mapped to `[0, 1]` for LUT
  lookup. `vmin >= vmax` raises.
- **`opacity_stops`** — accepted forms:
  - `Sequence[tuple[float, float]]` — `(scalar_in_0_1, alpha_in_0_1)`
    pairs, sorted ascending by scalar; piecewise-linearly interpolated
    into 256 samples.
  - `np.ndarray` of shape `(256,)`, `dtype=float32`, values in `[0, 1]`
    — used as-is.
  - `None` — defaults to `np.linspace(0, 1, 256)` (alpha = scalar).
- **`density_scale`** — global multiplier on the per-sample alpha. The
  knob users actually reach for to make a volume "more visible" or "more
  transparent." Default 1.0.
- **`step_size`** — world-space ray step in scene units. `None` →
  auto-derived as `min(box_dim_i / volume_dim_i) * 0.5` (half a voxel
  along the most finely-sampled axis). `<= 0` raises.
- **`max_steps`** — hard cap on samples per ray (early-exit safeguard).
  Rays that hit the cap commit accumulated `accum.rgba` and stop.

### Engine helpers

```python
def register_volume(
    self,
    data: np.ndarray,        # shape (Nz, Ny, Nx); float32; C-contiguous
    *,
    name: str | None = None,
) -> int:
    """Register a 3D scalar field, return an integer handle.

    Indexing convention: `data[k, j, i]` is the voxel at integer
    coordinates (i, j, k) along the local-space (x, y, z) axes — the
    standard numpy "slowest-axis-first" 3D layout. The texture is
    created at WGPU size `(Nx, Ny, Nz)`.
    """

def update_volume(self, vol_id: int, data: np.ndarray) -> None:
    """Replace voxel data for an existing handle. Same shape required.
    Bumps a dirty bit; renderer re-uploads on the next frame.
    """

def bind_compute_volume(self, vol_id: int, kernel_field: str) -> None:
    """v2: bind a Phase-2 compute kernel's `Writes[Volume3D]` field
    to this volume's storage texture. v1 raises NotImplementedError.
    """
```

---

## Internals

### Volume registry

`Engine._volume_registry: dict[int, _VolumeResource]` keyed by an
auto-incrementing integer (returned by `register_volume`). Each
`_VolumeResource` carries:

- `data: np.ndarray` — the source numpy array (kept for re-upload paths).
- `texture: wgpu.GPUTexture` — `r32float`, `texture_3d`, with WGPU
  size `(Nx, Ny, Nz)` derived from the numpy array's `(Nz, Ny, Nx)`
  shape (axis order reversed). Sample-only usage in v1
  (`TextureUsage.TEXTURE_BINDING | COPY_DST`). v2 will additionally OR
  in `STORAGE_BINDING`; format is already chosen to be compatible.
- `dirty: bool` — set by `update_volume`, cleared by the renderer after
  the next frame's upload.
- `name: str` — diagnostics.

Validation at registration time:
- `data.ndim != 3` → ValueError
- `data.dtype not in (float32,)` → ValueError ("convert to float32")
- `data.flags["C_CONTIGUOUS"] is False` → ValueError
- Any axis size `> device.limits.max_texture_dimension_3d` → ValueError
  (defer to wgpu's reported limit, typically 2048).

### Material baking

`VolumeMaterial.__init__` does the LUT bake:

- Color LUT: looked up from `manifoldx.viz.colormaps.LUT_TEXTURES[cmap]`
  (already a 1D RGBA8 texture; reused as-is, no copy).
- Opacity LUT: built into a `(256,)` float32 numpy array from
  `opacity_stops`, then uploaded to a fresh 1D R32F texture owned by
  the material instance (small — 1 KiB per material).
- `vmin >= vmax` → ValueError.
- `step_size is not None and step_size <= 0` → ValueError.

Both LUTs use the same sampler (linear filtering, clamp-to-edge).

### Render pass

New file: `src/manifoldx/render/volume_pass.py` (or new methods inside
`renderer.py` next to `_render_sprite_batches` etc., depending on whether
we split the renderer in this plan — see "Refactoring" below).

Pipeline state:
- Topology: triangle-list (3 verts, fullscreen quad covered by an
  oversized triangle to skip the diagonal seam — standard NDC
  fullscreen-triangle trick).
- Vertex shader: emits `gl_Position` from `vertex_index`, no per-volume
  vertex buffer.
- Fragment shader: per-fragment ray construction from inverse view +
  inverse projection, ray/box-AABB intersection in **world space** using
  `Transform.translation`, `Transform.rotation`, `Transform.scale`
  (encoded as a model matrix uniform), then a fixed-step march.
- Depth state: `depth_compare = LESS_EQUAL`, `depth_write_enabled = False`.
- Color state: alpha-blend `src.rgb * src.a + dst.rgb * (1 - src.a)`,
  `src.a + dst.a * (1 - src.a)`.
- Cull mode: `NONE` (we draw a fullscreen tri; cull is irrelevant).

Per-volume draw call (one per Volume entity per frame):
- Bind group 0: existing engine globals (camera matrices, viewport,
  time). No changes to the `Globals` struct.
- Bind group 1, layout (named `volume_bgl`):
  - 0: `volume_tex: texture_3d<f32>` (per-Volume registry entry).
  - 1: `vol_sampler: sampler` (shared engine-scoped sampler).
  - 2: `color_lut: texture_1d<f32>` (per-Material's chosen colormap
       LUT — already on GPU from the existing colormap atlas).
  - 3: `opacity_lut: texture_1d<f32>` (per-Material instance).
  - 4: `lut_sampler: sampler` (shared).
  - 5: `volume_uniforms: VolumeUniforms` (per-draw):
    ```wgsl
    struct VolumeUniforms {
        model_matrix:        mat4x4<f32>,  // entity Transform
        inv_model_matrix:    mat4x4<f32>,  // for world→local transform
        vmin:                f32,
        vmax:                f32,
        density_scale:       f32,
        step_size:           f32,
        max_steps:           u32,
        _pad0: u32, _pad1: u32, _pad2: u32,    // 16-byte align
    }
    ```

The fragment-shader main loop:
```wgsl
@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ndc = pixel_to_ndc(frag_pos.xy);
    let ray_origin_world = camera_pos();
    let ray_dir_world    = normalize(unproject(ndc) - ray_origin_world);

    // Box AABB in world space: take the 8 cube corners through model_matrix
    // and use slab-method against the resulting AABB. (The volume itself
    // is rotated, so we transform the ray into local space and intersect
    // with the unit cube there — cheaper than re-fitting the AABB.)
    let ro_local = (vu.inv_model_matrix * vec4(ray_origin_world, 1.0)).xyz;
    let rd_local =  (vu.inv_model_matrix * vec4(ray_dir_world,   0.0)).xyz;
    var t_near: f32; var t_far: f32;
    if (!ray_box_unit_cube(ro_local, rd_local, &t_near, &t_far)) {
        discard;
    }

    var t = max(t_near, 0.0);
    var accum: vec4<f32> = vec4<f32>(0.0);
    var step: u32 = 0u;
    loop {
        if (step >= vu.max_steps || t > t_far || accum.a > 0.99) { break; }
        let p_local = ro_local + t * rd_local;        // in [-0.5, 0.5]
        let p_uvw   = p_local + vec3<f32>(0.5);       // in [0, 1]
        let s       = textureSampleLevel(volume_tex, vol_sampler, p_uvw, 0.0).r;
        let s_n     = clamp((s - vu.vmin) / (vu.vmax - vu.vmin), 0.0, 1.0);
        let rgb     = textureSampleLevel(color_lut, lut_sampler, s_n, 0.0).rgb;
        let a       = textureSampleLevel(opacity_lut, lut_sampler, s_n, 0.0).r
                      * vu.density_scale * vu.step_size;
        accum = accum + (1.0 - accum.a) * vec4<f32>(rgb * a, a);
        t = t + vu.step_size;
        step = step + 1u;
    }
    return accum;
}
```

### Pipeline cache key

The renderer's `_get_or_create_pipeline` cache is currently keyed by
material kind. Add `"volume"` as a new kind. Volume pipelines never
collide with mesh / sprite / label / axis pipelines.

### Refactoring `renderer.py` (deferred)

`renderer.py` is currently 1532 lines, with `_render_mesh_batches`,
`_render_sprite_batches`, `_render_label_pass`, `_render_axis_pass`
all inlined. The eventual right shape is to split each pass into its
own module under `src/manifoldx/render/passes/` and reduce
`RenderPipeline.run` to a thin orchestrator. **This refactor is
deferred from v1 to a follow-up plan** so volumes ship without
churning every other render path. v1 simply adds
`_render_volume_pass` as a sibling of the existing four pass methods.

---

## Test plan

### Unit tests (no GPU)

- `tests/test_volume_registry.py`:
  - `register_volume` returns sequential ids starting at 0.
  - Non-3D array → ValueError.
  - Non-float32 dtype → ValueError with hint.
  - Non-C-contiguous → ValueError.
  - Axis size exceeding `max_texture_dimension_3d` → ValueError.
  - `update_volume` with a different shape → ValueError.
  - `update_volume` flips `dirty` to True.
  - `bind_compute_volume` raises `NotImplementedError("v2")`.

- `tests/test_volume_material.py`:
  - Default `opacity_stops=None` produces `linspace(0, 1, 256)`.
  - Stops `[(0,0),(1,1)]` produce `linspace(0, 1, 256)` (within rtol 1e-6).
  - Stops `[(0,0),(0.5,0),(0.5,1),(1,1)]` produce a step at index 128.
  - Pre-baked `(256,) float32` array used as-is.
  - Unknown `cmap` → ValueError listing available names.
  - `vmin >= vmax` → ValueError.
  - `step_size <= 0` → ValueError.
  - `max_steps <= 0` → ValueError.

### Integration tests (require a wgpu device)

- `tests/test_volume_render.py`:
  - **Empty-volume baseline**: spawn a 4³ all-zero volume; render at fixed
    camera; assert framebuffer alpha-channel is exactly 0 inside the box
    bounds (background passes through unmodified) and outside.
  - **Centered-blob smoke**: spawn a 32³ volume with a Gaussian centered
    at the origin; render at fixed camera (camera looking down −Z at
    `(0, 0, 5)`, framebuffer 64×64); assert center pixel `alpha > 0.5`,
    framebuffer is exactly background outside the projected box, and
    pixel values are reproducible byte-for-byte across two runs (no
    nondeterminism from RNG, time, or thread interleaving).
  - **vmin/vmax effect**: same volume rendered with `vmax=0.5` and
    `vmax=1.0` — 0.5 case has higher saturation in the central pixel.
  - **density_scale effect**: same volume with `density_scale=0.5` vs
    `density_scale=2.0` — center pixel alpha is monotonically higher
    in the latter.
  - **Multi-entity**: same `vol_id` referenced by two entities at
    `pos=(-2, 0, 0)` and `pos=(+2, 0, 0)` — exactly two visually
    separated regions of nonzero alpha.
  - **Re-upload**: `engine.update_volume(vol_id, new_array)` between
    frames; pixels at fixed camera differ from the previous frame.
  - **Step-size invariance**: same volume rendered at fixed camera
    with `step_size = auto` and with `step_size = auto / 2` produces
    nearly identical center-pixel rgba (within absolute tolerance
    ≈ 0.02 on alpha). This is the load-bearing property of the
    `alpha *= step_size` integration formula — without it the volume's
    visibility would silently drift with quality settings, which is
    the most common DVR-implementation bug.
  - **Pass-ordering preservation**: render a scene that contains a
    mesh, a sprite, a volume, a label, and an axis; assert the volume
    composites correctly behind the label and in front of opaque mesh
    (a yellow label on top of a red volume on top of a blue mesh
    produces the expected pixel order).

### Smoke render

- `examples/volume_demo.py`:
  - 64³ Gaussian volume.
  - Camera orbit, viridis colormap, opacity ramp `[(0, 0), (0.3, 0.05), (1, 0.6)]`.
  - `--render --duration 2 --fps 30 --quality low` produces a 60-frame
    video with no shader-validation errors and recognizable blob.
- Existing examples (`hello_world.py`, `cube.py`, `spheres.py`,
  `pbr_demo.py`, `axes_demo.py`, `nbody_compute.py`, `gas_compute.py`,
  `point_cloud_compute.py`) render unchanged at the pixel level — the
  volume pass is no-op when no Volume entities exist.

### CHANGELOG

A new "**v1 volume rendering**" entry under `[Unreleased]` listing the
exposed surface (`Volume`, `VolumeMaterial`, `engine.register_volume`,
`engine.update_volume`, `engine.bind_compute_volume` v2 stub).

---

## Out of scope (v1 → v2 / vN)

| Feature                                | Defer to | Reason |
|----------------------------------------|----------|--------|
| Compute-written volumes (`Writes[Volume3D]`) | v2 | Forward-compat hooks shipped in v1; needs transpiler extension. |
| Gradient lighting (Lambert / Phong)    | v2/v3 | Adds 6 sample fetches per step; v1 ships unlit per locked decision 4. |
| Isosurface rendering mode              | v2 | Different compositing branch; v1 is DVR-only per locked decision 1. |
| MIP rendering mode                     | v2 | Same. |
| Editable transfer-function widget      | vN | Needs a GUI framework; explicit non-goal. |
| Empty-space skipping (octree, blocks)  | vN | Performance optimization; v1 fixed-step is fast enough for ≤256³. |
| Streaming / out-of-core volumes (zarr, OpenVDB) | vN | Data-layer concern; orthogonal to the renderer. |
| Multi-channel volumes (RGB, vector fields) | vN | Different transfer-function shape (no longer scalar→RGBA). |
| Slicing / clip planes                  | vN | Useful, but a separate feature with its own API surface. |
| Picking ("what value at this voxel?")  | vN | Belongs to a broader picking story (also missing for points/meshes). |
| Pre-integrated transfer functions      | vN | Compositing improvement; v1 step-based is the baseline. |
| Jittered ray starts (anti-banding)     | vN | Users see banding only at very low `max_steps`; mitigated by raising it. |

---

## Architecture sketch

```
                          ┌──────────────────┐
   numpy 3D array  ─────► │ register_volume  │ ──► vol_id (int)
                          └──────────────────┘
                                   │
                                   ▼
                       ┌─────────────────────────┐
                       │  Engine._volume_registry │
                       │  vol_id → _VolumeResource│
                       │     ├─ texture_3d r32f   │
                       │     ├─ data (numpy)      │
                       │     └─ dirty bit         │
                       └────────────┬────────────┘
                                    │
                                    ▼
   ┌──────────────────────┐    ┌────────────┐    ┌────────────────────┐
   │ ECS entity:          │    │ Render     │    │ Fragment shader:   │
   │   Volume(volume_id)  │ ─► │ pass:      │ ─► │   ray/box,         │
   │   VolumeMaterial(...)│    │ volume     │    │   raymarch loop,   │
   │   Transform(scale,..)│    │ (5th pass) │    │   2× LUT sample    │
   └──────────────────────┘    └────────────┘    └────────────────────┘
                                    │
                                    ▼
                              composited frame
```

---

## Open questions (none blocking v1)

- Should `register_volume` accept `dtype=uint8` and quantize internally?
  Tabling for v2 — for v1 users convert with `data.astype(np.float32)`.
- Pre-multiplied vs. straight alpha in the opacity LUT? v1 uses straight
  alpha and pre-multiplies in the shader (`rgb * a`). Standard choice.
- World-space step size vs. screen-space adaptive? v1 is world-space
  (consistent quality regardless of zoom); adaptive would reduce cost
  when the volume is small on screen but adds shader logic. Punt.
