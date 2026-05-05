# Sci-Viz Primitives v1 — Particle / N-Body Visualization

**Date:** 2026-05-05
**Status:** Design approved, ready for implementation plan
**Scope:** First sub-project of the long-term "production-ready 3D scientific visualization engine" effort.

---

## 1. Context and framing

ManifoldX is positioned in its README as a real-time 3D rendering engine for scientific simulation, targeting 10⁴–10⁶ entities, with secondary applicability to interactive visualizations / serious games. The core ECS + instanced rendering + PBR pipeline is in place and shipped through `v0.3.0`. What is missing — and what gates real adoption by researchers — is a layer of **scientific visualization primitives**: domain-aware constructs (point clouds, glyphs, fields, axes, colormaps, legends) that turn raw entity rendering into an actual viz tool.

This document specifies **v1** of that layer, scoped narrowly to **particle / N-body visualization**. Particle-based simulations are the lowest-friction first target because three of the existing examples (`nbody.py`, `gas.py`, `boids.py`) already gesture at this use case using the generic `Mesh + Material + Transform` triad. v1 replaces that with a proper `PointCloud` primitive that scales, looks right, and exposes scalar attributes through colormaps.

The broader sci-viz primitives roadmap (vector / flow fields, volumetric data, surface scalar overlays, network graphs, glyph fields) is captured in §10 as an appendix. v1 deliberately ships the smallest coherent slice that yields visible value to a researcher.

---

## 2. Design principles (locked from brainstorming)

These constraints are load-bearing — they shaped every choice in this document.

1. **Engine stays small.** No `engine.point_cloud(...)` method. `Engine` does not grow into a god-object. Sci-viz primitives live in their own module (`manifoldx.viz`) and integrate via the existing ECS API (`engine.spawn(...)`).
2. **Hybrid API.** ECS components + materials are the canonical primitive. A separate functional shim layer (`viz.point_cloud(engine, ...)`) provides ergonomic access for users who do not need ECS literacy. The shim creates ECS entities under the hood; there is one rendering path.
3. **Single representation, scale-first.** v1 ships **only camera-facing point sprites** (sphere imposters), not instanced sphere meshes. The README claims 10⁴–10⁶ entities; instanced meshes do not meet that bar. One representation = one shader = one performance profile.
4. **GPU-side colormapping.** Scalar attributes upload as a storage buffer; fragment shader samples a 1D LUT texture. CPU-side colormap is rejected (bandwidth waste, no per-pixel interpolation).
5. **Text rendering ships in v1**, but at the simplest level that works: PIL → RGBA texture → camera-facing billboard quad. SDF fonts and rich text are explicit no-goals.
6. **No new mutation machinery.** Per-particle attributes (color value, radius) are normal ECS components updated through the existing `_FieldView` deferred-mutation path.
7. **The roadmap is preserved.** This document includes a roadmap appendix (§10) so that v1 decisions are made with the long-term arc in view.

---

## 3. User-facing surface

Two layered APIs targeting two user personas.

### 3.1 Canonical (ECS) — for users already inside the engine

```python
import numpy as np
import manifoldx as mx
from manifoldx.viz import PointCloud, ColormapMaterial, ScalarValue, Radius

engine = mx.Engine("N-body")

N = 5000
positions = np.random.randn(N, 3).astype(np.float32) * 10
masses = np.random.exponential(1.0, N).astype(np.float32)

cmap_mat = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=5.0, lit=False)

engine.spawn(
    PointCloud(),
    cmap_mat,
    mx.components.Transform(pos=positions),
    ScalarValue(value=masses),
    Radius(radius=0.05 * np.cbrt(masses)),
    n=N,
)

@engine.system
def gravity(query: mx.Query[mx.components.Transform, ScalarValue], dt):
    pos = query[mx.components.Transform].position.data
    m = query[ScalarValue].value.data
    forces = compute_pairwise_gravity(pos, m)  # user code
    query[mx.components.Transform].position += forces * dt

engine.cli()
```

The user composes the same way they already do — `engine.spawn(...)` plus regular components. `PointCloud` is a marker component telling the renderer to use the sprite path. `ScalarValue` and `Radius` are independent components, so they can be reused on non-sprite entities later (e.g. glyph fields in v2).

### 3.2 Functional shim — for users who don't want ECS upfront

```python
from manifoldx.viz import point_cloud, axes, scale_bar, colormap_legend

cloud = point_cloud(
    engine, positions,
    color=masses, size=0.05 * np.cbrt(masses),
    cmap="viridis", vmin=0.0, vmax=5.0,
)

axes(engine, extent=10.0, label_unit="AU")
scale_bar(engine, length=1.0, label="1 AU")
colormap_legend(engine, cloud.material, position="right", ticks=5)

@engine.system
def gravity(query, dt):
    cloud.positions += cloud.velocities * dt   # handle exposes _FieldView slices
```

`point_cloud` returns a `PointCloudHandle` — a thin dataclass holding the entity index range plus `_FieldView`-backed accessors (`.positions`, `.values`, `.radii`, `.velocities` if a `Velocity` component was added). Users who outgrow the shim can reach into the handle's entity range and use the canonical API alongside.

`Engine` is unchanged. None of these helpers live as methods on `Engine`.

---

## 4. Module layout

```
src/manifoldx/viz/
├── __init__.py           # Public re-exports
├── components.py         # PointCloud, ScalarValue, Radius, TextLabel, AxisFrame, ScaleBar
├── materials.py          # ColormapMaterial, LabelMaterial, AxisMaterial
├── colormaps.py          # 1D LUT data (viridis, magma, plasma, inferno, turbo, gray)
├── geometry.py           # SPRITE_QUAD, AXIS_LINES, SCALE_BAR_LINE built-in geometries
├── text.py               # PIL rasterizer + LabelTextureAtlas
└── shims.py              # point_cloud, axes, scale_bar, colormap_legend
```

`src/manifoldx/__init__.py` re-exports `viz` so `import manifoldx as mx; mx.viz.point_cloud(...)` works.

`pyproject.toml` gains a `[viz]` optional dependency group containing `pillow>=10.0` (text rasterization). `[all]` includes it.

---

## 5. ECS components added in v1

| Component | Storage shape | Field semantics |
|---|---|---|
| `PointCloud` | empty (marker, 0 floats) | Tag entity for the sprite render path. Resolves implicit geometry to `SPRITE_QUAD`. |
| `ScalarValue` | 1 float per entity | Per-particle scalar attribute; mapped through `ColormapMaterial`'s LUT. |
| `Radius` | 1 float per entity | Per-particle world-space radius; sprite quad scales to cover this from any view angle. |
| `TextLabel` | 1 uint32 per entity | Index into the global `LabelTextureAtlas`. `Transform.position` anchors the label in 3D space. |
| `AxisFrame` | 2 floats (extent, thickness) | Tag entity for axis line rendering with `AxisMaterial`. |
| `ScaleBar` | 2 floats (length, label_id) | Tag entity for scale bar; label_id indexes the same `LabelTextureAtlas` used by `TextLabel`. |

All components register through the existing `EntityStore._components` mechanism. `register()` and `get_data()` follow the conventions in `manifoldx.components`.

The marker pattern (`PointCloud` carries no data) is chosen over fat single-component (`PointCloud(value=, radius=)`) because:
- `ScalarValue` and `Radius` are independently useful — future glyph and surface materials will read the same `ScalarValue`.
- Composability with the ECS `Query[…]` story is preserved (a system can read `ScalarValue` without caring whether the entity is a sprite, a glyph, or a mesh vertex).
- `_FieldView` mutation works on each component independently without sub-field carving.

---

## 6. Materials

Three new materials, all registered through the existing `MaterialRegistry`.

### 6.1 `ColormapMaterial(cmap, vmin, vmax, lit=False)`

Per-batch uniform: `vmin`, `vmax`, `lit_flag`, padding. Per-batch texture binding: 1D RGBA8 colormap LUT (256 texels). Per-instance storage buffers (read in vertex/fragment shaders by `instance_index`):
- `transforms` (existing) — mat4×4 model matrix
- `scalar_values` — 1 float per instance
- `radii` — 1 float per instance

Vertex shader expands the unit quad into a camera-facing billboard scaled by per-instance radius and projected such that the resulting fragment disk has world-space radius equal to `Radius`. Fragment shader:
- Discards fragments outside the unit disk in the quad's local UV space.
- Reconstructs the sphere normal from quad UV (`n.xy = uv * 2 - 1; n.z = sqrt(1 - dot(n.xy, n.xy))`).
- If `lit_flag`, computes Lambert against a fixed view-space light direction `(0.5, 0.5, 1.0)` normalized. v1 does not consult the engine's `PointLight` list — sprite lighting is a sci-viz convenience feature, not photoreal; integration with the scene's `PointLight` set is a roadmap item.
- Samples the LUT texture at `(scalar_value - vmin) / (vmax - vmin)` (clamped to [0, 1]).
- Outputs the LUT color, optionally multiplied by Lambert.

`lit=False` is the sci-viz default — researchers want raw scalar mapping, not lit appearance.

### 6.2 `LabelMaterial`

Used by `TextLabel`, `AxisFrame` tick labels, `ScaleBar` caption, and `colormap_legend` annotations. Per-batch uniform: anchor mode (world / screen), depth-test override (always off in v1). Per-instance storage buffer: label texture atlas slice index. Texture binding: a 2D texture array holding all rasterized labels (each slice = one label's RGBA8 texture, fixed max size 256×64 for v1).

Vertex shader emits a camera-facing quad sized to the texture's pixel dimensions (mapped through a screen-space scale uniform so labels do not shrink with distance — v1 default is "labels stay readable at any zoom"). Fragment shader samples the atlas slice with alpha; alpha-blended over the scene.

### 6.3 `AxisMaterial`

Plain unlit colored line material. Per-batch uniform: 4 floats (RGB color + alpha). No per-instance variation; each axis (X/Y/Z) is its own batch with its own color (X = `#e64545`, Y = `#5fbf5f`, Z = `#5588ff`, defaults overridable via factory args). Geometry is `LineList` topology (two vertices per axis line); WebGPU `LineList` is the chosen topology.

---

## 7. Renderer changes

`renderer.py` and the `RenderPipeline.run` pipeline today assume one path: instanced meshes. v1 splits this into three.

### 7.1 Batch construction

`RenderPipeline.run` currently groups entities by `(geom_id, material_type)`. v1 extends this:

- **Mesh batch list** — entities with `Mesh` component, batched as today.
- **Sprite batch list** — entities with `PointCloud` component, batched by `(material_id)` only (geometry is implicit `SPRITE_QUAD`).
- **Label batch list** — entities with `TextLabel`, `AxisFrame`'s tick-label children, `ScaleBar`'s caption child, all batched into a single batch keyed by the global `LabelTextureAtlas`.
- **Axis batch list** — entities with `AxisFrame`'s line children, batched per axis.

Per-instance storage buffers are computed once per render pass:
- `transforms` — always, all batches.
- `scalar_values`, `radii` — only for sprite batches whose material is `ColormapMaterial`. Allocated lazily; populated by NumPy slicing from the relevant ECS component arrays (no Python loops).

### 7.2 Render order

Single render pass, three sub-pass-style draw groups in order:

1. **3D opaque pass** — depth-write on, depth-test on, blend off:
   - Mesh batches (existing).
   - Sprite batches.
   - Axis batches.
2. **Label pass** — depth-write off, depth-test off, alpha-blend on:
   - Label batches.

Order matters: labels must draw after axes/sprites so they remain visible regardless of z-order.

### 7.3 Pipeline cache

The pipeline cache key is extended from `(material_type)` to `(material_type, material_subtype)`, where `material_subtype` is the LUT identity for `ColormapMaterial` (so `cmap="viridis"` vs `cmap="magma"` share a pipeline but rebind the LUT texture). For `LabelMaterial` and `AxisMaterial`, `material_subtype` is `None`.

The current renderer line at `resources.py:488`:
```python
first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
```
is removed for `ColormapMaterial`'s scalar/radius binding path; uniform binding still uses the first-row fallback for `vmin`/`vmax` since those are per-batch, not per-instance.

### 7.4 File-level impact

- `renderer.py`: ~200 lines added/refactored. Three sub-functions (`_render_mesh_batches`, `_render_sprite_batches`, `_render_axis_batches`, `_render_label_pass`) replace the single render loop. Storage-buffer allocation lifted into a `_BatchBuffers` helper that holds per-batch arrays and resizes lazily.
- `resources.py`: ~50 lines added. Register `SPRITE_QUAD`, `AXIS_LINE`, `SCALE_BAR_LINE` as built-in geometries.

---

## 8. Text rendering implementation

`viz/text.py` provides:

```python
class LabelTextureAtlas:
    """
    Holds a 2D texture array of rasterized label strings.
    label_id -> array slice. PIL renders each unique string once,
    cached forever (no eviction in v1 — v1 expects ≤256 unique labels).
    """
    def get_or_create(self, text: str, font_size: int = 14) -> int: ...
    def upload_dirty(self, device, queue): ...
```

Implementation:
- `PIL.ImageFont.truetype` with a bundled DejaVu Sans Mono (TTF shipped in the package, ~300KB).
- Each label rendered to a fixed 256×64 RGBA8 image (text left-aligned, transparent background, white pixels).
- Atlas is a `wgpu.TextureViewDimension.D2_ARRAY` with up to 256 slices (v1 cap).
- Re-upload only when new labels added (dirty-flag tracked).

`functional shim → axes(engine, ...)` creates one `AxisFrame` line entity per axis + N `TextLabel` entities for the ticks, each calling `LabelTextureAtlas.get_or_create("0.5 AU")` etc.

No-goals (explicit): SDF fonts, kerning, RTL languages, custom user fonts, dynamic font size at render time, tex glyph caching with eviction. v1 caps at 256 unique labels per scene; exceeding this raises a clear error directing the user to file an issue requesting the proper text layer.

---

## 9. Functional API (`viz/shims.py`)

### 9.1 `point_cloud`

```python
def point_cloud(
    engine: Engine,
    positions: np.ndarray,            # (N, 3)
    *,
    color: np.ndarray | None = None,  # (N,) scalar values
    size: np.ndarray | float = 0.1,   # (N,) or scalar radius
    cmap: str = "viridis",
    vmin: float | None = None,        # auto from color.min() if None
    vmax: float | None = None,        # auto from color.max() if None
    lit: bool = False,
) -> PointCloudHandle: ...
```

Validates shapes, broadcasts scalar `size`, auto-computes `vmin`/`vmax` if not given, creates a `ColormapMaterial`, calls `engine.spawn(...)` once for `N` entities, returns:

```python
@dataclass
class PointCloudHandle:
    engine: Engine
    entity_range: tuple[int, int]   # [start, stop) in entity store
    material: ColormapMaterial

    @property
    def positions(self) -> _FieldView: ...   # writable view into Transform.pos
    @property
    def values(self) -> _FieldView: ...      # writable view into ScalarValue.value
    @property
    def radii(self) -> _FieldView: ...       # writable view into Radius.radius
```

Field views support all existing `_FieldView` operators (`+=`, `*=`, indexing). Mutation queues `UPDATE_COMPONENT` commands per the existing deferred-mutation path. **No new mutation infrastructure.**

### 9.2 `axes`

```python
def axes(
    engine: Engine,
    *,
    extent: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    labels: bool = True,
    label_unit: str = "",
    n_ticks: int = 5,
) -> AxesHandle: ...
```

Spawns three `AxisFrame` line entities + (if `labels`) `n_ticks` × 3 `TextLabel` entities.

### 9.3 `scale_bar`

```python
def scale_bar(
    engine: Engine,
    *,
    length: float = 1.0,
    label: str = "",
    position: Literal["bottom-left", "bottom-right", "top-left", "top-right"] = "bottom-left",
) -> ScaleBarHandle: ...
```

Anchored in screen space via the same screen-space-anchor mechanism as labels (label material's anchor mode set to `screen`).

### 9.4 `colormap_legend`

```python
def colormap_legend(
    engine: Engine,
    material: ColormapMaterial,
    *,
    position: Literal["right", "left", "top", "bottom"] = "right",
    ticks: int = 5,
    title: str = "",
) -> LegendHandle: ...
```

Spawns a thin screen-space rectangle textured with the material's LUT plus `ticks` `TextLabel` entities for the value annotations.

---

## 10. Roadmap (preserved across sub-projects)

The following items were identified during the codebase audit and the brainstorming for this sub-project. They are **out of scope for v1** but recorded here so the v1 design choices stay coherent with the long-term arc.

### 10.1 Sci-viz primitives roadmap

| Item | Depends on | Notes |
|---|---|---|
| **Vector / flow fields** | v1 + line-rendering primitive | Arrow glyphs, streamlines, slicing planes. Streamlines need integrator; arrows need oriented mesh. |
| **Volumetric scalar fields** | Raycasting compute pipeline | 3D texture upload, ray-marching fragment shader, transfer functions. Major sub-project. |
| **Mesh surfaces with scalar overlays** | v1 colormap material | Mostly additive — `ColormapMaterial` already exists, surface meshes need per-vertex attribute path. |
| **Networks / graphs in 3D** | v1 + line-rendering + label improvements | Nodes already work as point clouds; edges need line rendering. |
| **Glyph fields (arrows, ellipsoids, tensors)** | Per-instance mesh selection | Today's batching assumes one geometry per batch. Glyph fields need per-instance geometry index. Moderate refactor. |

### 10.2 Cross-cutting engine items (from audit)

| Item | Why it matters |
|---|---|
| Real backend abstraction (Jupyter / Streamlit / web) | Shipping path — researchers don't run desktop GLFW, they live in notebooks. |
| Picking + interaction layer (mouse → entity, hover, drag) | Non-interactive viz is non-viz. |
| Texture / UV support in materials | Needed for any "real-world" mesh; also unlocks proper text rendering. |
| Frustum + occlusion culling | Required for the >100k regime to hit framerate. |
| Compute-shader simulation path | GPU-side physics for >1M entities. |
| Phong material — actually distinct shader | Currently inherits BasicMaterial's flat shader (latent bug). |
| Dynamic light count | Hardcoded 4 lights; researchers will hit this. |
| Per-instance material variants | Currently all instances share one material; needed for some glyph use cases. |
| Anti-aliased / thick line rendering | Default `LineList` is 1px; axes look thin. |

### 10.3 Tooling / DX items

| Item | Why it matters |
|---|---|
| Entity inspector / scene debugger | Hard to debug an ECS without one. |
| Frame profiler | Required to validate the >100k claim. |
| Shader hot-reload | DX win for material development. |
| Custom user colormaps via `register_colormap` | Sci-viz users have domain-specific palettes. |
| `AGENTS.md` + `know-how/` adoption | Bring the repo into the workspace know-how convention. |

These items are tracked here, not promoted into immediate plan items. Each can become its own brainstorm → spec → plan cycle.

---

## 11. Testing strategy

### 11.1 Unit tests

- `tests/viz/test_colormaps.py` — viridis/magma/plasma/inferno/turbo/gray sample known values at LUT indices 0, 64, 128, 192, 255. Verify `vmin`/`vmax` clamping math (CPU-side check against `np.clip`).
- `tests/viz/test_components.py` — register / spawn / get_data for `PointCloud`, `ScalarValue`, `Radius`, `TextLabel`, `AxisFrame`, `ScaleBar`. Verify `_FieldView` works on `ScalarValue.value` and `Radius.radius`.
- `tests/viz/test_text_atlas.py` — render fixed string "5.0 AU" at fixed font size, hash compare against committed golden RGBA. Verify `get_or_create` is idempotent (same string → same slice). Verify atlas overflow at 256 labels raises a clear error.
- `tests/viz/test_geometry.py` — verify `SPRITE_QUAD` vertex count = 4 (or 6 indices), `AXIS_LINE` topology = `LineList`.
- `tests/viz/test_materials.py` — material uniform packing for `ColormapMaterial`, `LabelMaterial`, `AxisMaterial`. Verify pipeline cache key includes `material_subtype` for `ColormapMaterial` and not for the others.

### 11.2 Integration tests

- `tests/viz/test_point_cloud_spawn.py` — spawn 1k / 10k / 100k point cloud, assert no errors, assert entity count stable for 100 frames.
- `tests/viz/test_per_frame_update.py` — point cloud with a system that mutates `ScalarValue.value`; verify GPU buffer reflects the new values after one frame (read back via offscreen render).
- `tests/viz/test_render_modes.py` — same scene rendered through GLFW canvas and offscreen canvas; assert visual parity (golden image hash).

### 11.3 Visual regression

- Fixed-seed N-body 60-frame sequence rendered to PNG sequence; per-frame hash compared against committed golden frames under `tests/viz/golden/nbody_seed42/frame_NN.png.sha256`. One golden set per backend (desktop GLFW + offscreen).
- Triggered on PRs touching `viz/`, `renderer.py`, `resources.py`. Failure shows the diff against golden in CI output.

### 11.4 Performance smoke

- `tests/viz/test_perf_smoke.py` — 100k sprite cloud must achieve ≥30 FPS averaged over 5 seconds on the dev machine. Recorded in `.knowledge/log/YYYY-MM-DD.yaml` per release. Not a CI gate (machine-dependent); runs locally before tagging.

---

## 12. Explicit no-goals (v1)

- ❌ SDF / hinted / kerned text. PIL bitmap is the bar.
- ❌ Custom user-supplied colormaps. Hardcoded set: viridis, magma, plasma, inferno, turbo, gray.
- ❌ Logarithmic / categorical color scales. Linear only.
- ❌ Picking / hover tooltips. Separate sub-project (interaction layer).
- ❌ Frustum / occlusion culling for sprites. All sprites drawn every frame.
- ❌ Anti-aliased axes / thick lines. `LineList` (1px) only.
- ❌ Per-particle independent material variation. Whole `PointCloud` shares one material.
- ❌ Animation / timeline scrubber. Existing `--render` flag handles playback.
- ❌ Instanced sphere mesh path for particles. Sprites only — instanced spheres remain available via the generic `Mesh + Material` path for non-particle use cases.
- ❌ Lit point-cloud as default. `lit=False` is the default; opt-in via `lit=True`.

---

## 13. File change inventory

### 13.1 New files (~1500 lines estimated)

| File | Estimated lines | Responsibility |
|---|---|---|
| `src/manifoldx/viz/__init__.py` | 30 | Public re-exports. |
| `src/manifoldx/viz/components.py` | 150 | Six ECS components. |
| `src/manifoldx/viz/materials.py` | 400 | Three materials with WGSL shader code. |
| `src/manifoldx/viz/colormaps.py` | 150 | Six LUT arrays + helpers. |
| `src/manifoldx/viz/geometry.py` | 100 | Sprite quad, axis line, scale-bar line built-ins. |
| `src/manifoldx/viz/text.py` | 200 | PIL rasterizer + `LabelTextureAtlas`. |
| `src/manifoldx/viz/shims.py` | 150 | `point_cloud`, `axes`, `scale_bar`, `colormap_legend`. |
| `tests/viz/test_*.py` | 500 | Unit + integration + visual regression. |
| `examples/nbody_v2.py` | 50 | Migrated `nbody.py` using `point_cloud` + `axes` + `scale_bar`. |
| `src/manifoldx/viz/assets/DejaVuSansMono.ttf` | binary | ~300KB bundled font. |

### 13.2 Modified files

| File | Change |
|---|---|
| `src/manifoldx/renderer.py` | Split into mesh / sprite / axis / label render paths. ~200 lines added/refactored. |
| `src/manifoldx/resources.py` | Register `SPRITE_QUAD`, `AXIS_LINE`, `SCALE_BAR_LINE` built-ins. ~50 lines. |
| `src/manifoldx/__init__.py` | Re-export `viz`. ~5 lines. |
| `pyproject.toml` | Add `[viz]` extra with `pillow>=10.0`. ~3 lines. |
| `README.md` | Add a "Scientific Visualization" section showing the `point_cloud` API. ~50 lines. |
| `CHANGELOG.md` | Backfill `[0.3.0]` entry (currently missing); add `[0.4.0]` for this sub-project. |

### 13.3 Untouched

`ecs.py`, `engine.py`, `commands.py`, `components.py`, `systems.py`, `camera.py`, `types.py`, `backends.py`. The whole point of the design is that the engine core does not change.

---

## 14. Open questions explicitly resolved

These came up in brainstorming and were resolved here:

1. **`PointCloud` as a marker, not a fat component.** Resolved: marker, with `ScalarValue` and `Radius` as separate components for composability.
2. **Hardcoded colormap set vs `register_colormap` from day one.** Resolved: hardcoded in v1; `register_colormap` is a roadmap item (§10.3).
3. **Label storage: atlas vs per-label texture.** Resolved: atlas (texture array, up to 256 slices). Reason: axis labels can hit 30+ quickly per scene; atlas batches into a single draw call.
4. **Sprite imposter shading default: lit or unlit.** Resolved: unlit (sci-viz default); `lit=True` opt-in.
5. **Backend abstraction.** Resolved: out of scope. Existing `rendercanvas` + lazy-import path is good enough for v1; real abstraction is a separate sub-project.

---

## 15. Acceptance criteria

v1 is done when:

- [ ] `from manifoldx.viz import PointCloud, ColormapMaterial, ScalarValue, Radius` works and `engine.spawn(PointCloud(), ColormapMaterial(...), ScalarValue(...), Radius(...), Transform(...), n=N)` produces a working point cloud with colormapped scalars.
- [ ] `from manifoldx.viz import point_cloud, axes, scale_bar, colormap_legend` works; the functional shim creates entities and returns handles supporting `_FieldView` mutation.
- [ ] Six colormaps (viridis, magma, plasma, inferno, turbo, gray) render correctly against committed golden swatches.
- [ ] Text labels render with PIL → texture → billboard; axes show numeric tick labels with units; scale bar shows length caption; colormap legend shows tick values.
- [ ] N-body example (`examples/nbody_v2.py`) demonstrates 5000-body sim with `point_cloud` API, axes, scale bar, colormap legend.
- [ ] 100k point cloud renders at ≥30 FPS on the dev machine.
- [ ] Visual regression suite passes against committed golden frames.
- [ ] CHANGELOG entry written for `[0.4.0]`.
- [ ] README has a "Scientific Visualization" section.

---

## 16. Next step

Hand off to `superpowers:writing-plans` to produce the implementation plan that decomposes this design into sequenced, verifiable tasks.
