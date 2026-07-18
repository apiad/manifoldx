# Multi-Caster Shadows Design

Status: **spec / not yet implemented** (2026-07-18). Follow-up to the shipped
single-caster shadow subsystem (`.knowledge/analysis/2026-07-17-shadow-mapping-v1-design.md`).

## Goals

- Let **more than one light cast shadows at the same time** — concretely, a sun
  (directional) **and** a spot light (flashlight), each occluding only its own
  contribution.
- Every object then shows **two shadows**: a soft directional one from the sun
  and a sharp perspective one from the spot, crossing and overlapping as the
  flashlight moves. That is the target demo.
- Stay inside the existing architecture: reuse the depth-only shadow pass, the
  `group(2)` sampling seam, `compute_light_view_proj` / `compute_spot_light_view_proj`,
  and the slope-scaled-bias + PCF machinery. No new render-graph concepts.

## Non-goals (v1 of multi-caster)

- Arbitrary N casters with a general array (that is **Option B** below, deferred).
- Point-light (cubemap) shadows — orthogonal follow-up (roadmap item E).
- More than one sun or more than one spot. v1 is exactly **one sun + one spot**.
- Per-object shadow opt-out, translucent shadows, coloured shadows.

---

## Current state (single-caster) and why it blocks this

What exists today (all shipped):

- One depth map `RenderPipeline._shadow_map` (+ `_shadow_map_view`), one comparison
  sampler `_shadow_sampler`, bound as `@group(2)` on the `StandardMaterial` pipeline
  (`binding 0` = `texture_depth_2d`, `binding 1` = `sampler_comparison`).
- One `Globals.light_view_proj` (mat4 @240) drives both the depth pass VS and the
  fragment lookup.
- A single `Globals.shadow_caster` (u32 @404): `0` none, `1` sun, `2` spot — a
  **router**. The depth pass renders once from the active caster
  (`_caster_light_view_proj(engine, caster)`), and `shadowFactor()` applies that one
  map to whichever term `shadow_caster` names. The spot wins if both are set.
- `render/passes/shadow.py::render_shadow_map` runs the depth pass exactly once.

The blocker: one map, one matrix, one router. A sun map cannot occlude the sun
while a spot map occludes the spot in the same frame. We need **two maps + two
matrices + two independent lookups**.

---

## Option A — two dedicated maps (recommended v1)

Fix the caster set at **sun + spot**. This matches the demo exactly and is bounded.

### Architecture / data flow (per frame)

1. **Depth pass, sun** → renders all mesh geometry with `sun_light_view_proj`
   (ortho, auto-fit) into `_sun_shadow_map`.
2. **Depth pass, spot** → renders the same geometry with `spot_light_view_proj`
   (perspective) into `_spot_shadow_map`.
   Both encoded before the main render pass, same command encoder (like today's
   single pass).
3. **Main pass** → `StandardMaterial` fragment samples **both** maps:
   sun term × `sunShadowFactor()`, spot term × `spotShadowFactor()`, independently.

### Resources (`RenderPipeline._ensure_pipeline`)

- `_sun_shadow_map` / `_sun_shadow_map_view` and `_spot_shadow_map` /
  `_spot_shadow_map_view`, both `depth24plus`, `RENDER_ATTACHMENT | TEXTURE_BINDING`.
  Keep the existing `_shadow_map` name as the sun map to minimise churn, add the
  spot map alongside.
- Keep the single comparison sampler `_shadow_sampler` (shared).
- **Per-caster VP uniform buffers**: `_sun_vp_buf` and `_spot_vp_buf` (64 bytes each,
  a single mat4). Rationale below.
- Two `_BatchBuffers` for the depth passes are **not** needed — the pass uploads
  transforms once per frame and both sub-passes can read the same
  `_shadow_batch_buffers.transforms_buf` (transforms are caster-independent).

### The per-caster VP buffer (avoid a write-after-write hazard)

The depth-pass vertex shader currently reads `globals.light_view_proj`. With two
sub-passes we cannot write two different matrices into the *same* buffer slot in
one frame — `queue.write_buffer` is ordered before the command buffer, so the
second write clobbers the first and **both** sub-passes would read the spot matrix
(the same hazard called out for transforms in `render/passes/mesh.py`).

Fix: give the depth-pass VS its **own** small uniform (not the shared `Globals`).
Change `_SHADOW_SHADER`'s `@group(0) @binding(0)` from the full `Globals` to a
minimal `struct { vp: mat4x4<f32> }`, and bind `_sun_vp_buf` for sub-pass 1 and
`_spot_vp_buf` for sub-pass 2. Each buffer is written exactly once per frame. This
also shrinks the shadow shader (no need to mirror the whole 496-byte struct) and
generalises cleanly to N casters (one small VP buffer per caster).

### `Globals` layout change (416 → 496 bytes)

Current tail (416): `spot_*` block ends at 400, `spot_cos_outer` @400,
`shadow_caster` @404, pad @408/412. Append:

| offset | field | bytes |
|--------|-------|-------|
| 240 | `sun_light_view_proj` (mat4) — *was* `light_view_proj` | 64 |
| … | (existing sun/shadow/spot blocks unchanged, 304–416) | |
| 416 | `spot_light_view_proj` (mat4) | 64 |
| 480 | `sun_casts` (u32), `spot_casts` (u32), `_pad`, `_pad` | 16 |
| **496** | **total** (16-aligned: 496 = 16×31) | |

- Rename the existing `light_view_proj` → `sun_light_view_proj` (same offset 240;
  it already holds the sun's ortho matrix when the sun casts).
- Replace the `shadow_caster` **router** semantics with two explicit booleans
  `sun_casts` / `spot_casts` @480. (Keep the `shadow_caster` field as a now-unused
  pad, or reclaim it — cleaner to drop it and reflow the spot block; but a reflow
  touches more offsets, so leaving it as `_pad` is the low-risk choice.)
- `shadow_map_size` (@344) stays a single value if both maps share a resolution
  (recommended); add `spot_shadow_map_size` only if per-caster resolution is
  wanted (not in v1).
- `shadow_bias` / `shadow_pcf_radius` (@340/@348) are shared by both casters in v1.

Every `size=416` / `"size": 416` reference moves to **496**: `renderer._ensure_pipeline`
buffer create, `np.zeros(416)` in globals packing, `mesh.py` binding-0 size, and the
(now-removed-from-shadow-shader) `shadow.py` binding size. Update
`tests/test_globals_layout.py` (`GLOBALS_SIZE_BYTES`).

### `group(2)` bindings (add the second depth texture)

```wgsl
@group(2) @binding(0) var sun_shadow_map:  texture_depth_2d;
@group(2) @binding(1) var shadow_sampler:  sampler_comparison;
@group(2) @binding(2) var spot_shadow_map: texture_depth_2d;
```

Bind-group layout (`_shadow_bind_group_layout`) gains a third entry (depth texture
at binding 2). The mesh pass binds both real views when the respective caster is on,
else the `_shadow_placeholder_view` (1×1 depth) for the inactive one — so the layout
is always satisfied (same placeholder trick as today).

### Shadow pass (`render/passes/shadow.py`)

Generalise `render_shadow_map` to render **once per active caster**:

```
casters = []
if sun casts:  casters.append((sun_vp_buf,  sun_shadow_map_view,  sun_light_view_proj))
if spot casts: casters.append((spot_vp_buf, spot_shadow_map_view, spot_light_view_proj))
upload transforms once (shared)
for (vp_buf, target_view, vp_matrix) in casters:
    write vp_matrix -> vp_buf
    begin depth-only pass on target_view (clear depth=1)
    bind (vp_buf, transforms); draw every geometry (per-stride pipeline as today)
    end
```

The per-stride pipeline cache (`_shadow_pipelines`) and transform upload are
unchanged; only the bind group's binding-0 becomes the small VP buffer instead of
`_globals_buffer`, and we loop.

### Shader — two independent shadow factors

Split today's `shadowFactor(world_pos, N)` into two, each with its own matrix, map,
and light direction for the slope-scaled bias:

```wgsl
fn shadowLookup(vp: mat4x4<f32>, map: texture_depth_2d, world_pos: vec3<f32>,
                N: vec3<f32>, L: vec3<f32>) -> f32 {
    let lp = vp * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x*0.5+0.5, -ndc.y*0.5+0.5);
    if (out of [0,1] uv or ndc.z outside [0,1]) { return 1.0; }
    let ndl = max(dot(N, L), 0.0);
    let bias = globals.shadow_bias * (1.0 + 4.0*(1.0 - ndl));
    // PCF box (2r+1)^2 as today, using textureSampleCompareLevel
    ...
}
```

WGSL note: textures and samplers **may** be passed as function parameters, so a
single `shadowLookup(vp, map: texture_depth_2d, samp: sampler_comparison, ...)`
called twice (once per caster) is the clean, DRY path — confirm the installed
`wgpu-native` accepts texture params (it should; verify with a one-off shader at
implementation time). If a version balks, fall back to two explicit functions
`sunShadowFactor` / `spotShadowFactor` each hard-referencing its global texture, or
the `.replace()` shader-templating trick the codebase already uses for the textured
`StandardMaterial`.

`fs_main`:

```wgsl
if (globals.sun_intensity > 0.0) {
    var s = 1.0;
    if (globals.sun_casts != 0u) { s = sunShadowFactor(in.world_pos, N); }
    Lo += calculateSun(...) * s;
}
if (globals.spot_intensity > 0.0) {
    var s = 1.0;
    if (globals.spot_casts != 0u) { s = spotShadowFactor(in.world_pos, N); }
    Lo += calculateSpot(...) * s;
}
```

`_SHADOW_SHADER` no longer needs the full `Globals` struct (it reads only its VP
buffer), so it does **not** grow with this change.

### API

No new public surface needed. `set_sun`, `set_spot`, `enable_shadows` stay as-is.
The caster flags are derived each frame in the globals packing:

```
sun_casts  = 1 if (shadow_config and sun  is not None) else 0
spot_casts = 1 if (shadow_config and spot is not None) else 0
```

So: set a sun, set a spot, call `enable_shadows()` → **both cast**. Setting only one
degrades to today's single-shadow behaviour. Setting neither → no shadows.

(Optional future knob: `enable_shadows(sun_casts=..., spot_casts=...)` to force a
light to *not* cast even when present. Not needed for v1.)

### Edge cases

- Only sun set → `spot_casts=0`, spot map is the placeholder, spot term unshadowed
  (and `spot_intensity=0` so the term is skipped anyway).
- Only spot set → symmetric.
- Neither shadow-casting light → both placeholders, `shadowFactor` early-returns 1.0.
- Auto-fit still applies to the **sun** map only; the spot map's frustum comes from
  the spot's position/direction/outer_angle/range (perspective), as today.

### Testing (GPU-gated + numpy)

- `test_dual_casters_two_shadows`: a single object under a sun **and** a spot from a
  different direction. Assert **two distinct darkened regions** appear (two shadow
  blobs at different offsets), and that disabling the spot removes one while the
  other stays. Concretely: diff (shadows-off vs on) has ≥2 separated connected
  components of darkening, or the darkened-pixel count with both casters >
  either-alone.
- `test_sun_map_and_spot_map_allocated`: both `_sun_shadow_map` and
  `_spot_shadow_map` exist with the configured resolution when both lights + shadows
  are set.
- `test_globals_buffer_is_496_bytes`.
- `test_spot_only_still_single_shadow` / `test_sun_only_still_single_shadow`:
  regression — one-light scenes behave exactly as the shipped single-caster path.
- Pure-numpy: `sun_casts`/`spot_casts` derivation from engine state.

### Demo

`examples/dual_shadow_demo.py`: a fixed sun (raking, soft PCF shadow) **plus** an
orbiting flashlight (spot) over the teapot + a few objects on a horizontal floor.
Each object casts a soft sun shadow that stays put and a sharp spot shadow that
sweeps as the flashlight orbits — the two crossing is the whole point.

### Effort

~half day. Breakdown: second shadow map + placeholder wiring (~1 h), per-caster VP
buffer + shadow-pass loop (~1–1.5 h), `Globals` 416→496 + packing both matrices +
flags (~1 h), shader split into two factors + `fs_main` (~1 h), tests + demo (~1.5 h).

---

## Option B — general N-caster array (scalable follow-up)

When shadows are wanted from an arbitrary set of lights (several spots, point
lights, multiple suns), the two-fixed-slots approach doesn't scale. The general
form:

- **`texture_depth_2d_array`** with N layers (one shadow map per caster), a single
  bind. Cap N (e.g. 4–8) for a fixed layout.
- **Storage buffer of `light_view_proj`** (one mat4 per caster) + a small per-caster
  metadata record (light index, type = sun/spot/point, layer, bias). Lights and
  their shadow layers are matched by index.
- **Shadow pass loops over active casters**, rendering each into its array layer
  with its VP (one small VP buffer written per caster, or a dynamic-offset uniform).
- **Shader loops** over the lights: for each shadow-casting light, index its array
  layer via `textureSampleCompareLevel(array_map, sampler, uv, layer, depth)` and
  attenuate that light's term.
- Point lights need a **cube** array (`texture_depth_cube_array`) or a separate path
  (roadmap item E) — cubemaps don't fit the 2D array cleanly, so point-light shadows
  likely stay a parallel subsystem.

Trade-offs: more flexible, but heavier (variable-length shader loops, array
textures, per-light metadata plumbing) and slower to get right (~1 day+). Overkill
until a scene genuinely needs >2 shadow casters.

**Migration A→B:** Option A's per-caster VP buffers and split-factor shader are the
natural stepping stones — A's two explicit factors become one indexed loop, and its
two maps become two layers of the array. Doing A first is not wasted work.

---

## Decision

Build **Option A** (two dedicated maps: sun + spot) for the concrete "sun + one
flashlight" demo. Keep **Option B** (N-caster array) as the roadmap entry for when a
scene needs arbitrary casters. Point-light/cubemap shadows (item E) remain a
separate subsystem regardless of A or B.

## Open questions

- Shared vs per-caster `bias` / `pcf_radius`: v1 shares them. If the sharp spot
  shadow and the soft sun shadow want different PCF radii, promote `pcf_radius` to
  per-caster (cheap: two u32s in `Globals`).
- Should `shadow_caster` (@404) be reclaimed or left as pad? Left as pad in v1 to
  avoid reflowing offsets; reclaim during a later cleanup if the struct is touched.
- Resolution per map: shared in v1; a large scene under a tight flashlight might want
  a smaller spot map — trivial to split later.
