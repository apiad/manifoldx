# Shadow Mapping v1 — Directional Hard Shadows Design

## Goals

- The single biggest missing visual cue: geometry casting shadows. The engine has directional / point / spot lights but nothing occludes light.
- Ship the thinnest coherent slice that already looks great: **one directional light**, **hard shadow map**, sampled in the `StandardMaterial` PBR path.
- Deterministic and testable — a fixed, configurable orthographic frustum, so behaviour is reproducible headless.
- Fit the existing "one module-level function per render pass" architecture; add a new pass without disturbing the others.

## Non-goals (v1)

- Point-light (cubemap) and spot-light shadows. Directional only.
- PCF soft shadows with a larger kernel — VS2 follow-up (a localized shader change).
- Auto-fitting the light frustum to the scene AABB — follow-up. v1 uses a configurable fixed extent.
- Shadows for sprite / label / volume / axis / GUI paths, or for non-`StandardMaterial` materials.
- Cascaded shadow maps, contact-hardening, translucent shadows.

---

## Prerequisite discovered during planning: directional light is non-functional

`StandardMaterial`'s fragment shader only iterates a `PointLightData` array — there is **no directional light term**. The `DirectionalLight` class exists but `renderer._render_scene_passes` packs every light (via `get_data()`) into the same `array<PointLightData, 4>`, so a `DirectionalLight` is silently interpreted as a *point light at position = its direction vector*. No example uses `DirectionalLight`; all use `PointLight`. There is nothing to shadow.

Therefore VS1 must first **introduce a real directional "sun"** into `StandardMaterial`, then shadow it. This gives `DirectionalLight` its first correct consumer. The sun is engine-level state (like IBL), separate from the point-light array.

## Architecture

Shadow + sun state is engine-level (like lights and IBL). The system adds a directional term to the PBR shader and one pass **before** the existing main render pass:

1. **Sun as engine state** — `engine.set_sun(DirectionalLight(...))` stores a single directional sun. Its `get_data()` (32 bytes: `direction`+pad, `color`+`intensity`) is packed into `Globals`. `StandardMaterial` gains a `calculateSun(...)` term (Lambert + GGX, no distance attenuation), gated on `sun_intensity > 0`. This works with or without shadows.
2. **`src/manifoldx/render/passes/shadow.py`** — new depth-only pass. Renders all mesh-pass geometry from the sun's POV into an offscreen depth texture (the shadow map). No color attachment, minimal vertex-only pipeline (transforms by `light_view_proj`).
3. **Light-space matrix** — CPU-side numpy in a new `manifoldx.shadow` module: `light_view_proj = ortho(extent, near, far) @ look_at(eye, target, up)`, where `eye = target - normalize(direction) * back_distance`. Pure Python, unit-testable without a GPU.
4. **`Globals` uniform extension** — add `light_view_proj: mat4x4<f32>` (64B), `sun_direction: vec3<f32>`+pad, `sun_color: vec3<f32>`+`sun_intensity: f32`, `shadow_enabled: u32`, `shadow_bias: f32`, `shadow_map_size: f32`, +4B pad. Grows the struct 240 → 352 bytes (16-aligned: 352 = 16×22).
5. **Mesh pass bind group 2** — the shadow map (`texture_depth_2d`) + a comparison sampler, with a 1×1 placeholder bound when shadows are off. (Group 0 = main/transforms/material/lights, group 1 = IBL, group 2 = shadow.)
6. **`StandardMaterial` WGSL** — project `world_pos` into light space, sample the shadow map with `textureSampleCompareLevel` (hardware 2×2 PCF, near-free), and attenuate **only the sun term** by the resulting shadow factor. Point lights and IBL are unshadowed.

Per-frame order becomes: **shadow pass → [main pass: mesh → skybox → volume → label → axis]**.

New `Globals` byte layout (appended after the existing 240 bytes):

| offset | field | bytes |
|--------|-------|-------|
| 240 | `light_view_proj` (mat4x4) | 64 |
| 304 | `sun_direction` (vec3) + pad | 16 |
| 320 | `sun_color` (vec3) + `sun_intensity` (f32) | 16 |
| 336 | `shadow_enabled` (u32), `shadow_bias` (f32), `shadow_map_size` (f32), pad | 16 |
| **352** | **total** | |

Offsets 304 and 320 exactly match `DirectionalLight.get_data()`'s 32-byte block, so the sun packs in with one `memcpy`.

---

## What casts and what receives

- **Casts:** the single directional sun set via `engine.set_sun(...)`. If no sun is set, the shadow pass is skipped and `shadow_enabled = 0`. All mesh-pass entities are rasterized into the shadow map.
- **Receives:** only `StandardMaterial`. Other materials are untouched in v1.
- The shadow factor multiplies the **sun term only**. IBL ambient/specular and point lights are **not** shadowed in v1 (ambient light does not cast a directional shadow — correct by construction).

---

## API

Set the sun (directional lighting — works standalone, even without shadows):

```python
engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -1.0, -0.3)))
```

Then make it cast (engine-level shadow rig):

```python
engine.enable_shadows(
    target=(0, 0, 0),   # center of the ortho box the sun looks at
    extent=10.0,        # half-width of the orthographic box (world units)
    resolution=2048,    # shadow map is resolution x resolution
    near=0.1,
    far=50.0,
    bias=0.005,         # constant depth bias to kill shadow acne
)
```

`back_distance` (how far along `-direction` the virtual eye sits) defaults to `far * 0.5`; not user-facing in v1. Calling `enable_shadows()` allocates the shadow map + comparison sampler lazily on first frame. Without a sun **or** without `enable_shadows()`, `shadow_enabled = 0`; the sun term still lights the scene once `set_sun` is called.

---

## Data & shader details

- **Globals (352B):** existing fields unchanged; appended `light_view_proj` (mat4), `sun_direction`+pad, `sun_color`+`sun_intensity`, `shadow_enabled`, `shadow_bias`, `shadow_map_size`, pad (see the layout table above). The renderer writes `light_view_proj` each frame from the sun's direction + the configured rig, and packs the sun's `get_data()` at offset 304.
- **Shadow map:** `depth24plus`, `resolution²`, usage `RENDER_ATTACHMENT | TEXTURE_BINDING`, viewed as `texture_depth_2d`. Comparison sampler with `compare = less`.
- **Shadow factor (fragment):**
  1. `let lp = globals.light_view_proj * vec4(world_pos, 1.0);`
  2. perspective divide → NDC → `[0,1]` UV (flip Y), depth `lp.z` (wgpu NDC z ∈ `[0,1]`).
  3. `let shadow = textureSampleCompareLevel(shadow_map, shadow_sampler, uv, current_depth - bias);`
  4. fragments outside `[0,1]` UV or beyond far → fully lit (`shadow = 1.0`), so nothing outside the frustum goes black.
- Acne handled by the constant `shadow_bias`; peter-panning is acceptable at v1's bias magnitude. (Slope-scaled bias is a follow-up if needed.)

---

## Testing

**Pure numpy (no GPU):**
- `light_view_proj`: orthographic + look-at correctness — a point at `target` maps near NDC origin; a point inside the `extent` box maps within `[-1,1]` XY and `[0,1]` Z; a point behind the eye maps outside.
- `set_sun()` stores the sun; `enable_shadows()` stores config; `shadow_enabled` reflects presence of both a sun and an enabled rig.

**Headless render (GPU-gated, `get_offscreen_canvas` / `pytest.skip`):**
- `Globals` uniform is exactly 352 bytes.
- A sun set via `set_sun` lights a `StandardMaterial` sphere (Lambert gradient — lit side brighter than the away side), with no point lights present.
- Shadow map texture is created with the configured resolution and `depth24plus` format when shadows are enabled; not created when disabled.
- **Visual assertion:** render a white `StandardMaterial` plane with a sphere hovering above it under a sun. Mean luminance of the plane pixels in the sphere's shadow footprint is markedly lower than a lit region of the same plane. With shadows disabled, the two regions match.

**Demo:** `examples/shadow_demo.py` — a ground plane + a sphere (or the teapot) under a single directional sun, camera orbiting to show the cast shadow move. Runs interactively and via the `--render` MP4 smoke path.

---

## Roadmap / follow-ups

**Shipped**
- **VS1 — directional hard shadows** (2026-07-17). Sun term + depth-only shadow pass + `group(2)` comparison-sampler lookup.
- **VS2 — PCF soft shadows** (2026-07-17). `enable_shadows(pcf_radius=...)` averages a `(2r+1)²` comparison grid; `r=0` hard, `r=1` (3×3) default. Radius rides the unused `Globals` pad slot (no size change).
- **A — auto-fit frustum** (2026-07-17). `enable_shadows(auto_fit=True)` (default) fits the ortho box to the scene bounding sphere each frame via `RenderPipeline._scene_bounds` (per-geometry local AABB cached, transformed by every instance) → `_sun_light_view_proj` derives target/extent/near/far. `auto_fit=False` keeps the manual path.
- **B — slope-scaled bias** (2026-07-17). `shadowFactor(world_pos, N)` scales the bias by `1 + 4·(1 − N·L)` (L toward the casting light), so grazing surfaces stop self-shadowing without peter-panning. Removed the need to hand-tune `bias`.
- **D — spot lights + spot shadows** (2026-07-17). `engine.set_spot(SpotLight)` — flashlight cone (inner→outer falloff + 1/d² + range cutoff) as a real `StandardMaterial` term. Perspective shadow via `compute_spot_light_view_proj`; a `shadow_caster` field (0/1/2) routes the shadow to sun or spot (spot has priority). `Globals` 352→416B. Demo `examples/spot_demo.py`.

**Prioritized next (recommended order)**

Block 1 — quality (remaining):
- **C) Contact-hardening (PCSS)** — *advanced, optional.* Our PCF is a fixed-radius blur (uniform softness). PCSS does a blocker-search pass to vary the kernel by occluder distance (sharp at contact, soft far away). Builds on the PCF we have. ~half day. Normal-offset bias (sampling at `world_pos + N·offset`, complementing the shipped slope-scaled bias) also slots in here if acne ever resurfaces at very high grazing angles.

Block 2 — new light types (each reuses the depth-only pass + `group(2)` sampling):
- **E) Point-light shadows (cubemap)** — *the hard one.* A point light emits in all directions, so one 2D map can't capture it: use a 6-face `depth24plus` cube shadow map (`texture_depth_cube`), render the scene once per face with a 90° perspective matrix (6× the shadow-pass cost), store linear light→fragment distance, and sample with the world-space `light→fragment` direction. Cheaper-but-worse alternative: dual-paraboloid (2 hemispherical maps). Gate behind a per-light `casts_shadow` flag. ~1 day.

Block 3 — scale:
- **F) Multiple shadow casters** — today only one light casts (sun *or* spot). Full spec written: **`.knowledge/analysis/2026-07-18-multi-caster-shadows-design.md`** — Option A (two dedicated maps for sun + spot, ~half day, gives the two-crossing-shadows demo) and Option B (general N-caster array, ~1 day+). Deferred until wanted.
- **G) Cascaded Shadow Maps (CSM)** — *for large/outdoor scenes.* One ortho map over a big scene gives low resolution everywhere → blocky shadows. Split the camera frustum into N depth ranges, each with its own tighter map fit to that slice; pick the cascade in the shader by fragment view-depth. ~1 day. Overkill for current demos; only when scenes grow.

**Cleanup (low priority):** the shadow pass gathers + uploads transforms independently of the main pass (`_collect_mesh_instances` + `_shadow_batch_buffers`) — a small per-frame duplication. Could be unified by hoisting batch computation so both passes share it.

Recommended next: nothing pressing — the single-map light types (sun, spot) are done. Reach for **E (point/cubemap)**, **F (multi-caster)**, **C (PCSS)** or **G (CSM)** only when a concrete scene needs them.
