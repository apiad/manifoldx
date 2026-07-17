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

## Architecture

Shadow state is engine-level (like lights and IBL). The system adds one pass **before** the existing main render pass:

1. **`src/manifoldx/render/passes/shadow.py`** — new depth-only pass. Renders all mesh-pass geometry from the sun's POV into an offscreen depth texture (the shadow map). No color attachment, minimal vertex-only pipeline.
2. **Light-space matrix** — CPU-side numpy: `light_view_proj = ortho(extent, near, far) @ look_at(eye, target, up)`, where `eye = target - normalize(direction) * back_distance`. Pure Python, unit-testable without a GPU.
3. **`Globals` uniform extension** — add `light_view_proj: mat4x4<f32>` (64B) + `shadow_enabled: u32`, `shadow_bias: f32`, `shadow_map_size: f32`, + 4B pad. Grows the struct 240 → 320 bytes.
4. **Mesh pass bind group 2** — the shadow map (`texture_depth_2d`) + a comparison sampler. (Group 0 = main/transforms/material/lights, group 1 = IBL, group 2 = shadow.)
5. **`StandardMaterial` WGSL** — project `world_pos` into light space, sample the shadow map with `textureSampleCompareLevel` (hardware 2×2 PCF, near-free), and attenuate **only the directional light term** by the resulting shadow factor.

Per-frame order becomes: **shadow pass → [main pass: mesh → skybox → volume → label → axis]**.

---

## What casts and what receives

- **Casts:** exactly one `DirectionalLight` with the new `cast_shadow=True` kwarg (default `False`). If multiple are flagged, the first in `engine._lights` wins; if none, the shadow pass is skipped and `shadow_enabled = 0`. All mesh-pass entities are rasterized into the shadow map.
- **Receives:** only `StandardMaterial`. Other materials are untouched in v1.
- The shadow factor multiplies the directional contribution only. IBL ambient/specular and point lights are **not** shadowed in v1 (ambient light does not cast a directional shadow — correct by construction).

---

## API

Directional light gains an opt-in flag:

```python
sun = DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -1.0, -0.3), cast_shadow=True)
engine.add_light(sun)
```

Shadow rig configured engine-level:

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

`back_distance` (how far along `-direction` the virtual eye sits) defaults to `far * 0.5` clamped so the box is fully in front of the near plane; not user-facing in v1. Calling `enable_shadows()` allocates the shadow map + comparison sampler lazily on first frame. Without it, `shadow_enabled = 0` and everything renders exactly as today.

The `cast_shadow` flag lives on `DirectionalLight` (not in its GPU `get_data()` layout — shadow selection is CPU-side only, so the light's uniform bytes are unchanged and no shader light struct is touched).

---

## Data & shader details

- **Globals (320B):** existing fields unchanged; appended `light_view_proj` (mat4), `shadow_enabled`, `shadow_bias`, `shadow_map_size`, pad. The renderer writes `light_view_proj` each frame from the flagged sun's direction + the configured rig.
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
- `enable_shadows()` stores config; `shadow_enabled` reflects presence/absence of a flagged casting light.

**Headless render (GPU-gated, `get_offscreen_canvas` / `pytest.skip`):**
- `Globals` uniform is exactly 320 bytes.
- Shadow map texture is created with the configured resolution and `depth24plus` format when shadows are enabled; not created when disabled.
- **Visual assertion:** render a white `StandardMaterial` plane with a sphere hovering above it under a flagged sun. Mean luminance of the plane pixels in the sphere's shadow footprint is markedly lower than a lit region of the same plane. With shadows disabled, the two regions match.

**Demo:** `examples/shadow_demo.py` — a ground plane + a sphere (or the teapot) under a single directional sun, camera orbiting to show the cast shadow move. Runs interactively and via the `--render` MP4 smoke path.

---

## Follow-ups (explicitly out of v1)

- **VS2:** larger-kernel PCF for soft edges (shader-local change, reuses the same comparison sampler).
- Spot-light shadows (single perspective map — near-identical pipeline).
- Point-light shadows (6-face cubemap depth).
- Auto-fit the ortho frustum to the scene AABB (drop `extent`/`target` micromanagement).
- Slope-scaled / normal-offset bias; cascaded shadow maps for large scenes.
