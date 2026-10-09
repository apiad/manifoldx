# Render quality v1: reversed-Z, tonemap curves, FXAA, cascaded shadows, SSAO

Status: proposed (issue #30). Three phases, one PR each, in order.

## Why

uh-twin renders the University of Havana's Colina, about 550 m across, in the desktop
viewer. Four of the renderer's limits show there:

| Problem | Cause in the code | Effect |
|---|---|---|
| Z-fighting at distance | `depth24plus` main depth (`engine.py`, `_render_only`), forward [0,1] projection (`camera.py`, `get_projection_matrix`). Resolution grows with distance² / near: with near 0.1 m it is 2.4 cm at 200 m and 15 cm at 500 m. | Ground layers 3 to 25 cm apart flicker from an overview. |
| Coarse, washed-out shadows | One map; `_sun_light_view_proj` (`renderer.py`) auto-fits an ortho box to the bounding sphere of every mesh (`_scene_bounds`). | The Colina gets about 27 cm per texel at 4096². |
| Flat lighting | StandardMaterial adds a fixed 3% ambient (or IBL) with no occlusion. The tonemap is a fixed per-channel Reinhard `c/(c+1)` with no exposure (`resources.py`, StandardMaterial WGSL). | Corners, bases and courtyards read as flat as open walls. Sunlit pale stone saturates toward grey. |
| Aliased edges | Main pass `sample_count=1`, no AA. | Stair edges, columns and roof lines crawl when the camera moves. |

Measured on zion's Quadro M2000M (Vulkan, the adapter `high-performance` selects; the
Intel HD 530 only appears in wgpu's adapter probe). The bottleneck is the CPU, about
41 ms per frame in Python at uh-twin's scene, not the GPU. Every addition here must cost
GPU time, not Python per frame or per draw.

## Goals

- No visible z-fighting between surfaces 2 cm apart anywhere in a 1 km scene.
- A tonemap curve choice and an exposure, default unchanged.
- Sharp shadows near the camera and stable ones far from it.
- Contact darkening (ambient occlusion) in corners and at the foot of walls.
- Smooth edges.
- Existing apps and tests render the same unless they opt in, except where noted
  (reversed-Z is internal).

## Non-goals (v1)

- HDR render target and bloom. Tonemapping stays in the lit shaders, see below.
- A depth or normal pre-pass. It would double draw submission, which is the CPU cost
  this engine cannot afford.
- MSAA. It multiplies every attachment and needs resolve passes; FXAA covers the need
  at a fraction of the cost.
- Temporal effects (TAA, temporal SSAO).
- The web viewer (three.js). uh-twin will adopt three.js's own equivalents separately.

## Architecture after v1

```
encoder
  shadow pass(es)    depth-only, forward Z: one per cascade into a depth texture array
  main pass          colour: offscreen `scene_color` (swapchain format, *-srgb)
                     depth:  `depth32float`, reversed Z (clear 0, compare greater)
                     draws:  skybox, mesh, sprite, volume, label, axis (as today, minus GUI)
  ssao pass          (phase 3, if enabled) half-res AO from `scene depth` into `ao`
  ssao blur          (phase 3) bilateral blur `ao`
  final pass         fullscreen triangle into the swapchain:
                     scene_color x AO (if enabled), FXAA (if enabled), else a straight copy
  gui pass           onto the swapchain, load (not clear), as today's last draw
```

Today the main pass draws straight into the swapchain view with the GUI last
(`_render_only`, `RenderPipeline.render`, `_render_scene_passes`). v1 moves the scene
into `scene_color` and adds the final pass. The GUI moves out of
`_render_scene_passes` into its own render pass on the swapchain after the final pass.
Its pipeline is unchanged; its colour attachment's `load_op` becomes `load`.

`scene_color` uses the swapchain's format (`_configure_swapchain` picks an `-srgb`
variant). Shaders keep writing linear colour, the target encodes sRGB, and sampling it
in the final pass decodes back to linear, so the copy path is exact. Both
`scene_color` and the depth texture are recreated with the canvas size, where
`_render_only` recreates depth today.

`render_frame` stills and the id pass go through the same pipeline:

- **Id pass (`_render_ids`):** must render with FXAA and AO off. It already resets
  environment, fog and GUI for the duration, so do the same with the new state. Label
  colours must reach the read-back bit-exact; `tests/test_id_pass.py` is the guard.
- **Supersampled stills (`supersample > 1`):** keep their CPU downsample.

## Phase 1: reversed-Z, tonemap curves and exposure, FXAA

### Reversed-Z

- Main depth becomes `depth32float`, cleared to 0.0.
- The projection maps near to 1 and far to 0. In `camera.py`, `get_projection_matrix`:
  `[2,2] = near/(far-near)`, `[2,3] = far*near/(far-near)`, `[3,2] = -1`. Keep finite far.
- Every main-pass pipeline flips its compare: `less` becomes `greater`, `less_equal`
  becomes `greater_equal`. That covers axis/line, sprite, mesh, label, skybox, volume
  (in `renderer.py`, `skybox.py`, `volume.py`). The GUI keeps `always`.
- The skybox's far-plane trick writes z = 0 instead of z = w (`skybox.py`).
- The volume shader's NDC unprojection swaps its near and far z (`volume.py`).
- `Camera.project` and anything else reading the projection (`test_camera.py`,
  `test_camera_project.py`) follow the new matrix.
- Shadow passes stay forward Z: their maps, compare and the PCF lookup in
  StandardMaterial are untouched.

### Tonemap curves and exposure (in-shader)

Tonemapping stays where it is, at the end of the lit shaders (StandardMaterial and the
skybox), so unlit materials, labels, sprites and colormaps keep their exact colours.
What changes:

```python
engine.set_tonemap("reinhard", exposure=1.0)   # default: today's look, bit for bit
engine.set_tonemap("agx", exposure=1.0)        # AgX base (Troy Sobotka's), the recommended look
engine.set_tonemap("aces", exposure=1.0)       # Narkowicz's ACES fit
engine.set_tonemap("none", exposure=1.0)       # linear, clamped by the target
```

- Two new globals fields, `tonemap_mode: u32` and `exposure: f32`, in the padding after
  the fog block or appended. Keep `test_globals_layout.py` in step, and every consumer
  that binds a fixed globals size (shadow 416 B, skybox 240 B, mesh 432 B).
- Both shaders call one shared WGSL function, `tonemap(c * exposure, mode)`.
- Fog keeps its place after the tonemap.
- The skybox applies the same function instead of its own Reinhard.
- Mode `reinhard` at exposure 1.0 must reproduce today's StandardMaterial output
  exactly, so `test_colour_space.py`, `test_gamma.py` and `test_shadow_render.py`
  pass unchanged.

### FXAA

```python
engine.set_antialias("fxaa")   # or None (default: a straight copy)
```

- FXAA 3.11, the "quality 12" preset. It runs in the final pass on `scene_color`.
- Luma is computed from the sampled (linear) colour as `sqrt(dot(rgb, (0.299, 0.587, 0.114)))`,
  a perceptual approximation, since there is no gamma-encoded copy to read.
- When off, the final pass is a copy: sample `scene_color` with `textureLoad` (no
  filtering), write it out.

### Phase 1 tests

| Test | Check |
|---|---|
| reversed-Z | Two parallel planes 2 cm apart at 500 m, camera near 0.1: every pixel of their overlap shows the nearer plane's colour. It fails on today's depth. |
| projection | `test_camera.py` and `test_camera_project.py` updated to the reversed matrix; a point at near maps to depth 1 and at far to 0. |
| tonemap default | Existing colour tests pass unchanged. |
| tonemap curves | For `agx` and `aces`: output is monotonic in input luminance over a ramp of lit spheres, maps black to black, and stays below 1 for inputs up to 16. |
| exposure | Exposure 2 at mode `none` doubles a mid-grey lit pixel (within ±2 levels). |
| FXAA | A black-on-white diagonal edge: with FXAA the edge pixels take intermediate values, without it they are 0 or 255. Flat regions are unchanged within ±1. |
| copy path | With FXAA off, `render_frame` output equals today's (`test_render_frame.py`, the id pass tests). |

## Phase 2: cascaded shadow maps

```python
engine.enable_shadows(..., cascades=3)   # default 1: today's behaviour
```

- **Map and splits.** The shadow map becomes a `depth24plus` texture array with
  `cascades` layers at `resolution`². The camera's view range splits by the practical
  scheme (λ = 0.75 between logarithmic and uniform), from near to `shadow_distance`.
  `shadow_distance` is a new `enable_shadows` argument; it defaults to the camera's far
  plane, clamped to 300 m.
- **Fitting each cascade.** Each cascade gets an ortho light projection fitted to the
  bounding sphere of its frustum slice, so the box does not change size when the camera
  turns. The light-space origin snaps to whole texels so shadows do not shimmer when
  the camera moves.
- **Shadow pass.** It runs once per cascade into its layer, drawing what intersects that
  cascade's box. A CPU AABB test against the box is enough. The Python cost is one draw
  per batch per cascade, which is acceptable at uh-twin's 24 batches. Note it in
  CHANGELOG.
- **Uniforms.** Globals gain `light_view_proj[4]` and `cascade_splits: vec4<f32>`
  (view-space far distance of each cascade). The existing single `light_view_proj`
  becomes element 0.
- **Shader.** StandardMaterial picks the cascade from the fragment's view depth, applies
  PCF in that layer, and blends linearly over the last 10% of each cascade to hide the
  seam.
- **Unchanged.** Spot-light shadows keep their single perspective map. Cascades apply
  to the sun only.
- **`auto_fit`.** With `cascades=1` it keeps today's whole-scene fit.

### Phase 2 tests

| Test | Check |
|---|---|
| sharpness | A pole on a plane 10 m from the camera, in a 1 km scene: the shadow's penumbra width in pixels with 3 cascades is under half of that with 1. |
| far coverage | An object at 250 m still casts a shadow with 3 cascades. |
| stability | Moving the camera by sub-texel steps along x changes no shadow-edge pixel by more than 1 position (snapping works). |
| default | `cascades=1` reproduces `test_shadow_render.py` unchanged. |

## Phase 3: SSAO

```python
engine.enable_ssao(radius=1.0, intensity=1.0, samples=16)   # off by default
```

- **Input.** The scene's depth: the main pass's `depth32float`, now sampled, so it needs
  the `TEXTURE_BINDING` usage. Normals are reconstructed from depth derivatives in the
  AO shader, so there is no normal target and no pre-pass.
- **AO pass.** Half resolution, `r8unorm` target. The kernel is a normal-oriented
  hemisphere of `samples` points, rotated per pixel by a 4×4 noise tile. The range check
  fades occluders farther than `radius`.
- **Blur.** A 4×4 bilateral blur, depth-aware, so AO does not bleed across silhouettes.
- **Applied in the final pass** as `color * mix(1, ao, intensity)` before FXAA. This
  darkens direct light in occluded spots too: a forward renderer without a pre-pass
  cannot separate ambient from direct after the fact. At the default intensity this is
  the usual trade. A later version with a pre-pass could feed AO into StandardMaterial's
  ambient term only.
- **Reversed-Z.** The depth is reversed-Z, so reconstruction uses the reversed
  projection.
- **Unaffected.** Labels, the GUI and the id pass bypass AO. The GUI is drawn after the
  final pass; the id pass turns AO off for its duration.

### Phase 3 tests

| Test | Check |
|---|---|
| contact | A box on a plane, camera above at 45°: the plane's mean luminance in a strip touching the box is at least 10% below a strip 3 m away; with SSAO off, the two are within 2%. |
| open plane | A lone plane with SSAO on is unchanged within ±2 levels (no self-occlusion). |
| silhouettes | A sphere in front of a distant wall: the wall pixels next to the silhouette are not darker than 2 m away from it (the range check works). |
| id pass | Exact labels with SSAO enabled on the engine. |

## Cost and acceptance

- **Python per frame.** Measured with `cProfile` over 20 frames of uh-twin's scene, the
  added time stays under 1 ms per frame per phase, cascade draws excepted. Those are
  stated in CHANGELOG with their measured cost.
- **GPU.** At 1400×860 on the Quadro M2000M the frame rate does not drop below today's.
  The window is CPU-bound, so GPU work up to the CPU frame time is free. Measure it with
  `render_frame` timings at supersample 1, median of 30.
- **Acceptance.** uh-twin's viewer opts in to AgX with exposure tuned, FXAA, 3 cascades
  and SSAO. Its views (overview, escalinata, plaza, matcom, patio) are compared side by
  side with the baseline captures taken on 2026-10-09 before v1.

## Delivery

One PR per phase against `main`, each with its CHANGELOG entry (`### Features`), its
tests, `make test` and `make lint` green, and before/after captures in the PR body.
The plan: `.knowledge/plans/2026-10-09-render-quality-v1.md`.
