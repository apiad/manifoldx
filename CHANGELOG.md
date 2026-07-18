# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Known limitations

- **IBL skybox: first-frame device lockup on NVIDIA Quadro M2000M / Vulkan** — enabling `env.show_skybox = True` on the *very first* presented frame deadlocks the device (the frame's submit never completes) on this GPU/driver (580.159.03). It is a driver first-frame-init lockup, not a code defect: depth format (`depth24plus` everywhere), depth config, bind group, and pipeline layout are all correct, and the pass renders perfectly once *any* prior frame has been presented — a constant-colour skybox renders, and the mesh IBL path samples the same cube fine. The skybox therefore stays **default-OFF**; `examples/ibl_demo.py` is unaffected because it only toggles the skybox via the `S` key after startup (post-first-frame). The only broken usage is scripting `show_skybox = True` before the first present. No warm-up-frame workaround was added (would mask an old-GPU driver quirk in the shared render loop). Investigation + evidence table: `.knowledge/analysis/2026-07-05-ibl-v1-design.md` (§ "Known limitation: first-frame device lockup").

### Features

- **Shadow mapping v1 (directional)** — `StandardMaterial` now casts and receives hard shadows from a single directional sun. New `engine.set_sun(DirectionalLight(...))` adds a real directional term to the PBR shader — previously `DirectionalLight` was silently packed into the point-light array and never worked as a directional light, so this gives the class its first correct consumer. `engine.enable_shadows(target, extent, resolution, near, far, bias)` makes the sun cast: a new depth-only shadow pass (`src/manifoldx/render/passes/shadow.py`, encoded before the main pass in `Engine._draw_frame`) rasterizes all mesh geometry from the sun's POV into a `depth24plus` shadow map, which the `StandardMaterial` fragment shader samples through a comparison sampler (`textureSampleCompareLevel`, hardware 2×2 PCF) to attenuate the sun term. Fragments outside the sun's ortho frustum stay lit; a constant `bias` kills shadow acne. Light-space matrix math is the pure-numpy `manifoldx.shadow.compute_light_view_proj` (ortho × look-at, wgpu z∈[0,1]). The `Globals` uniform grew 240 → 352 bytes (`light_view_proj` mat4 + sun direction/color/intensity + `shadow_enabled`/`shadow_bias`/`shadow_map_size`); the shadow map binds as `@group(2)` on the `StandardMaterial` pipeline, with a 1×1 placeholder depth texture when shadows are off so the pipeline layout is always satisfied. Point-light and IBL contributions are unshadowed in v1 (ambient does not cast a directional shadow). Only `StandardMaterial` casts/receives; sprite/label/volume/axis/GUI paths are untouched. Demo at `examples/shadow_demo.py` (four objects — three spheres + a cube in metallic/matte/polished materials — orbiting and bobbing at different heights over a central pillar and floor, their cast shadows sweeping the ground, climbing the pillar, and crossing each other). **PCF soft shadows (VS2):** `enable_shadows(pcf_radius=...)` averages a `(2r+1)²` grid of comparison taps around each shadow lookup for soft edges — `0` = a single hard tap (VS1 behaviour), `1` = 3×3 (default), `2` = 5×5, etc.; the radius rides in the previously-unused `Globals` pad slot, so the uniform size is unchanged. Spot/point-light shadows and auto-fit frustum are follow-ups. Design: `.knowledge/analysis/2026-07-17-shadow-mapping-v1-design.md`. Plan: `.knowledge/plans/2026-07-17-shadow-mapping-v1-plan.md`.

- **IBL v1 (Image-Based Lighting)** — split-sum physically-based environment lighting for `StandardMaterial`. New `manifoldx.ibl` module: `EnvironmentMap` dataclass with builders `from_color(rgb)`, `from_sky(zenith, horizon, ground)`, `from_image(path, exposure)`, and `from_hdr(path)` (Radiance RGBE decoder included). `engine.set_environment(env_or_preset)` activates an environment; built-in presets: `"studio"`, `"sky"`, `"neutral"`, `"dark"`. On first use, `_precompute()` runs CPU-side: equirectangular → 128 px cubemap, 64 px irradiance map (256 cosine-weighted samples), 8-mip prefiltered specular chain (512 GGX-importance samples per mip), all stored as float16. Pre-baked 512×512 Smith-GGX BRDF LUT ships as a binary asset (`src/manifoldx/assets/ibl/brdf_lut.npy`; regenerable via `scripts/gen_brdf_lut.py`). The `StandardMaterial` WGSL shader was extended to add a correct `proj: mat4x4<f32>` field (fixing a latent `camera_pos` offset bug), `@group(1)` IBL texture bindings, `fresnelSchlickRoughness`, and a split-sum IBL block that additively combines diffuse irradiance + specular split-sum with existing directional/point lights. The `Globals` uniform was extended from 224 → 240 bytes to add `ibl_intensity f32`, `ibl_enabled u32`, and 8 bytes of pad. Optional skybox: new `src/manifoldx/render/passes/skybox.py` draws a fullscreen triangle at depth=1.0 (far plane) sampling the prefiltered map with Reinhard tone map + sRGB gamma; activated when `env.show_skybox = True`. Demo at `examples/ibl_demo.py` (4×4 metallic-roughness grid, press 1/2/3 to cycle environments, S to toggle skybox). Design: `.knowledge/analysis/2026-07-05-ibl-v1-design.md`. Plan: `.knowledge/plans/2026-07-05-ibl-v1-plan.md`.

- **Textured PBR v1** — first slice of texture support in the core mesh-PBR path. New `manifoldx.textures` module exposes `load_texture(engine, path) -> TextureHandle`, backed by a per-engine `TextureRegistry`; images decode via Pillow and upload as `Rgba8UnormSrgb` (so sampling auto-decodes sRGB to linear in the shader). `StandardMaterial` accepts a new `albedo_map: TextureHandle | None` kwarg; when set, the fragment shader samples `textureSample(albedo_tex, albedo_sampler, uv).rgb` instead of using the scalar `material.albedo`. Geometry dicts now carry an optional `"uvs"` key; `sphere()` and `plane()` emit UVs by default. `GeometryRegistry.create_buffers` interleaves `[pos, normal, uv]` (stride 32B) when UVs are present, else falls back to `[pos, normal]` (stride 24B); the mesh-pass pipeline now reads the geometry's actual buffer stride so a scalar `StandardMaterial` bound to a UV-bearing geometry still renders correctly. The pipeline cache key includes `material_subtype` so scalar and textured `StandardMaterial` coexist in one scene without collision. Trying to attach a textured `StandardMaterial` to a geometry without UVs raises `MaterialGeometryMismatchError` at pipeline-cache time. New `manifoldx.assets.obj.load_obj` (re-exported as `manifoldx.load_obj`) parses Wavefront OBJ files with all four face-line forms (`f v`, `f v/vt`, `f v/vt/vn`, `f v//vn`); fan-triangulates polygon faces; rejects mixed forms and negative indices with line-numbered errors. First bundled binary assets: `examples/assets/teapot/teapot.obj` (UV-mapped Newell teapot, ~93 KB) and a procedural `teapot_albedo.png` regenerable via `scripts/gen_teapot_albedo.py`; and `examples/assets/sun_earth_moon/{earth,moon}.jpg` (public-domain NASA imagery). Two new demos: `examples/teapot_demo.py` and `examples/sun_earth_moon_demo.py`. Plan: `.knowledge/plans/2026-06-09-textured-pbr-v1.md`. Design: `.knowledge/analysis/2026-06-09-textured-pbr-v1-design.md`.

- **GUI v1** — in-engine retained-mode GUI for sim controls + HUD readouts. New `manifoldx.gui` package: `Panel`, `Text`, `ValueDisplay`, `Button`, `Slider`, `Toggle`; stack + flex layout; theme + named-class styling with per-widget overrides. Rendering via a new `gui` render pass at the end of the order, using `RectMaterial` (WGSL signed-distance rounded rect) for backgrounds and the existing `LabelTextureAtlas` for glyphs (no new font dependency). Interaction via `_GuiBridge` — subscribes to pointer events on the bus, hit-tests the widget tree top-down, dispatches per-widget state mutations, and exposes `engine.gui.pointer_over_gui` for cooperative consume by user systems. Slider supports drag capture and emits `:change` continuously plus `:commit` on pointer-up. Example at `examples/gui_demo.py`.

### Fixes

- **Mesh-pass per-material uniform buffers** — `mesh_batches` previously keyed only on `(geom_id, mat_type, mat_subtype)`, so two material instances of the same class on the same geometry collapsed into one batch and the first instance's uniform data won. The bug had been silently affecting `pbr_demo.py` for a while (every cube rendered with the first cube's color; every sphere with the first sphere's). Made visible by the new textured-PBR work: Earth and Moon — both textured `StandardMaterial` on `sphere()` — initially rendered with the same texture. Fixed by extending the batching key to include `mat_id` and allocating one material-uniform buffer per `mat_id` in the mesh pass.

## [0.9.0] - 2026-05-10

### Features

- **Input layer v1** — keyboard, mouse, and resize events ride the event bus from event-driven-system v1. New `manifoldx.input` module exposes `KeyEvent`, `PointerEvent`, `WheelEvent`, and `ResizeEvent` dataclasses (re-exported from the package root). Handlers register the same way as any other event: `@engine.on("key_down")`, `@engine.on("pointer_move")`, etc. Alongside discrete events, a polling state at `engine.input` exposes Bevy-style `is_pressed(key)` / `just_pressed(key)` / `just_released(key)` for keys and mouse buttons, plus `mouse_pos`, `mouse_delta`, `wheel_delta`, and `viewport_size`. Polling state is updated immediately on every event (so `is_pressed` reads are live), while `just_*` sets and `*_delta` accumulators are this-frame-bound and reset at the new step 2.5 of `_draw_frame`. New `examples/input_orbit.py` (drag-to-orbit + wheel-zoom via events) and `examples/input_fly.py` (WASD fly cam via polling). Design: `.knowledge/analysis/2026-05-08-input-events-design.md`.
- **Event-driven system v1** — new `engine.emit(name, payload)` / `@engine.on(name)` event bus running parallel to the frame loop. Sync handlers run inline at the head of the next frame; async handlers are scheduled on the active asyncio loop (rendercanvas's in interactive `run()` mode, a private fallback in headless `render()` / tests) and progress between draw callbacks. `await engine.tick()`, `await engine.delay(seconds)`, and `await engine.elapsed_at(target)` are the async waiter primitives, plus `await engine.run_blocking(fn, ...)` as the escape hatch for genuinely blocking work. Globals (camera, lights) are written immediately; ECS reads come from a `ReadOnlyView` (mutations route through `engine.commands` / `engine.spawn` / `engine.destroy`). The legacy `@engine.startup` / `@engine.shutdown` / `@engine.update` decorators have been replaced by `@engine.on("startup" | "shutdown" | "frame")`; the `'frame'` event payload carries `dt`, `elapsed`, and `frame`. `engine.quit()` cancels all in-flight async tasks, allowing `try/finally` cleanup to run before the loop closes. Errors from handlers propagate and crash the frame loop in v1 by design. New `examples/event_dolly.py` (async camera dolly with `await engine.delay(...)`) and `examples/event_pulse.py` (sync handler emitting `engine.spawn`). Design: `.knowledge/analysis/2026-05-08-event-driven-system-design.md`.
- **Volume rendering v1** — direct volume rendering (DVR) of a 3D scalar field via fragment-shader raymarching. New `Volume` ECS component (handle into a per-engine `VolumeRegistry`), new `VolumeMaterial` with reusable colormap LUT + per-material 256-sample opacity LUT (built from piecewise stops or a numpy array). `engine.register_volume(numpy_array)` / `engine.update_volume(handle, new_array)` for CPU upload; `engine.bind_compute_volume(...)` is reserved as a v2 stub. The volume pass renders between sprite and label with depth-test on / depth-write off and pre-multiplied alpha blending; a fullscreen-quad fragment shader does ray/box-AABB intersection in entity-local space, then a fixed-step front-to-back composite sampling two LUTs per step. New `examples/volume_demo.py` shows a 64³ Gaussian blob. v1 is unlit and CPU-uploaded; lighting, isosurface/MIP modes, compute-written volumes (`Writes[Volume3D]`), editable transfer-function widgets, and slicing are explicit non-goals deferred to follow-up plans. Design: `.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md`.
- **Compute systems Phase 2 — Python → WGSL shader compiler** — `Compute.compile()`'s default body now traces a typed-Python `def main(self, i)` body to WGSL via `manifoldx.compute.transpile`. Kernel authors write plain Python with PEP-526 annotations on every local; the transpiler emits the bind-group header, helper functions (`fn _<ClassName>_<name>`), and the `@compute @workgroup_size(W) fn main(...)` wrapper. `examples/nbody_compute.py` is now a Compute subclass with a `pair_accel(...) -> vec3` helper method instead of an inlined WGSL string; numerics agree with the Phase-1 hand-written kernel within `rtol=1e-5`.
- **`engine.compute(cls)` validates synchronously** — WGSL is compiled via `device.create_shader_module(...)` at registration time. Errors surface as `ComputeShaderCompileError` (category `wgpu-validation`) before any frame runs, alongside Python-source-level transpiler errors (`unsupported-construct`, `missing-annotation`, `unknown-name`, `type-mismatch`, `implicit-promotion`, `recursion`).
- **Per-Component `_layout` table** — `Component.__init_subclass__` now derives `{field_name: (offset_in_floats, length_in_floats)}` from the existing field annotations. Pre-base-class built-ins (`Transform`, `Mesh`, `Material`) ship explicit `_layout` class attrs. The transpiler reads this table to emit storage-buffer offset arithmetic without component-specific knowledge.
- **`examples/point_cloud_compute.py`** — protoplanetary-disk demo (~10000 colormapped dust particles + central star) ported to a Phase-2 compute kernel. The kernel has no inner loop — per-particle Keplerian gravity around the origin — and writes a *scalar* component (`ScalarValue.value = current speed`), so the GPU-driven colormap pipeline picks up speeds without any CPU round-trip.
- **vec3/vec4 swizzle reads in compute kernels** — `accel.x`, `next_pos.y`, `self.transforms[i].pos.z` and similar `.x/.y/.z[/w]` attribute access on vec3/vec4-typed expressions now lower to WGSL component access, returning f32. Compound forms (swizzle on top of an indexed binding read) work too.
- **`examples/gas_compute.py`** — ideal-gas box-bounce demo ported to a Phase-2 compute kernel. First example to use vec3 swizzle inside a kernel: per-axis bounce conditions read individual components (`if next_pos.x < -self.half_size`). Pairwise elastic collisions are now also on GPU, reformulated to be race-free: each thread `i` surveys all neighbours and accumulates *its own* impulse from approaching overlaps; thread `j` independently computes the equal-and-opposite impulse for itself, so no thread ever writes another thread's velocity. Also the first example to exercise `vec3 / f32` broadcast (collision normal `diff / dist`).
- **Phase-2 DSL is now mypy-clean in kernel bodies** — `vec3`/`vec4` are real classes with arithmetic dunders; math builtins (`dot`, `cross`, `length`, `sqrt`, `pow`, `abs`, etc.) carry proper PEP-484 signatures; `Reads` / `Writes` / `ReadsWrites` are `Generic[T]` with `__getitem__(int) -> T` under TYPE_CHECKING; `Uniform[T]` is a PEP-695 type alias for `T`; `Float` is a TYPE_CHECKING alias for `float`. Both compute examples now type-check at zero errors against `mypy`. Auto-bound sentinel-string uniforms (e.g. `dt: Uniform[float] = "frame_dt"`) keep needing a `# type: ignore[assignment]` per uniform.
- **`examples/smoke_demo.py`** — tileable Perlin-FBM volumetric smoke, exercising the volume pass with a moving camera and a wrap-padded source for seamless looping.

### Refactors

- **`compile()` override is the escape hatch, not the default.** Phase-1 user kernels that override `compile()` continue to work unchanged; only the default body changed.
- **`renderer.py` split by render pass** — `_render_mesh_batches`, `_render_sprite_batches`, `_render_label_pass`, `_render_axis_pass`, `_render_volume_pass` and their pass-private helpers move out of `RenderPipeline` into `src/manifoldx/render/passes/{mesh,sprite,label,axis,volume}.py` as module-level functions taking the renderer as their first argument. `RenderPipeline.render` becomes a thin dispatcher; cross-pass shared state (pipeline cache, `_BatchBuffers`, `TransformCache`, `_get_or_create_pipeline`) stays in `renderer.py`. `renderer.py` shrinks from 1991 → 964 lines; full test suite (392) and `examples/volume_demo.py` smoke-render unchanged.

## [0.8.0] - 2026-05-06

### Features
- **Sci-viz Plan 1 (foundation)** — `manifoldx.viz` subpackage for scientific-visualization primitives. Six built-in 1D RGBA8 colormap LUTs (viridis, magma, plasma, inferno, turbo, gray) precomputed from matplotlib. New ECS components: `PointCloud` (marker), `ScalarValue` (per-particle scalar attribute), `Radius` (per-particle world-space radius). New `ColormapMaterial` maps the per-instance scalar through a 1D LUT in the fragment shader; default unlit, optional `lit=True` Lambert against a fixed view-space light direction.
- **Sprite render path** — Camera-facing point sprites with sphere-imposter fragment shading, scaled by per-instance `Radius`. New `SPRITE_QUAD` built-in geometry. `RenderPipeline` splits batches into mesh and sprite groups; sprite path uploads parallel storage buffers (`transforms`, `scalar_values`, `radii`) per frame and binds a per-cmap LUT texture.
- **`[viz]` extra** — Optional `pillow>=10.0` dependency group, staged for Plan 2 text rendering.
- **Sci-viz Plan 2 (text rendering)** — `manifoldx.viz` adds the `TextLabel` ECS component, the `LabelMaterial` camera-facing-billboard material with depth-test on / depth-write off / alpha-blend on, and the `LabelTextureAtlas` host-side cache that rasterizes strings via PIL (DejaVu Sans Mono bundled, 256×64 RGBA8 tiles, sRGB-correct) and uploads them lazily to a `texture_2d_array` (256-slice cap in v1).
- **Label render pass** — `RenderPipeline.run` now batches `TextLabel + LabelMaterial` entities into a third draw group dispatched after the 3D opaque pass. Pipeline cache key extended with a `"label"` fourth element so the world-anchored label pipeline never collides with sprite or mesh pipelines.
- **`engine.get_label_atlas()`** — lazy accessor for the per-engine atlas, used by the renderer and by user code that wants to register strings up front.
- **Sci-viz Plan 3 (axes, screen anchoring, scale bars, colormap legends)** — `LabelMaterial(anchor_mode="screen")` actually renders in NDC (was a silent fallback in Plan 2); `AxisFrame` component + `AxisMaterial` line material with `anchor_mode` (`"world"` | `"screen"`) + native LineList rendering pipeline. Built-in `axis_line_x` / `axis_line_y` / `axis_line_z` geometries auto-registered alongside `sprite_quad`. Screen-anchored axes give scale bars without a dedicated component.
- **`LabelTextureAtlas.register_colormap_legend(cmap, orientation=...)`** — rasterizes a colormap LUT into one of the 256×64 atlas slices and returns the slot index. Renders via the existing screen-anchored LabelMaterial path — no new material or pipeline needed for legends.
- **`Globals` uniform extended** — `viewport_size: vec2<f32>` (208 → 224 bytes) replaces the v1 `1024.0` calibration constant; labels and screen-anchored primitives scale to actual viewport pixel dimensions on any canvas.
- **`examples/axes_demo.py`** — small standalone demo that exercises every Plan 3 primitive end-to-end: world-anchored axes with billboard end-cap labels, screen-anchored HUD overlay, screen-anchored scale bar, screen-anchored colormap legend. Camera orbits to highlight the world-vs-screen-anchored contrast.

### Refactors (Plan 3)
- **Pipeline factory `_get_or_create_pipeline`** — accepts `line=True` for a LineList-topology, opaque, depth-write-on path. Three bind slots (globals, transforms, material uniform); buffer is 32 bytes to fit `AxisMaterial`'s rgba + anchor_mode + 3 pad uniform.
- **`RenderPipeline.run` routing** — priority `axis > label > sprite > mesh`; new `_render_axis_pass` and dedicated `_axis_batch_buffers` parallel the existing sprite/label paths.
- **`AxisMaterial.pipeline_subtype = anchor_mode`** — world and screen modes get separate pipelines in the cache (parallels Plan 2's `LabelMaterial`).
- **Component base unchanged** — the new `AxisFrame` (2 floats: `extent`, `thickness`) auto-registers via the `Component` annotation pattern, no boilerplate in user code.

- **Sci-viz Plan 4 (Altair-style declarative shim)** — `manifoldx.viz` adds a grammar-of-graphics API on top of the imperative ECS primitives Plans 1-3 shipped:

  ```python
  import manifoldx.viz as mxv

  chart = (
      mxv.points(positions=p, color=mxv.color(s, cmap="viridis"), size=r)
      + mxv.mesh(geometry=sphere(0.3, 16), material=...)
      + mxv.axes(extent=4)
      + mxv.legend(cmap="viridis", title="Speed")
      + mxv.scale_bar(label="2 units")
      + mxv.lights([...])
  )

  @chart.simulate
  def step(dt):
      p[:] += v * dt   # mutate live arrays in place
      s[:] = np.linalg.norm(v, axis=1)

  chart.cli()
  ```

  Six marks (`points`, `mesh`, `axes`, `legend`, `scale_bar`, `lights`), three Channel wrappers (`color`, `size`, `position`) for per-channel config, `+` operator for layered single-scene composition, `@chart.simulate(dt)` decorator for per-frame mutation. The shim holds references to the user's numpy arrays — a per-frame sync system copies them into ECS storage so live mutation propagates to the GPU. Escape hatch: `chart.engine` returns the underlying Engine for users who need to drop down to imperative ECS ops mid-design (e.g. camera config, custom systems).

- **`examples/scatter_plot.py`** — the "hello world" of the declarative API: 500-particle 3D scatter with axes, colormap legend, scale bar, two PBR lights, and live physics in ~30 lines (vs ~120 for the equivalent imperative-ECS demo).

- **`manifoldx.random`** — initial-condition generators for demos. Six position generators (`positions_uniform`, `_in_box`, `_in_sphere`, `_on_sphere`, `_in_disk`, `_gaussian`), five velocity generators (`velocities_gaussian`, `_uniform`, `_on_sphere`, `_tangent`, `_orbit`), two scalar generators (`scalars_uniform`, `scalars_gaussian`). All return float32 arrays. Optional `rng` kwarg accepts None (entropy seed), int (deterministic), or an existing `numpy.random.Generator`.

- **`manifoldx.physics`** — vectorized simulation primitives. `all_pairs(positions)` returns a dataclass with `diff` / `dist` / `dist_safe`; `gravity(positions, masses, G, softening)` returns N-body acceleration; `central_gravity(positions, GM, softening, center)` for single-source cases; `box_boundary(positions, velocities, half_size, dt, mode)` and `sphere_boundary(positions, velocities, radius, mode, strength, dt)` for in-place velocity reflection; `elastic_collisions(positions, velocities, radius, restitution)` for equal-mass pair collisions.

- **Examples rewritten** to use `mx.random.*` and `mx.physics.*`. Net effect on the four physics-heavy demos (nbody, gas, boids, point_cloud_demo): 99 lines of inline physics → 20 lines of helper calls; same visual behavior. Initial-condition setup compressed similarly: ~50 lines saved across five demos.

### Deferred from Plan 4 spec
- **Multi-viewport composition** (`|`, `&`). Renderer rewrite required; Plan 5+.
- **Generic `lines` mark.** Would need a generalized line material beyond axes; defer until concrete demand.
- **Standalone `text` mark.** Today's `TextLabel` + `LabelMaterial` covers this imperatively via the escape hatch; reconsider when usage shows it's wanted.
- **Reactive transforms** (Altair's `transform_calculate` etc.). The `@chart.simulate` callback covers everything with a fraction of the surface area.
- **Per-frame domain auto-scale.** `mxv.color(arr)` infers `(vmin, vmax)` from data once at build time; per-frame re-derivation would mean re-uploading material uniforms each tick.
- **Camera channel.** `chart.engine.camera.fit(...)` via the escape hatch suffices for now; promote to a `mxv.camera(...)` mark when ergonomics warrant.

### Deferred from Plan 3 spec
- **`ScaleBar` tag component** — the spec called for `ScaleBar(length, label_id)` as a routing tag, but on review the renderer doesn't need a distinct identity for scale bars: a screen-anchored axis line + a screen-anchored label compose into one. Plan 4's `scale_bar(...)` shim will package the composition.
- **Dedicated `ColormapLegendMaterial` + pipeline** — same simplification: legends route through the existing label pipeline by stashing the LUT as an atlas slice. A future `vertical` colormap orientation would benefit from a dedicated material when the atlas tile aspect (4:1) becomes a real constraint, but v1 horizontal legends work fine.
- **Line thickness honoring** — `AxisFrame.thickness` field is reserved; Plan 3 ships native LineList (1px on Vulkan/Metal). Quad-extrusion deferred until thicker lines have a concrete requirement.

### Refactors
- **`_BatchBuffers` helper** — Per-batch GPU buffer management lifted out of `RenderPipeline` into a dedicated helper, supporting capacity-tracking lazy allocation for `transforms`, `scalar_values`, and `radii`.
- **Globals uniform extended** — `vp + view + proj + camera_pos + pad` (208 bytes). The sprite vertex shader projects view-space billboards directly through `globals.proj`, removing a brittle `transpose(view)` approximation. Existing material shaders (Basic, Phong, Standard) updated to match the new layout.
- **Pipeline cache key** — Now `(geom_id, mat_type, mat_subtype, sprite)` so different colormaps share a sprite pipeline but rebind only the LUT texture.
- **`GeometryRegistry`** — Added name-based lookup APIs (`register_by_name`, `get_id`, `get_by_name`, `__contains__`) and auto-registers `SPRITE_QUAD` on init.
- **`_BatchBuffers` extended** — additional lazily-allocated `label_indices` buffer parallels the existing `transforms` / `scalar_values` / `radii` storage paths.
- **Pipeline factory `_get_or_create_pipeline`** — now accepts `label=True` for an alpha-blended, depth-write-off path with bind slots for a `texture_2d_array` + sampler.

### Fixes
- **Colormap LUT sRGB** — LUT textures now use `rgba8unorm_srgb` so the GPU sRGB-decodes on sample. Previously the matplotlib-encoded LUT bytes were sampled as linear and the framebuffer applied sRGB encoding on write, producing colors brighter than matplotlib's intended swatch.

## [0.2.0] - 2026-04-06

### Features
- **N-body simulation** — 500-body gravitational simulation with pure-numpy vectorized physics (250k force pairs/frame)
- **Ideal gas simulation** — 500 particles with elastic collisions and virtual bounding box walls
- **Boids flocking** — 300 boids with separation, alignment, cohesion rules + 4 wandering predators they flee from
- **Camera far plane** — Added near/far clipping planes (0.1/1000) for distant object support
- **Engine quit fix** — Fixed event loop termination for proper window closing

### Documentation
- Expanded motivation section with scientific simulation use cases
- Added PyGfx comparison explaining why ECS is better for data-driven simulations
- Added detailed showcase of three complex demos with vectorization patterns
- Simplified install instructions (pip + uv)

### Refactors
- Split nbody.py into gravity-only demo; gas.py for collision demo
- Tuned boids/gas parameters for better visual balance

### Fixes
- Boids now significantly scared of predators (20x fear force, 10-unit detection radius)
- Camera test now uses correct near/far values for VP matrix validation

## [0.1.0] - 2026-04-06

### Initial Release
- ECS with Structure-of-Arrays (SoA) layout
- Instanced GPU rendering with material-specific pipelines
- PBR (Physically Based Rendering) with GGX BRDF
- Basic materials (Phong, Standard)
- Geometries: cube, sphere, plane
- Camera with orbit controls and fit/bounds helpers
- Examples: hello_world, cube, pbr_demo, spheres