# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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