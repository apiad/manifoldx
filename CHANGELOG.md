# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Features
- **Sci-viz Plan 1 (foundation)** — `manifoldx.viz` subpackage for scientific-visualization primitives. Six built-in 1D RGBA8 colormap LUTs (viridis, magma, plasma, inferno, turbo, gray) precomputed from matplotlib. New ECS components: `PointCloud` (marker), `ScalarValue` (per-particle scalar attribute), `Radius` (per-particle world-space radius). New `ColormapMaterial` maps the per-instance scalar through a 1D LUT in the fragment shader; default unlit, optional `lit=True` Lambert against a fixed view-space light direction.
- **Sprite render path** — Camera-facing point sprites with sphere-imposter fragment shading, scaled by per-instance `Radius`. New `SPRITE_QUAD` built-in geometry. `RenderPipeline` splits batches into mesh and sprite groups; sprite path uploads parallel storage buffers (`transforms`, `scalar_values`, `radii`) per frame and binds a per-cmap LUT texture.
- **`[viz]` extra** — Optional `pillow>=10.0` dependency group, staged for Plan 2 text rendering.

### Refactors
- **`_BatchBuffers` helper** — Per-batch GPU buffer management lifted out of `RenderPipeline` into a dedicated helper, supporting capacity-tracking lazy allocation for `transforms`, `scalar_values`, and `radii`.
- **Globals uniform extended** — `vp + view + proj + camera_pos + pad` (208 bytes). The sprite vertex shader projects view-space billboards directly through `globals.proj`, removing a brittle `transpose(view)` approximation. Existing material shaders (Basic, Phong, Standard) updated to match the new layout.
- **Pipeline cache key** — Now `(geom_id, mat_type, mat_subtype, sprite)` so different colormaps share a sprite pipeline but rebind only the LUT texture.
- **`GeometryRegistry`** — Added name-based lookup APIs (`register_by_name`, `get_id`, `get_by_name`, `__contains__`) and auto-registers `SPRITE_QUAD` on init.

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