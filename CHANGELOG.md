# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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