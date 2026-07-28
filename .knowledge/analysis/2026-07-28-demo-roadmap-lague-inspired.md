# ManifoldX Demo Roadmap — Sebastian Lague inspired

**Date:** 2026-07-28
**Status:** living roadmap (draft for discussion)

## Why this fits

[Sebastian Lague's Coding Adventures](https://www.youtube.com/playlist?list=PLFt_AvWsXl0ehjAfLFsp1PGaatzAwo0uK) are, almost entirely: **data-driven simulation + procedural generation + volumetric/atmospheric rendering**. That is manifoldx's exact thesis — numpy-SoA ECS with vectorized physics, the Python→WGSL compute transpiler, instanced rendering, DVR volume rendering, and the new `modeling`/`fields` stack. Much of his catalog maps onto engine capabilities that *already exist* (the repo ships `nbody`, `boids`, `gas`, volume rendering, compute kernels). So these demos are a natural showcase, not a reach.

His signature projects (from [github.com/SebLague](https://github.com/SebLague?tab=repositories)): Fluid-Sim (SPH), Smoke-Simulation, Fluid-Planet, Hydraulic-Erosion, Procedural-Landmass-Generation, Marching Cubes, Clouds, Solar-System, Ant-Simulation, Slime-Simulation, Boids, Ray-Tracing, atmospheric scattering (planets), Software-Rasterizer.

## Capability legend

- **HAVE** — engine already supports it (maybe an example exists).
- **SMALL** — a demo on existing capability + a minor primitive.
- **PRIMITIVE** — needs a new reusable engine/modeling primitive first.
- **DEEP** — a genuinely hard algorithm or subsystem.

---

## Theme A — Agent & particle simulations (home turf: ECS + instancing + compute)

The engine's core strength. Vectorized numpy over entity arrays + instanced sprites.

| Demo | Lague ref | manifoldx | Notes |
|---|---|---|---|
| Boids / flocking | Boids | **HAVE** (`examples/boids.py`) | Polish + scale + trails |
| N-body / solar system | Solar-System | **HAVE** (`examples/nbody.py`) | Add orbital trails, a lit sun (bloom later) |
| Ant colony (pheromones) | Ant-Simulation | **PRIMITIVE** | Needs a compute read/write **trail map** (2D field texture) agents deposit to and sense |
| Slime / Physarum | Slime-Simulation | **PRIMITIVE** | Same trail-map primitive + a blur/decay compute pass (reaction-diffusion) |
| Reaction-diffusion / cellular automata | — | **SMALL** | Compute ping-pong on a texture; Gray-Scott, Game of Life |

**Reusable primitive this theme unlocks:** a **compute-addressable 2D field / trail texture** with ping-pong (read one, write other) + a display pass. That single primitive powers ants, slime, CA, and later fluid grids.

---

## Theme B — Fluid simulation (the flagship Lague demos)

| Demo | Lague ref | manifoldx | Notes |
|---|---|---|---|
| SPH particle fluid (2D) | Fluid-Sim | **DEEP** | Density/pressure/viscosity forces + **spatial-hash neighbor search**; numpy or compute. Render as point sprites |
| SPH fluid (3D) | Fluid-Sim | **DEEP** | 3D extension; screen-space fluid surface later |
| Smoke / gas (Eulerian grid) | Smoke-Simulation | **DEEP** | Advect a velocity+density grid; display via existing **volume rendering** |
| Fluid on a sphere | Fluid-Planet | **DEEP** | SPH constrained to a sphere surface |

**Reusable primitives:** a **spatial hash grid** (neighbor queries — also useful for boids at scale) and **SPH kernels**. This is the marquee, highest-wow track; 2D SPH first as the thin slice.

---

## Theme C — Procedural terrain & worlds (already in flight via `modeling`/`fields`)

| Demo | Lague ref | manifoldx | Notes |
|---|---|---|---|
| Heightmap terrain + streaming | Procedural-Landmass | **HAVE** (`modeling_terrain_stream.py`) | Done |
| Hydraulic erosion | Hydraulic-Erosion | **PRIMITIVE** | Droplet sim over the heightfield → eroded field; pure numpy/compute, composes with `fields` |
| Marching cubes (caves, overhangs, arches) | Marching Cubes | **PRIMITIVE** | Isosurface extraction from a 3-D scalar `Field` → `Mesh`. Unlocks caves/underwater/voxel worlds |
| Procedural planet | (planets) | **SMALL** | `icosphere.displace(field)` works *today*; add biome coloring by latitude/height/slope + water sphere |
| Biome / splat coloring | — | **SMALL** | Slope/altitude fields → gradient (extends `color_by`; a normal-derived field helper) |

**Reusable primitives:** **marching cubes** (3-D field → mesh — big unlock, connects to fluids/clouds too) and a **normal/slope field** helper.

---

## Theme D — Volumetric & atmospheric rendering (makes everything look stunning)

| Demo | Lague ref | manifoldx | Notes |
|---|---|---|---|
| Atmospheric scattering (sky/sunsets) | (planets, solar system) | **DEEP** | Raymarched Rayleigh/Mie scattering; the engine has volume-raymarch infra to build on |
| Volumetric clouds | Clouds | **DEEP** | Raymarch a 3-D worley/fbm `Field` with lighting; `fields` already generates the shapes |
| Planetary atmosphere halo | Fluid-Planet/planets | **SMALL** after scattering | Atmosphere shell around the procedural planet |

**Reusable primitive:** a **raymarched volumetric material** (density field + lighting) generalizing the current DVR. Turns clouds, smoke, atmosphere into one capability.

---

## Theme E — Rendering research (advanced / optional)

| Demo | Lague ref | manifoldx | Notes |
|---|---|---|---|
| SDF ray marching | Ray Marching | **PRIMITIVE** | Raymarch signed-distance scenes; connects to `modeling` booleans/fields |
| Compute path tracer | Ray-Tracing | **DEEP** | GPU path tracer via the compute transpiler |
| Post FX (bloom, tonemap, DoF) | — | **SMALL** | A post-process pass; makes suns/fluids/clouds pop |

---

## Proposed sequencing

Ordered by leverage of existing strengths, dependency chains, and wow-per-effort.

1. **Phase 1 — Simulation showcases.** Polish `boids`/`nbody`; build the **trail-map primitive**, then **slime** + **ant** + a reaction-diffusion demo. Fast, high-wow, showcases the ECS/compute core. *(Unlocks: compute trail texture.)*
2. **Phase 2 — Terrain deepening.** **Hydraulic erosion** (composes with `fields`, high value), then **marching cubes** (caves/overhangs), then **procedural planet**. Direct continuation of the current modeling work. *(Unlocks: marching cubes, slope fields.)*
3. **Phase 3 — SPH fluid (flagship).** **2-D SPH** thin slice (needs the **spatial hash**), then 3-D. The marquee demo. *(Unlocks: spatial hash, SPH kernels.)*
4. **Phase 4 — Volumetrics & atmosphere.** **Volumetric clouds** and **atmospheric scattering** on the raymarched volume material — makes the terrain/planet demos cinematic. *(Unlocks: volumetric raymarch material.)*
5. **Phase 5 — Rendering research + post FX.** Bloom/tonemap post pass, SDF raymarching, eventually a compute path tracer.

Each phase, like the PCG series, is expected to yield **reusable engine primitives** (trail texture, spatial hash, marching cubes, volumetric material, post-FX) — the demos are the forcing function, the primitives are the durable output.

## Open questions

- Which phase first? (Sim showcases are the fastest wins; SPH is the highest-wow but deepest.)
- Do we want a shared "demo gallery" harness (a menu to launch demos, on-screen captions) for showing these off?
- Perf target per demo (interactive vs render-only), and how much moves onto the GPU compute path vs numpy.
