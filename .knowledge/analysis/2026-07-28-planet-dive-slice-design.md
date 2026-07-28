# Planet Dive — Tier-1 slice design (CPU)

**Date:** 2026-07-28
**Status:** draft for review
**Scope:** The first vertical slice of a "fly into a planet à la No Man's Sky (minimalistic)" demo — on the CPU. Locks the *look* (procedural planet with continents + oceans, an atmosphere halo, a scripted space→surface descent) so later phases (GPU fields, near-surface detail, polish) elevate a working demo rather than build blind. Part of the planet-dive arc: **slice (this) → GPU fields (transpile `Field`→WGSL) → detail/polish**.

## Target experience

Camera starts a few planet-radii out: a lit, biome-colored sphere with a blue atmosphere halo on its limb, against a dark sky. It eases toward the surface; the sky brightens and fog thickens as it enters the atmosphere; it levels off and skims a procedural surface with oceans and mountains. Scripted, deterministic (renders to a clean MP4). No ship.

## Components

### 1. Planet mesh (reuses `modeling` + `fields`)

- `GeoMesh.icosphere(subdivisions=S, radius=R)` — `S≈6` for the slice (≈82 k verts; detail comes with GPU fields).
- **Signed terrain** for continents + oceans: a `Field` `terrain` (ridged mountains + fbm + domain-warp) remapped so ~40 % sits below zero. Displace along the normal by `amount * terrain(pos)` → land pokes out past `R`, basins sink below.
- **Biome color** via `color_by(field, Gradient)` where the driving field blends **altitude** (`fields.distance()` remapped from sea to peak) and **latitude** (`|y|/r` → polar snow). Deep→shore→grass→rock→snow gradient. *(Slope-based rock is deferred — needs a normal-derived field helper; add in the detail phase.)*
- Generated **once at startup in a `submit_process` worker** (no render-thread freeze); the slot is filled when ready.

### 2. Ocean (reuses primitives)

- A smooth `GeoMesh.icosphere(subdivisions=S-1, radius=R)` (base radius = sea level) with a glossy blue `StandardMaterial` (low roughness, small metallic). Land above `R` shows; basins below `R` read as sea. Nearly free, big flashy payoff.

### 3. Atmosphere halo — new **`AtmosphereMaterial`** (blended)

- A back-of-shell sphere at `~1.05·R`. New `Material` subclass whose fragment shader computes a **fresnel** term `f = pow(1 - max(dot(N, V), 0), falloff)` (`V` from `globals.camera_pos`), output `vec4(sky_color * intensity, f)` (or additive `f`), giving a bright limb and a near-transparent centre → the classic halo from space; envelops on descent.
- **Blending:** rendered with an alpha/additive blend state, depth-test **on**, depth-write **off**, no back-face cull — following the existing label/volume/GUI blended-pass precedent. The mesh-pass pipeline gains a small hook: a material may declare a `blend`/`pipeline_subtype == "glow"` so its pipeline is built blended + depth-write-off and drawn after the opaque planet.

### 4. Camera descent + altitude blend (demo logic; primitives exist)

- Scripted path from `~3.5·R` to skim altitude (`R + small`), easing in; `camera.set_pose(pos, target)` per frame with the target ahead along the path (tangent as it levels off).
- **Altitude** `alt = |cam| - R` drives, each frame: `engine.enable_fog(...)` density/near-far (0 in space → thick near surface), `engine.background_color` (space near-black → sky blue), and `AtmosphereMaterial.intensity`. One `lerp` by a normalized altitude.

### 5. Sun + sky

- A `DirectionalLight` → day/night terminator on the planet for free (no shadows this slice).
- Optional starfield: a sprite/point-cloud shell of white points for the "space" backdrop (nice-to-have; can be a follow-up).

## New vs reused

- **Reused:** `icosphere`/`displace`/`color_by`/`fields`, `StandardMaterial`, `enable_fog`, `background_color`, `submit_process`, `camera.set_pose`, sprites (stars).
- **New:** `AtmosphereMaterial` (fresnel glow) + the small **mesh-pass blend hook** (material-declared blend + depth-write-off). One field helper if we do latitude/altitude blend cleanly.

## Testing

- **Host:** the planet pipeline (`icosphere.displace(signed terrain).color_by(...)`) produces valid geometry (finite, in-range indices, unit normals, colors present); ocean sphere valid; `AtmosphereMaterial._compile()` contains the fresnel term and reads `globals.camera_pos`; its `pipeline_subtype`/blend flag is set.
- **GPU (gated):** a scene with planet + ocean + atmosphere renders without error; the blended atmosphere pipeline builds.
- **Visual:** render the descent to MP4; check (a) halo on the limb from orbit, (b) oceans/continents read, (c) the space→sky fog/color transition on descent, (d) no depth artifacts from the blended shell.

## Non-goals (this slice)

- GPU field evaluation (next sub-project — the `Field`→WGSL transpiler).
- Continuous LOD / near-surface high-detail patches (detail phase; the surface is a single moderate-subdiv mesh here — coarse up close is expected).
- Physically-based atmospheric scattering (the fresnel shell is a stylized stand-in), volumetric clouds.
- Interactive flight (scripted descent only), ship model.
- Slope-based biome texturing, triplanar/splat mapping.
