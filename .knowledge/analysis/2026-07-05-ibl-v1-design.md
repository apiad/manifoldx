# IBL v1 — Image-Based Lighting Design

## Goals

- Physically-grounded ambient + specular lighting via environment maps, "good enough for paper figures" as the minimum bar.
- Additive with existing directional lights — IBL provides ambient/specular, direct lights still contribute.
- Zero-friction defaults (built-in presets) plus an escape hatch for custom env maps.
- Only `StandardMaterial` (PBR path) receives IBL. Other materials (`BasicMaterial`, `ColormapMaterial`, `VolumeMaterial`) are unaffected.

## Non-goals (v1)

- GPU-compute precomputation (CPU path is fast enough for the use case).
- Per-entity IBL occlusion or local IBL probes.
- Realtime env map updates (once set, the env map is static for the session).
- IBL for sprite / label / axis / GUI render paths.

---

## Architecture

IBL state is engine-level (like lights in `Globals`), not per-entity. The system consists of:

1. **`src/manifoldx/ibl.py`** — `EnvironmentMap` class with builders + CPU precomputation.
2. **GPU resources** — three textures uploaded once at `engine.set_environment(...)` time.
3. **Globals uniform extension** — `ibl_intensity: f32` + `ibl_enabled: u32` added to the existing struct.
4. **Mesh pass bind group 3** — new IBL texture bind group attached to the mesh pipeline.
5. **`StandardMaterial` WGSL** — split-sum IBL terms added to the fragment shader.
6. **Skybox pass** — optional render pass (before mesh pass) that draws the env map as background.

---

## `EnvironmentMap` and builders

`EnvironmentMap` is a plain Python object holding equirectangular float32 data (H×W×3, linear). It does not interact with the GPU — `engine.set_environment()` triggers upload.

```python
# Built-in presets (Python functions, no asset files)
engine.set_environment("studio")    # soft overhead studio
engine.set_environment("sky")       # blue sky + ground
engine.set_environment("neutral")   # even grey ambient
engine.set_environment("dark")      # near-black, specular highlights only

# From an HDR file (Radiance RGBE format — Poly Haven etc.)
env = EnvironmentMap.from_hdr("env.hdr")

# From any Pillow-supported image (JPEG, PNG) — equirectangular panorama
env = EnvironmentMap.from_image("panorama.jpg", exposure=1.0)

# Procedural sky gradient
env = EnvironmentMap.from_sky(
    zenith=(0.05, 0.15, 0.6),
    horizon=(0.5, 0.65, 0.9),
    ground=(0.05, 0.05, 0.05),
)

# Uniform ambient (simplest — coloured light from all directions)
env = EnvironmentMap.from_color((0.3, 0.3, 0.3))

engine.set_environment(env)
engine.set_environment(None)   # disable IBL
```

`engine.environment` returns the active `EnvironmentMap` or `None`. Two writable attrs:

```python
engine.environment.intensity = 1.0        # scales both diffuse + specular terms
engine.environment.show_skybox = False    # draw env map as scene background
```

`set_environment()` may be called at any time; it re-uploads GPU textures immediately.

Built-in presets are thin wrappers over `from_sky` / `from_color` with tuned parameters — no binary assets required.

---

## RGBE / image loading

`from_hdr` ships a pure-Python RGBE decoder (~60 lines of NumPy) in `src/manifoldx/ibl.py`. No new dependency.

`from_image` uses Pillow (already a `[viz]` dependency). sRGB input is linearised via the standard gamma decode before precomputation.

---

## GPU resources

Three textures, allocated once per `set_environment()` call:

| Texture | Format | Size | Notes |
|---|---|---|---|
| `irradiance_cubemap` | rgba16float | 64×64×6 | Diffuse irradiance per hemisphere |
| `prefiltered_cubemap` | rgba16float | 128×128×6, 8 mips (0–7) | Specular, roughness-indexed by mip level |
| `brdf_lut` | rg16float | 512×512 | Smith-GGX split-sum LUT, pre-baked, ships as asset |

`brdf_lut` is environment-independent and pre-baked offline via `scripts/gen_brdf_lut.py`, stored as `src/manifoldx/assets/ibl/brdf_lut.npy`. Loaded once at first `set_environment()` call and cached.

---

## CPU precomputation pipeline

Runs in `EnvironmentMap._precompute()`, called lazily on first GPU upload:

1. **Equirectangular → cubemap** — sample the source image at the 6 face directions, output `(6, H, W, 3)`.
2. **Irradiance convolution** — for each texel direction on a 64×64 cubemap, integrate radiance over the hemisphere with a cosine kernel (Riemann sum, ~2048 samples, <1s for 64px output).
3. **Prefiltered mip chain** — for each of 8 roughness levels (mips 0–7, roughness 0→1), GGX importance-sample the env map (1024 samples per texel) into a 128px→1px mip. Slowest step: ~2s for a 512px source map.
4. Upload all three textures to GPU via `device.create_texture` + `queue.write_texture`.

Results are cached on the `EnvironmentMap` object so repeated `set_environment()` calls with the same object skip recomputation.

---

## Globals uniform extension

Current size: 224 bytes. New fields appended:

```wgsl
struct Globals {
    // ... existing fields ...
    ibl_intensity: f32,   // +4
    ibl_enabled:   u32,   // +4
    _pad_ibl:      vec2f, // +8  → 240 bytes total
}
```

Python-side `Globals` dataclass updated in `src/manifoldx/resources.py`. All existing code that writes globals re-packs with the new fields at the tail — no alignment break.

---

## Mesh pass: bind group 3

The mesh pipeline gains a fourth bind group (IBL) attached alongside the existing three (globals, transforms, material):

```
@group(3) @binding(0) var irradiance_map:   texture_cube<f32>;
@group(3) @binding(1) var prefiltered_map:  texture_cube<f32>;
@group(3) @binding(2) var brdf_lut:         texture_2d<f32>;
@group(3) @binding(3) var env_sampler:      sampler;
```

When no environment is set, a 1×1 black cubemap and a neutral BRDF LUT are bound as placeholders so the pipeline layout stays constant — no shader recompile on `set_environment()`.

---

## StandardMaterial WGSL changes

Added to the fragment shader after the existing direct-light accumulation:

```wgsl
if globals.ibl_enabled != 0u {
    let NdotV = max(dot(N, V), 0.0001);
    let R = reflect(-V, N);
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let F  = fresnel_schlick_roughness(NdotV, F0, roughness);
    let kD = (1.0 - F) * (1.0 - metallic);

    // Diffuse irradiance
    let irradiance = textureSample(irradiance_map, env_sampler, N).rgb;
    let diffuse_ibl = kD * irradiance * albedo;

    // Specular (split sum)
    let MAX_MIPS = 7.0;  // mips 0-7, roughness 0→1 maps to mip 0→7
    let prefilt   = textureSampleLevel(prefiltered_map, env_sampler, R, roughness * MAX_MIPS).rgb;
    let brdf_samp = textureSample(brdf_lut, env_sampler, vec2f(NdotV, roughness)).rg;
    let spec_ibl  = prefilt * (F * brdf_samp.r + brdf_samp.g);

    color += (diffuse_ibl + spec_ibl) * ao * globals.ibl_intensity;
}
```

`fresnel_schlick_roughness` is a new helper (Schlick with roughness bias for IBL) added above `main`.

---

## Skybox pass

When `engine.environment.show_skybox = True`, a new pass runs before the mesh pass:

- Fullscreen triangle, no vertex buffer.
- Fragment shader reconstructs world-space direction from NDC + inverse view-proj.
- Samples `prefiltered_map` at mip 0 (sharpest env image).
- Depth output: `out.pos.z = out.pos.w` (renders at far plane) with depth-test `less-equal`, depth-write off — scene geometry always draws in front.
- Separate pipeline in the pipeline cache; does not share the mesh pipeline.

Lives in `src/manifoldx/render/passes/skybox.py`.

---

## Files changed / created

| Path | Change |
|---|---|
| `src/manifoldx/ibl.py` | New — `EnvironmentMap`, builders, RGBE decoder, CPU precomputation |
| `src/manifoldx/assets/ibl/brdf_lut.npy` | New — pre-baked 512×512 BRDF LUT |
| `scripts/gen_brdf_lut.py` | New — offline script that regenerates the LUT |
| `src/manifoldx/engine.py` | `set_environment()`, `environment` property, IBL globals wiring |
| `src/manifoldx/resources.py` | `Globals` struct extension (+16 bytes) |
| `src/manifoldx/render/passes/mesh.py` | Bind group 3 layout + IBL placeholder textures |
| `src/manifoldx/render/passes/skybox.py` | New — skybox render pass |
| `src/manifoldx/renderer.py` | Dispatch skybox pass; wire IBL bind group into mesh pass |
| `src/manifoldx/materials.py` | `StandardMaterial` WGSL: IBL fragment terms + `fresnel_schlick_roughness` helper |
| `tests/test_ibl.py` | Precomputation math tests (shape, dtype, range assertions) |
| `examples/ibl_demo.py` | New — metallic spheres at varying roughness under studio + custom env |

---

## Testing strategy

- `test_ibl.py` — unit-test each builder (shape, dtype, value range). Test irradiance output is non-negative, prefiltered chain has correct mip count, BRDF LUT loads cleanly. All CPU-only, no GPU device required.
- `examples/ibl_demo.py` — smoke-render with `--render` flag for visual regression (metallic/rough sphere grid).
- Existing `make test` suite must pass unchanged (Globals size change is the main regression risk — covered by the globals upload path).
