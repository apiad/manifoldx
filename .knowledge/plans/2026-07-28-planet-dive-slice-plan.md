# Planet Dive — Tier-1 slice — Implementation Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** A CPU `demos/planet_dive.py` — procedural planet (continents + oceans, biome color) + fresnel atmosphere halo + scripted space→surface descent. Design: `.knowledge/analysis/2026-07-28-planet-dive-slice-design.md`.

**Tech:** Python 3.13+, numpy, wgpu, `uv`. Tests: `.venv/bin/python -m pytest`.

## Global Constraints

- Additive: don't touch existing opaque/PBR paths; the "glow" pipeline branch is guarded by `material_subtype == "glow"`.
- `AtmosphereMaterial` mirrors `BasicMaterial` (unlit, `binding_slot = 0`, one `vec4` uniform → `needs_lights` is False because its shader has no `@binding(3)`).
- The glow shader declares Globals as `{vp, view, proj, camera_pos, _pad}` (offsets 0/64/128/192) so `camera_pos` is read correctly from the shared 432-byte buffer.
- Conventional commits `feat(...)`.

## Grounded symbols

- `BasicMaterial` (`resources.py`): `binding_slot=0`, `_compile()->str`, `uniform_type()->{"color":"vec4<f32>"}`, `get_data(n,reg)->(n,4) f32`. Template for `AtmosphereMaterial`.
- Renderer mesh pipeline (`renderer.py` ~1040-1055): `primitive.cull_mode=back`, `depth_stencil.depth_write_enabled=True`, `fragment.targets=[{"format": texture_format}]` — the three fields the glow hook makes conditional. `material_subtype = getattr(material,"pipeline_subtype",None)` (line ~595). `needs_lights = "@binding(3)" in shader_source` (~913).
- Modeling: `GeoMesh.icosphere(subdivisions,radius)`, `.displace(field,amount,along)`, `.color_by(field,gradient)`, `fields.*`, `Gradient`. Engine: `submit_process`, `enable_fog`, `background_color`, `camera.set_pose`, `spawn->EntityHandle`.

---

### Task 1: `AtmosphereMaterial` (fresnel glow, blended)

**Files:** `src/manifoldx/resources.py` (+ export in `__init__` if BasicMaterial is exported); test `tests/test_atmosphere_material.py`.

**Interfaces:** `AtmosphereMaterial(color, intensity=1.0)`; `pipeline_subtype == "glow"`; uniform `vec4` = `(color.rgb, intensity)`; `_compile()` → WGSL fresnel shader.

- [ ] **Step 1: failing test**
```python
# tests/test_atmosphere_material.py
import numpy as np
from manifoldx.resources import AtmosphereMaterial


def test_glow_subtype_and_shader():
    m = AtmosphereMaterial("#88bbff", intensity=1.5)
    assert m.pipeline_subtype == "glow"
    src = AtmosphereMaterial._compile()
    assert "camera_pos" in src and "@binding(3)" not in src   # unlit -> needs_lights False
    assert "pow(" in src                                       # fresnel term


def test_glow_uniform_is_rgb_intensity():
    d = AtmosphereMaterial((0.5, 0.7, 1.0), intensity=2.0).get_data(3, None)
    assert d.shape == (3, 4)
    assert np.allclose(d[0], [0.5, 0.7, 1.0, 2.0])
```
- [ ] **Step 2:** run → FAIL (no `AtmosphereMaterial`).
- [ ] **Step 3:** add to `resources.py` (after `BasicMaterial`):
```python
_ATMOSPHERE_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad: f32,
};
struct Transforms { models: array<mat4x4<f32>> };
struct GlowUniforms { params: vec4<f32> };   // rgb = colour, a = intensity

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: GlowUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    out.position = globals.vp * vec4<f32>(out.world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(globals.camera_pos - in.world_pos);
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.5);
    return vec4<f32>(material.params.rgb, fresnel * material.params.a);
}
"""


class AtmosphereMaterial(Material):
    """Unlit fresnel rim-glow (blended) — an atmosphere halo shell."""

    binding_slot = 0

    def __init__(self, color, intensity: float = 1.0):
        self.color = color
        self.intensity = intensity

    @property
    def pipeline_subtype(self):
        return "glow"

    @classmethod
    def _compile(cls) -> str:
        return _ATMOSPHERE_SHADER

    @classmethod
    def uniform_type(cls):
        return {"params": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        if isinstance(self.color, str):
            h = self.color.lstrip("#")
            rgb = [int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255]
        else:
            rgb = list(self.color[:3])
        return np.tile(np.array([*rgb, self.intensity], dtype=np.float32), (n, 1))
```
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** `git commit -m "feat(materials): AtmosphereMaterial (fresnel rim-glow, glow subtype)"`

---

### Task 2: renderer "glow" pipeline hook (blend + depth-write-off + no cull)

**Files:** `src/manifoldx/renderer.py`; test — GPU-gated render smoke (or covered by the demo render in Task 3).

- [ ] **Step 1:** in the mesh pipeline creation (~1040-1055), make the three fields conditional on `material_subtype == "glow"`:
```python
            _glow = material_subtype == "glow"
            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={...unchanged...},
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": wgpu.FrontFace.ccw,
                    "cull_mode": wgpu.CullMode.none if _glow else wgpu.CullMode.back,
                },
                depth_stencil={
                    "format": wgpu.TextureFormat.depth24plus,
                    "depth_write_enabled": not _glow,
                    "depth_compare": wgpu.CompareFunction.less,
                },
                fragment={
                    "module": shader_module,
                    "entry_point": "fs_main",
                    "targets": [_glow_target(texture_format) if _glow
                                else {"format": texture_format}],
                },
            )
```
with a small helper near the top of the pass module:
```python
def _glow_target(fmt):
    return {
        "format": fmt,
        "blend": {
            "color": {"src_factor": wgpu.BlendFactor.src_alpha,
                      "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                      "operation": wgpu.BlendOperation.add},
            "alpha": {"src_factor": wgpu.BlendFactor.one,
                      "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                      "operation": wgpu.BlendOperation.add},
        },
    }
```
(Confirm the pipeline block lives in `renderer.py`; if it's in `render/passes/mesh.py`, apply there. Match the file that owns the `~1040` block.)
- [ ] **Step 2-4:** verified by the demo render in Task 3 (a glow sphere composites as a halo, no depth artifacts). No standalone unit test (pipeline internals).
- [ ] **Step 5:** `git commit -m "feat(renderer): glow pipeline branch (alpha blend, depth-write off, no cull)"`

---

### Task 3: `demos/planet_dive.py` + biome/terrain

**Files:** create `demos/planet_dive.py`; test `tests/test_planet_dive_demo.py`.

**Build:**
- **Terrain field** (signed): `land = (fields.ridged(...)*a + fields.fbm(...)*b).warp(...)`, then `terrain = land.remap(0,1,-SEA,+PEAK)` so ~40% is below 0.
- **Planet:** `icosphere(6, R).displace(terrain_over_sphere, amount=1)` where the field is sampled at sphere positions. Color: `color_by(biome, palette)` with `biome = distance().remap(R-SEA, R+PEAK, 0, 1)` blended with latitude `coord("y")`. Generate via `engine.submit_process(build_planet)` at startup (top-level `build_planet()` returning a geometry dict; module spawn-safe under `if __name__=="__main__":`).
- **Ocean:** `icosphere(5, R)` glossy blue `StandardMaterial`.
- **Atmosphere:** `icosphere(4, R*1.05)`, `AtmosphereMaterial("#8fb8ff", intensity)`, **spawned last** (draws last → correct compositing).
- **Sun:** `DirectionalLight`.
- **Descent system:** ease camera from `3.5R` to `R+alt`; `set_pose`; altitude → `enable_fog` + `background_color` lerp (space `#05060a` → sky `#8fb8ff`).

- [ ] **Step 1: failing test** — planet + biome pipeline valid:
```python
# tests/test_planet_dive_demo.py
import numpy as np
from manifoldx.modeling import Mesh, fields, Gradient


def test_planet_pipeline_valid_with_oceans():
    R = 10.0
    land = (fields.ridged(seed=3, freq=0.3) * 0.8 + fields.fbm(seed=7, freq=0.9) * 0.2)
    terrain = land.remap(0.0, 1.0, -1.5, 2.5)           # signed: basins below 0
    planet = (Mesh.icosphere(subdivisions=4, radius=R)
              .displace(terrain, amount=1.0)
              .color_by(fields.distance().remap(R - 1.5, R + 2.5, 0.0, 1.0),
                        Gradient([(0, "#123"), (0.5, "#4a7a3a"), (1, "#fff")])))
    geo = planet.to_geometry()
    r = np.linalg.norm(geo["positions"], axis=1)
    assert r.min() < R and r.max() > R                  # basins below, mountains above
    assert "colors" in geo and np.all(np.isfinite(geo["positions"]))
    assert np.allclose(np.linalg.norm(geo["normals"], axis=1), 1.0, atol=1e-4)
```
- [ ] **Step 2:** run → should PASS (guards the pipeline). If displace-at-sphere-positions needs the field sampled at world positions, it already does (`displace` samples at `mesh.positions`).
- [ ] **Step 3:** write `demos/planet_dive.py` (spawn-safe: top-level `build_planet()` + `main()` under `__main__`). Atmosphere spawned last.
- [ ] **Step 4:** render `.venv/bin/python demos/planet_dive.py --render --duration 8 --fps 30 --output /tmp/planet.mp4`; extract frames; visually verify: (a) limb halo from orbit, (b) oceans/continents, (c) space→sky fog transition on descent. `make lint`. `make test`.
- [ ] **Step 5:** CHANGELOG (`### Features`: `AtmosphereMaterial` + glow pipeline + `demos/planet_dive.py`). Commit.

---

## Self-Review

**Coverage:** AtmosphereMaterial (T1), glow pipeline hook (T2), planet+ocean+atmosphere+descent demo (T3). **Types:** `AtmosphereMaterial` mirrors `BasicMaterial` surface; `pipeline_subtype="glow"` drives the renderer branch; uniform is a single `vec4`. **Risk:** draw order (atmosphere last), depth interactions (depth-write off) — validated visually in T3. **Deferred:** GPU fields, LOD/near-surface detail, physical scattering, slope biomes, interactive flight (all non-goals per the spec).
