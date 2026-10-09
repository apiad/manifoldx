# Render quality v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The tests for each phase are written by the orchestrator and already committed before the phase is dispatched: do not weaken or delete them, make them pass.

**Goal:** No z-fighting at distance, selectable tonemap curves with exposure, FXAA, cascaded sun shadows and SSAO in manifoldx's renderer.

**Architecture:** The scene renders into an offscreen `scene_color` (swapchain format) with a reversed-Z `depth32float`; a new final pass copies it to the swapchain (with FXAA, and AO from phase 3), then the GUI draws in its own pass. Tonemapping stays inside the lit shaders (StandardMaterial, skybox) through one shared WGSL function selected by two new globals fields. Shadows become a texture array of cascades.

**Tech Stack:** Python 3.13, wgpu-py (WGSL), numpy, pytest with the offscreen canvas.

**Spec:** `.knowledge/analysis/2026-10-09-render-quality-v1-design.md` (issue #30). Read it first.

## Global Constraints

- Default rendering is unchanged unless an app opts in: `tonemap` default `("reinhard", 1.0)`, `antialias` default `None`, `cascades` default `1`, SSAO off. Reversed-Z is internal.
- Unlit materials (FlatMaterial, BasicMaterial, labels, sprites, colormaps, axis) are never tonemapped.
- The id pass (`render_frame(pass_="ids")`, `Engine._render_ids`) renders with FXAA and SSAO off and stays bit-exact.
- Python cost per frame added by a phase: under 1 ms on uh-twin's scene (cascade draws excepted and stated in CHANGELOG).
- Globals buffer stays 432 bytes in phase 1; `tonemap_mode` (u32) goes at byte 232 and `exposure` (f32) at byte 236, today's padding, inside the 240 bytes every consumer binds.
- Conventional commits; one commit per task; CHANGELOG entry under `[Unreleased]` → `### Features` per phase.
- Run only the test files a task names. The orchestrator runs `make test` in a clean copy.

## Review Focus

- A window resize after the first frame: `scene_color` and the depth texture are recreated together with the canvas size, and the final pass binds the new view (no stale bind group). Offscreen tests cannot resize; Task 3 Step 7 makes it a review item.
- An app that sets `set_tonemap` after the first frame: the next frame uses it (the globals are uploaded every frame). Covered in Task 2 by setting it between two `render_frame` calls.
- `render_frame(supersample=2)` stills with FXAA on: the final pass runs on the large target before the CPU downsample. Covered in Task 3.
- An engine with no meshes at all (empty scene, GUI only): the final pass still runs and the GUI still draws. Covered by `tests/gui/test_render_gpu.py` and the GUI test in Task 3.
- The skybox at reversed-Z: it must still be behind everything (z = 0, compare `greater_equal`). Covered by `tests/test_skybox*.py` if present, otherwise by the existing IBL tests; Task 1 adds no new one, review it by eye in a capture.

---

## Phase 1 (one PR): reversed-Z, tonemap curves, FXAA

Tests: `tests/test_render_quality_phase1.py` (committed). All 22 fail on main for the right reasons.

### Task 1: Reversed-Z depth32float

**Files:**
- Modify: `src/manifoldx/camera.py` (`get_projection_matrix`)
- Modify: `src/manifoldx/engine.py` (`_render_only`: depth format and clear; any other main-depth creation, e.g. in `_draw_still`/`_render_ids` paths if they create their own)
- Modify: `src/manifoldx/renderer.py` (depth_compare of the axis/line, label, sprite and mesh pipelines)
- Modify: `src/manifoldx/render/passes/skybox.py` (vertex z and depth_compare)
- Modify: `src/manifoldx/render/passes/volume.py` (depth_compare and the NDC unprojection)
- Test: `tests/test_render_quality_phase1.py` (the three reversed-Z tests), plus run `tests/test_camera.py tests/test_camera_project.py tests/test_volume_render.py tests/test_id_pass.py`

**Interfaces:**
- Produces: `Engine._depth_texture` is `depth32float`, cleared to 0.0; main-pass compares are `greater` / `greater_equal`. Task 3 attaches this depth texture to the GUI pass with `depth_load_op=load`.

- [ ] **Step 1: Run the reversed-Z tests and watch them fail**

Run: `uv run pytest tests/test_render_quality_phase1.py -k "projection or depth_is or planes" -v`
Expected: 3 FAIL (`0.0 == 1.0`, `depth24plus`, "2304 of 2304 pixels show the farther plane").

- [ ] **Step 2: Reverse the projection**

In `camera.py`, `get_projection_matrix`, replace the depth terms:

```python
        # Reversed Z (WebGPU depth range [0, 1]): near maps to 1, far to 0, so the
        # float depth buffer's precision sits where perspective needs it (#30).
        proj[2, 2] = near / (far - near)
        proj[2, 3] = (far * near) / (far - near)
        proj[3, 2] = -1.0
```

Update the docstring line about the formula accordingly.

- [ ] **Step 3: Main depth texture and clear**

In `engine.py`, `_render_only`, create the depth texture as `wgpu.TextureFormat.depth32float` with usage `RENDER_ATTACHMENT | TEXTURE_BINDING` (phase 3 samples it), and set `"depth_clear_value": 0.0`. Search `engine.py` for every other `depth24plus` main-depth creation and every `depth_clear_value` of a main pass and change them the same way (`grep -n "depth24plus\|depth_clear_value" src/manifoldx/engine.py`). The shadow map in `render/passes/shadow.py` stays `depth24plus`, clear 1.0, compare `less`.

- [ ] **Step 4: Flip every main-pass depth compare**

`grep -n "depth_compare" src/manifoldx/renderer.py src/manifoldx/render/passes/*.py`. In the main-pass pipelines: `less` → `greater`, `less_equal` → `greater_equal`. Leave `always` (GUI) and everything in `shadow.py`. Every `depth_stencil` state's `"format"` for a main-pass pipeline becomes `wgpu.TextureFormat.depth32float` (the GUI pipelines too, since they render against the main depth).

- [ ] **Step 5: Skybox and volume**

`skybox.py` vertex shader: `out.pos = vec4<f32>(p, 0.0, 1.0);` with the comment "z = 0 → depth 0, the far plane under reversed Z". Its pipeline compare is now `greater_equal`.

`volume.py` fragment shader: the unprojection uses NDC z 1.0 for the near point and 0.0 for the far one:

```wgsl
    let inv_vp_near = inverse_mat4(globals.vp) * vec4<f32>(ndc, 1.0, 1.0);
    let inv_vp_far  = inverse_mat4(globals.vp) * vec4<f32>(ndc, 0.0, 1.0);
```

Check the rest of the volume shader for any depth comparison against `1.0` meaning "far" and flip it.

- [ ] **Step 6: Run and fix**

Run: `uv run pytest tests/test_render_quality_phase1.py -k "projection or depth_is or planes" tests/test_camera.py tests/test_camera_project.py tests/test_volume_render.py tests/test_id_pass.py tests/test_shadow_render.py -v`
Expected: all PASS. `Camera.project` returns `w`, so it is unaffected; if a camera test pinned the old matrix values, report it instead of editing the test.

- [ ] **Step 7: Commit**

```bash
git add src/manifoldx/camera.py src/manifoldx/engine.py src/manifoldx/renderer.py src/manifoldx/render/passes/skybox.py src/manifoldx/render/passes/volume.py
git commit -m "feat(render): reversed-Z depth32float for the main pass (#30)"
```

### Task 2: Tonemap curves and exposure

**Files:**
- Modify: `src/manifoldx/engine.py` (`set_tonemap`, `tonemap` property, state in `__init__`)
- Modify: `src/manifoldx/renderer.py` (`_render_scene_passes`: write bytes 232-239 of the globals)
- Modify: `src/manifoldx/resources.py` (the `Globals` WGSL struct in StandardMaterial and any other struct that declares bytes 232-239; the shared `tonemap` function; StandardMaterial's final colour)
- Modify: `src/manifoldx/render/passes/skybox.py` (its `Globals` struct up to 240 bytes and its fragment)
- Test: `tests/test_render_quality_phase1.py` (the tonemap tests), plus run `tests/test_colour_space.py tests/test_gamma.py tests/test_shadow_render.py tests/test_globals_layout.py tests/test_fog.py tests/test_textured_material.py`

**Interfaces:**
- Produces: `Engine.set_tonemap(mode: str, exposure: float = 1.0) -> None`, `Engine.tonemap -> tuple[str, float]`, module constant `TONEMAP_MODES = {"reinhard": 0, "none": 1, "agx": 2, "aces": 3}` in `manifoldx/engine.py`. WGSL `fn tonemap(c_in: vec3<f32>, mode: u32, exposure_in: f32) -> vec3<f32>` (in `TONEMAP_WGSL`) shared by StandardMaterial and the skybox.

- [ ] **Step 1: Run the tonemap tests and watch them fail**

Run: `uv run pytest tests/test_render_quality_phase1.py -k "tonemap or reinhard or filmic or black or exposure or skybox or unlit" -v`
Expected: FAIL (`set_tonemap` / `tonemap` missing; the skybox source still has Reinhard).

- [ ] **Step 2: Engine API**

```python
TONEMAP_MODES = {"reinhard": 0, "none": 1, "agx": 2, "aces": 3}  # 0 is the default: zeroed globals keep today's look
```

In `Engine.__init__`: `self._tonemap = ("reinhard", 1.0)`.

```python
    def set_tonemap(self, mode: str = "reinhard", exposure: float = 1.0):
        """The curve lit materials map light through, and the exposure applied first.

        "reinhard" (default, c/(c+1) per channel), "agx", "aces" (Narkowicz's fit) or
        "none" (linear, clamped by the target). Unlit materials are never tonemapped.
        """
        if mode not in TONEMAP_MODES:
            raise ValueError(f"unknown tonemap {mode!r}; expected one of {sorted(TONEMAP_MODES)}")
        if not exposure > 0:
            raise ValueError(f"exposure must be positive, got {exposure!r}")
        self._tonemap = (mode, float(exposure))

    @property
    def tonemap(self):
        return self._tonemap
```

- [ ] **Step 3: Upload the two fields every frame**

In `renderer.py`, `_render_scene_passes`, where the globals bytes are filled (after the `ibl_enabled` write at 228-232):

```python
        from manifoldx.engine import TONEMAP_MODES
        mode, exposure = getattr(engine, "_tonemap", ("reinhard", 1.0))
        globals_data[232:236] = np.frombuffer(np.uint32(TONEMAP_MODES[mode]).tobytes(), dtype=np.uint8)
        globals_data[236:240] = np.frombuffer(np.float32(exposure).tobytes(), dtype=np.uint8)
```

Remove the "bytes 232-239: padding" comment.

- [ ] **Step 4: WGSL**

In every WGSL `Globals` struct that spans byte 232 (StandardMaterial's in `resources.py`, the skybox's in `skybox.py`, and any other `grep -n "_pad.*232\|offset 232" src/manifoldx`), replace the padding fields at 232 and 236 with:

```wgsl
    tonemap_mode:    u32,           // offset 232: 0 reinhard, 1 none, 2 agx, 3 aces
    exposure:        f32,           // offset 236: 0 reads as 1
```

Add one shared WGSL string constant in `resources.py`, `TONEMAP_WGSL`, included in StandardMaterial's shader and imported into the skybox's:

```wgsl
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3<f32>) -> vec3<f32> {
    // AgX base (Troy Sobotka), in Benjamin Wrensch's minimal form.
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = inset * c;
    v = clamp(log2(max(v, vec3<f32>(1e-10))), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = agx_contrast((v - vec3<f32>(min_ev)) / (max_ev - min_ev));
    v = outset * v;
    // AgX outputs display-referred (gamma 2.2) values; the sRGB target encodes, so linearise.
    return pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    // Krzysztof Narkowicz's fit.
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap(c_in: vec3<f32>, mode: u32, exposure_in: f32) -> vec3<f32> {
    let exposure = select(exposure_in, 1.0, exposure_in == 0.0);
    let c = c_in * exposure;
    switch mode {
        case 1u: { return c; }
        case 2u: { return agx(c); }
        case 3u: { return aces(c); }
        default: { return c / (c + vec3<f32>(1.0)); }
    }
}
```

StandardMaterial: replace `color = color / (color + vec3<f32>(1.0));` with `color = tonemap(color, globals.tonemap_mode, globals.exposure);`. Fog stays after it. The textured variant is generated from this shader by string replacement (`resources.py` near the textured material): make sure `TONEMAP_WGSL` is present in it too.

Skybox fragment: replace the Reinhard lines with `let mapped = tonemap(color, globals.tonemap_mode, globals.exposure);` and return it.

- [ ] **Step 5: Run and fix**

Run: `uv run pytest tests/test_render_quality_phase1.py -k "tonemap or reinhard or filmic or black or exposure or skybox or unlit" tests/test_colour_space.py tests/test_gamma.py tests/test_shadow_render.py tests/test_globals_layout.py tests/test_fog.py tests/test_textured_material.py -v`
Expected: all PASS. `test_reinhard_at_exposure_one_matches_the_old_shader` must pass exactly: with mode 0 and exposure 1 the arithmetic is the old one.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/engine.py src/manifoldx/renderer.py src/manifoldx/resources.py src/manifoldx/render/passes/skybox.py
git commit -m "feat(render): tonemap curves (reinhard, agx, aces, none) with exposure (#30)"
```

### Task 3: scene_color, the final pass, FXAA, the GUI pass

**Files:**
- Create: `src/manifoldx/render/passes/final.py` (the final pass: pipeline, bind group, draw)
- Modify: `src/manifoldx/engine.py` (`set_antialias`, `antialias` state; `_render_only`: `scene_color`, the three passes; `_render_ids`: FXAA off for the id render)
- Modify: `src/manifoldx/renderer.py` (`render` no longer draws the GUI; a new `render_gui(engine, render_pass)` does)
- Test: `tests/test_render_quality_phase1.py` (FXAA, antialias, id pass, GUI tests) plus `tests/gui/test_render_gpu.py tests/test_render_frame.py tests/test_id_pass.py tests/test_gamma.py tests/test_colour_space.py`

**Interfaces:**
- Consumes: Task 1's `depth32float` main depth.
- Produces: `Engine.set_antialias(mode: str | None) -> None` (`None` or `"fxaa"`), `Engine.antialias -> str | None`; `Engine._scene_color` / `_scene_color_view` recreated with the depth texture; `render/passes/final.py: render_final(rp, engine, encoder, src_view, dst_view, fxaa: bool, ao_view=None)`. Phase 3 passes `ao_view`.

- [ ] **Step 1: Run the FXAA tests and watch them fail**

Run: `uv run pytest tests/test_render_quality_phase1.py -k "edge or fxaa or antialias or id_pass or gui" -v`
Expected: FAIL (`set_antialias` missing).

- [ ] **Step 2: Engine API**

```python
ANTIALIAS_MODES = (None, "fxaa")
```

`__init__`: `self._antialias = None`.

```python
    def set_antialias(self, mode=None):
        """None (default) or "fxaa": FXAA over the finished scene, before the GUI."""
        if mode not in ANTIALIAS_MODES:
            raise ValueError(f"unknown antialias {mode!r}; expected None or 'fxaa'")
        self._antialias = mode

    @property
    def antialias(self):
        return self._antialias
```

- [ ] **Step 3: The offscreen scene target**

In `_render_only`, next to the depth texture (same size check, recreated together):

```python
            self._scene_color = self._device.create_texture(
                size=(tex_size[0], tex_size[1], 1),
                format=self._texture_format,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self._scene_color_view = self._scene_color.create_view()
```

The main render pass's colour attachment becomes `self._scene_color_view` (same clear value). After `self._render_pipeline.render(self, render_pass)` and `render_pass.end()`:

```python
        from manifoldx.render.passes.final import render_final
        render_final(self._render_pipeline, self, command_encoder, self._scene_color_view, texture_view,
                     fxaa=self._antialias == "fxaa")

        gui_pass = command_encoder.begin_render_pass(
            color_attachments=[{"view": texture_view, "resolve_target": None,
                                "load_op": wgpu.LoadOp.load, "store_op": wgpu.StoreOp.store}],
            depth_stencil_attachment={"view": self._depth_texture_view, "depth_load_op": wgpu.LoadOp.load,
                                      "depth_store_op": wgpu.StoreOp.store},
        )
        self._render_pipeline.render_gui(self, gui_pass)
        gui_pass.end()
```

In `renderer.py`, `render` stops calling `_gui_pass.render_gui_pass`; add:

```python
    def render_gui(self, engine, render_pass):
        """The GUI, drawn onto the finished frame after the final pass."""
        if not self._initialized or self._device is None:
            return
        from manifoldx.render.passes import gui as _gui_pass
        _gui_pass.render_gui_pass(self, engine, render_pass)
```

- [ ] **Step 4: The final pass**

`src/manifoldx/render/passes/final.py`: a fullscreen-triangle pipeline targeting `engine._texture_format`, no depth, one bind group (`scene` texture, a linear-filtering sampler, a 16-byte uniform `vec4<f32>(1/width, 1/height, fxaa, 0)`), cached on `rp` by format. The bind group is rebuilt when `src_view` changes (compare `id(src_view)`), which covers resizes.

```wgsl
struct Params { texel: vec2<f32>, fxaa: f32, _pad: f32 };
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VOut;
    o.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return o;
}

fn luma(c: vec3<f32>) -> f32 { return sqrt(dot(c, vec3<f32>(0.299, 0.587, 0.114))); }
fn tap(uv: vec2<f32>) -> vec3<f32> { return textureSampleLevel(scene, samp, uv, 0.0).rgb; }

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let m = textureLoad(scene, vec2<i32>(in.pos.xy), 0).rgb;
    if params.fxaa < 0.5 {
        return vec4<f32>(m, 1.0);  // the copy path: exact
    }
    // FXAA, Lottes' console variant (FXAA_PC_CONSOLE).
    let t = params.texel;
    let uv = in.uv;
    let l_nw = luma(tap(uv + vec2<f32>(-t.x, -t.y)));
    let l_ne = luma(tap(uv + vec2<f32>(t.x, -t.y)));
    let l_sw = luma(tap(uv + vec2<f32>(-t.x, t.y)));
    let l_se = luma(tap(uv + vec2<f32>(t.x, t.y)));
    let l_m = luma(m);
    let l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    if l_max - l_min < max(0.0312, l_max * 0.125) {
        return vec4<f32>(m, 1.0);  // no edge here: flat regions pass through untouched
    }
    var dir = vec2<f32>(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
    let reduce = max((l_nw + l_ne + l_sw + l_se) * 0.03125, 1.0 / 128.0);
    let rcp_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp_min, vec2<f32>(-8.0), vec2<f32>(8.0)) * t;
    let a = 0.5 * (tap(uv + dir * (1.0 / 3.0 - 0.5)) + tap(uv + dir * (2.0 / 3.0 - 0.5)));
    let b = a * 0.5 + 0.25 * (tap(uv - dir * 0.5) + tap(uv + dir * 0.5));
    let l_b = luma(b);
    if l_b < l_min || l_b > l_max {
        return vec4<f32>(a, 1.0);
    }
    return vec4<f32>(b, 1.0);
}
```

`render_final` begins a render pass on `dst_view` (load_op clear, black), sets the pipeline and bind group, writes the params buffer, `draw(3)`, ends. Phase 3 adds the AO multiply before the FXAA branch; leave a `// phase 3: ao` marker only if it helps, no stub code.

- [ ] **Step 5: The id pass bypasses FXAA**

In `Engine._render_ids`, where it saves and restores the environment, fog and GUI, also save `self._antialias`, set it to `None` for the id render, restore it in the same `finally`.

- [ ] **Step 6: Run and fix**

Run: `uv run pytest tests/test_render_quality_phase1.py tests/gui/test_render_gpu.py tests/test_render_frame.py tests/test_id_pass.py tests/test_gamma.py tests/test_colour_space.py tests/test_volume_render.py -v`
Expected: all PASS, including all 20 tests of `test_render_quality_phase1.py`.

- [ ] **Step 7: Review Focus checks**

`test_a_tonemap_set_between_frames_applies_to_the_next_one` and `test_supersampled_stills_run_the_final_pass_on_the_large_target` are in the phase 1 test file and must pass. A window resize cannot be exercised offscreen (`render_frame` fixes the size): make sure `_scene_color` is recreated in the same branch as the depth texture and that `render_final` rebuilds its bind group when `id(src_view)` changes, and say in your report that you did.

- [ ] **Step 8: CHANGELOG and commit**

Add under `[Unreleased]` → `### Features`:

```markdown
- **Render quality v1, phase 1.** Reversed-Z `depth32float` for the main pass (two planes
  2 cm apart at 500 m no longer swap); `engine.set_tonemap("reinhard" | "agx" | "aces" |
  "none", exposure=...)` for lit materials, default unchanged; `engine.set_antialias("fxaa")`
  in a new final pass, with the GUI drawn after it. The scene renders into an offscreen
  target first (#30).
```

```bash
git add src/manifoldx/engine.py src/manifoldx/renderer.py src/manifoldx/render/passes/final.py CHANGELOG.md
git commit -m "feat(render): offscreen scene target, final pass with FXAA, GUI after it (#30)"
```

---

## Phase 2 (one PR): cascaded shadow maps

Tests: `tests/test_render_quality_phase2.py`, written and committed by the orchestrator before this phase is dispatched (sharpness, far coverage, stability, default unchanged, as the spec's phase 2 table).

### Task 4: Cascades

**Files:**
- Modify: `src/manifoldx/engine.py` (`enable_shadows(..., cascades=1, shadow_distance=None)`)
- Modify: `src/manifoldx/render/passes/shadow.py` (texture array, one pass per cascade, per-cascade culling)
- Modify: `src/manifoldx/renderer.py` (`_sun_light_view_proj` → per-cascade fits; globals `light_view_proj[4]`, `cascade_splits`; the placeholder becomes a 1-layer array)
- Modify: `src/manifoldx/resources.py` (StandardMaterial: cascade selection, PCF in the layer, seam blend)
- Test: `tests/test_render_quality_phase2.py`, `tests/test_shadow_render.py`, `tests/test_globals_layout.py`

**Interfaces:**
- Produces: `enable_shadows(target=..., extent=..., resolution=2048, near=..., far=..., bias=0.005, pcf_radius=1, auto_fit=True, cascades=1, shadow_distance=None)`. The globals buffer grows: `light_view_proj` becomes `array<mat4x4<f32>, 4>` and `cascade_splits: vec4<f32>` is added. Pick offsets that keep every consumer's bound size consistent and update `tests/test_globals_layout.py`'s pinned size in the same commit; the orchestrator reviews that change.

- [ ] **Step 1:** Run `tests/test_render_quality_phase2.py` and watch it fail.
- [ ] **Step 2:** Splits: practical scheme, λ = 0.75, between the camera's near and `shadow_distance` (default `min(camera.far, 300.0)`); `cascade_splits[i]` is cascade i's far distance in view space.
- [ ] **Step 3:** Per cascade: the 8 corners of the camera frustum slice in world space, their bounding sphere (centre, radius); an ortho light projection of half-size `radius` looking along the sun direction from `centre - dir * (radius + scene_pad)`; snap the light-space origin to whole texels (`radius * 2 / resolution`) so sub-texel camera motion does not move shadow edges.
- [ ] **Step 4:** Shadow pass: a `depth24plus` array texture of `cascades` layers; one render pass per layer with that cascade's matrix; draw the batches whose world AABB intersects that cascade's light box (reuse `_scene_bounds`'s per-geometry local AABB cache).
- [ ] **Step 5:** StandardMaterial: view-space depth of the fragment selects the first cascade whose split exceeds it; PCF in `textureSampleCompareLevel(shadow_map, shadow_sampler, uv, layer, depth)`; over the last 10% of a cascade, blend with the next one's factor. Spot shadows keep the single-map path (`shadow_caster`).
- [ ] **Step 6:** `cascades=1` reproduces today: one layer, the whole-scene `auto_fit`. Run `tests/test_shadow_render.py` unchanged.
- [ ] **Step 7:** Run phase 2 tests, `tests/test_shadow_render.py`, `tests/test_globals_layout.py`; CHANGELOG entry with the measured Python cost of the extra shadow draws on uh-twin's scene (24 batches) if the orchestrator gives you the number, otherwise leave the number out; commit `feat(render): cascaded sun shadow maps (#30)`.

---

## Phase 3 (one PR): SSAO

Tests: `tests/test_render_quality_phase3.py`, written and committed by the orchestrator before dispatch (contact darkening, open plane unchanged, silhouettes, id pass exact).

### Task 5: SSAO

**Files:**
- Create: `src/manifoldx/render/passes/ssao.py` (AO and blur passes)
- Modify: `src/manifoldx/engine.py` (`enable_ssao(radius=1.0, intensity=1.0, samples=16)`, `disable_ssao()`; `_render_only` runs the passes between the main pass and the final pass; `_render_ids` turns SSAO off)
- Modify: `src/manifoldx/render/passes/final.py` (multiply by AO before FXAA)
- Test: `tests/test_render_quality_phase3.py`, `tests/test_id_pass.py`, `tests/test_render_quality_phase1.py`

**Interfaces:**
- Consumes: Task 1's sampleable `depth32float`; Task 3's `render_final(..., ao_view=None)`.
- Produces: `Engine.enable_ssao(radius: float = 1.0, intensity: float = 1.0, samples: int = 16)`, `Engine.disable_ssao()`, `Engine.ssao -> dict | None`.

- [ ] **Step 1:** Run `tests/test_render_quality_phase3.py` and watch it fail.
- [ ] **Step 2:** AO pass at half resolution into `r8unorm`: reconstruct view-space position from reversed-Z depth and the inverse projection; normal from `cross(dpdx(p), dpdy(p))`; `samples` hemisphere points (generated once on CPU, seeded, scaled toward the centre) rotated by a 4×4 noise texture; occlusion counts samples whose projected depth is in front, weighted by `smoothstep(0, 1, radius / abs(p.z - sample_z))`.
- [ ] **Step 3:** Bilateral 4×4 blur at half resolution, weights falling off with view-depth difference.
- [ ] **Step 4:** `render_final` multiplies `m` (and the FXAA taps) by `mix(1.0, ao, intensity)` sampled from the blurred AO with linear filtering.
- [ ] **Step 5:** `_render_ids` saves and restores the SSAO state like antialias.
- [ ] **Step 6:** Run phase 3 tests, `tests/test_id_pass.py`, `tests/test_render_quality_phase1.py`; CHANGELOG; commit `feat(render): SSAO from depth, applied in the final pass (#30)`.
