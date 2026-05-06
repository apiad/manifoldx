# Sci-Viz Primitives v1 — Plan 3: AxisFrame + ScaleBar + colormap_legend

**Status:** Draft outline — task list pinned, full TDD bodies still to be written.

**Spec:** `.knowledge/analysis/2026-05-05-sci-viz-primitives-v1-design.md` (sections 5, 6.3, 9.2–9.4).

**Predecessors:**
- Plan 1 (foundation: PointCloud / ColormapMaterial / sprite path) — landed.
- Plan 2 (text rendering: TextLabel / LabelMaterial / atlas / label pass) — landed.

**Successors:**
- Plan 4 — functional shim API (`point_cloud()`, `axes()`, `scale_bar()`, `colormap_legend()`). Wraps the primitives this plan introduces.
- Plan 5 — visual regression infrastructure.

---

## Architecture

Three primitive families share two foundation pieces:

**Foundation**
- **Viewport uniform.** Replace the `1024.0` calibration constant in the label vertex shader with the actual viewport pixel resolution. Add `viewport_size: vec2<f32>` to the existing 208-byte `Globals` uniform (nudges it to 224 bytes — also touches every material shader's `Globals` struct).
- **Screen-anchored labels.** Make `LabelMaterial(anchor_mode="screen")` actually render in screen space (today it accepts the literal but silently falls back to world-anchored). Adds a screen-space branch to the label vertex shader; reuses the same atlas + bind group layout.

**AxisFrame** (3D world-anchored axes)
- New `AxisFrame` component (2 floats: extent, thickness).
- New `AxisMaterial` (unlit, per-batch RGBA color uniform).
- New line-rendering pipeline path with `LineList` primitive topology — the engine's first.
- Rendered alongside meshes/sprites/labels in the same render pass.

**ScaleBar** (2D screen overlay)
- New `ScaleBar` component (2 floats: length, label_id).
- Renders as a screen-anchored thick line + tick marks + length label.
- Reuses `AxisMaterial` for the bar geometry but in screen-space mode (or a dedicated `ScreenLineMaterial` — TBD in Task 12).

**colormap_legend** (2D screen overlay textured with a material's LUT)
- No new component — composed from existing primitives at the functional-shim level (Plan 4).
- BUT requires a new `ColormapLegendMaterial` (or a screen-mode of `ColormapMaterial`) that paints a 1D LUT onto a screen-anchored rectangle.
- Plan 3 ships the rendering capability; Plan 4 wires the convenience function.

---

## Sequencing

Tasks ordered by dependency, with TDD discipline: write failing test, confirm fail, implement, confirm pass, commit. Plan 2's 16-task template applies here; the count below is provisional.

### Foundation (Tasks 1–5)

| # | Task | Files |
|---|------|-------|
| 1 | Extend `Globals` uniform with `viewport_size: vec2<f32>` (208 → 224 bytes); update upload site in `RenderPipeline.run` | `renderer.py` |
| 2 | Update existing material shaders (Basic, Phong, Standard, Colormap, Label) to match the new `Globals` layout | `materials.py`, `viz/materials.py` |
| 3 | Replace `1024.0` calibration in `_LABEL_SHADER` with `globals.viewport_size.x` (world-anchored path) | `viz/materials.py` |
| 4 | Add screen-anchored branch to `_LABEL_SHADER`; gated by `material.anchor_mode == 1.0` | `viz/materials.py` |
| 5 | Integration test — `LabelMaterial(anchor_mode="screen")` renders to a fixed screen position regardless of camera angle | `tests/viz/test_label_integration.py` (extend) |

### AxisFrame (Tasks 6–11)

| # | Task | Files |
|---|------|-------|
| 6 | `AxisFrame` ECS component (Component subclass, fields: `extent: Float`, `thickness: Float`) | `viz/components.py` |
| 7 | `AxisMaterial` WGSL shader — unlit color, `LineList` vertex stage with thickness via screen-space line expansion (or kept thin via wgpu line primitive) | `viz/materials.py` |
| 8 | `AxisMaterial` Python class — uniform `(r, g, b, a)`, `pipeline_subtype=None` | `viz/materials.py` |
| 9 | Line-rendering pipeline path in `_get_or_create_pipeline` (`line=True` branch — `LineList` topology, no instancing for axes) | `renderer.py` |
| 10 | `_render_axis_pass` in `RenderPipeline` — analogous to `_render_label_pass`; one draw per axis batch | `renderer.py` |
| 11 | Integration test — spawn three axes (X red, Y green, Z blue), render, verify expected pixel colors at expected screen locations | `tests/viz/test_axis_integration.py` |

### ScaleBar (Tasks 12–14)

| # | Task | Files |
|---|------|-------|
| 12 | `ScaleBar` ECS component (`length: Float`, `label_id: Float`) | `viz/components.py` |
| 13 | Rendering — screen-anchored line + tick marks + label, reuses `AxisMaterial` (screen mode) and existing label pass | `renderer.py`, `viz/materials.py` |
| 14 | Integration test — `ScaleBar` at "bottom-left" produces non-transparent pixels in the expected screen quadrant | `tests/viz/test_scalebar_integration.py` |

### colormap_legend rendering capability (Tasks 15–17)

| # | Task | Files |
|---|------|-------|
| 15 | `ColormapLegendMaterial` (or screen mode of `ColormapMaterial`) — 1D LUT on a screen-anchored rectangle | `viz/materials.py` |
| 16 | Pipeline + render-pass support for the legend material | `renderer.py` |
| 17 | Integration test — render a legend with a viridis LUT, verify pixel sampling matches the expected colormap gradient | `tests/viz/test_legend_integration.py` |

### Wrap up (Tasks 18–19)

| # | Task | Files |
|---|------|-------|
| 18 | Run full suite + smoke-render `point_cloud_demo.py` to verify Plans 1 + 2 visuals unchanged | none modified |
| 19 | CHANGELOG entry under `[Unreleased]` for Plan 3 features + refactors | `CHANGELOG.md` |

---

## Open scope decisions (need pinning before full plan body is written)

These are the calls I want Alex to make before I expand each task into the full TDD body. None of them are deal-breakers — pick a default, change later if needed.

1. **Line topology in wgpu.** wgpu supports `LineList` natively but ignores the line-width attribute on most backends (Vulkan/Metal: width=1 always). Two options:
   - **(a) Native `LineList`** — single-pixel lines, no `thickness` control. Simple, but `AxisFrame.thickness` becomes meaningless and any thicker visual requires post-processing.
   - **(b) Quad-extrusion in shader** — vertex shader expands a unit-line quad into a screen-space-thickness rectangle. Honors `thickness`, works on all backends, but more shader complexity.
   - **My pick:** (b). Plan 1's sprite shader already does view-space billboard expansion; the precedent fits.

2. **Globals uniform growth.** Adding `viewport_size: vec2<f32>` bumps Globals from 208 to 216 bytes (or 224 with explicit pad). Every existing material shader's `Globals` struct needs to match. Touch list: Basic, Phong, Standard, Colormap, Label = 5 shaders. Mechanical update; no semantics change. Safe.

3. **`AxisMaterial` shape.** The spec says "per-batch uniform: 4 floats (RGB color + alpha), no per-instance variation". Three axes = three batches with three colors. That's clean. Alternative: per-instance color via storage buffer (one batch, three instances). Cleaner topology but more shader plumbing.
   - **My pick:** spec-canonical (three batches, per-batch color uniform).

4. **Screen-anchored material design.** Two options for how `LabelMaterial`, `AxisMaterial`, and `ColormapLegendMaterial` express their screen-mode:
   - **(a) `anchor_mode` field on each material**, branching inside the vertex shader.
   - **(b) Separate material classes** (`LabelMaterial` vs `ScreenLabelMaterial`, etc.) with shared shader source.
   - Plan 2 already chose (a) for `LabelMaterial`. Sticking with (a) keeps the surface uniform.

5. **Test coverage for the viewport uniform.** A change to `Globals` size affects every existing material. Three options for how aggressively to gate this:
   - **(a) Trust the existing material tests** — they all run under offscreen render and will catch shape mismatches.
   - **(b) Add a focused test** that asserts `Globals` is exactly 224 bytes and each material's WGSL `Globals` struct matches.
   - **My pick:** (b). Cheap to write, catches a class of bugs that's invisible until a frame renders wrong.

6. **Plan size.** 19 tasks vs Plan 2's 16. Acceptable as-is, or split into Plan 3a (foundation + AxisFrame, Tasks 1–11) and Plan 3b (ScaleBar + legend, Tasks 12–17 + wrap)?
   - Splitting gives a natural review checkpoint after AxisFrame lands. Combined keeps momentum and one CHANGELOG entry.
   - **My pick:** keep as one plan; we already proved the rhythm works on Plan 2.

---

## Out of scope for Plan 3

- **Functional shims** (`axes()`, `scale_bar()`, `colormap_legend()`) — Plan 4. This plan ships the building blocks; Plan 4 wires user-friendly factories.
- **Visual regression infrastructure** — Plan 5.
- **Antialiased lines** — wgpu doesn't support natively; deferred until a real AA pipeline is needed.
- **Per-instance per-axis colors** — uniform-only, three-batch design ships in v1; per-instance color is a follow-up if needed.
- **Multi-material legends** (two side-by-side colormaps) — single-material legend in v1.

---

## Self-review (after expanding)

(To be written when the full TDD body is fleshed out — same shape as Plan 2's self-review section.)
