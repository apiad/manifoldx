# Renderer-split refactor implementation plan

**Goal:** Split `src/manifoldx/renderer.py` (1991 lines) so each render pass lives in its own module under `src/manifoldx/render/passes/`. `RenderPipeline.render` / `RenderPipeline.run` become thin orchestrators. No behavior change — the existing test suite (392 tests) is the verification.

**Scope note:** Pure mechanical refactor. No new features, no API changes, no improved error messages. Every changed line must trace to "moved code from renderer.py into a sibling module."

**Approach:**

- `RenderPipeline` keeps its existing class structure, fields, and caches. Per-pass *bodies* move out as **module-level functions** in `src/manifoldx/render/passes/<name>.py`, taking the `RenderPipeline` instance as their first argument. This is the minimum-churn shape — no class-per-pass abstraction, no ownership reshuffle of caches.
- Pass-private helpers (`_make_label_bind_group`, `_make_sprite_bind_group`, `_get_or_create_lut_texture`, the four volume helpers) move alongside their pass.
- Cross-pass shared helpers (`_get_or_create_pipeline`, `_BatchBuffers`, `TransformCache`, `_color_hex_to_vec4`) stay in `renderer.py` for v1 of the refactor.
- After all five passes are extracted, `RenderPipeline.render` is a thin dispatcher; obsolete delegate methods are deleted.

**Verification:** `make test` after every step. Full green is the gate.

---

## File structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/manifoldx/render/__init__.py` | create | Empty package marker. |
| `src/manifoldx/render/passes/__init__.py` | create | Empty package marker. |
| `src/manifoldx/render/passes/volume.py` | create | `render_volume_pass` + four `_get_or_create_*` helpers + `_write_volume_uniforms`. |
| `src/manifoldx/render/passes/mesh.py` | create | `render_mesh_batches`. |
| `src/manifoldx/render/passes/sprite.py` | create | `render_sprite_batches` + `_make_sprite_bind_group` + `_get_or_create_lut_texture`. |
| `src/manifoldx/render/passes/label.py` | create | `render_label_pass` + `_make_label_bind_group`. |
| `src/manifoldx/render/passes/axis.py` | create | `render_axis_pass`. |
| `src/manifoldx/renderer.py` | modify | Each step: remove the moved methods, replace `render()` call sites with the imported function. |

---

## Task 1: Create the package skeleton

- [ ] Create `src/manifoldx/render/__init__.py` (empty).
- [ ] Create `src/manifoldx/render/passes/__init__.py` (empty).
- [ ] `make test` → green.
- [ ] Commit: `chore(render): add render/passes package skeleton`

## Task 2: Extract the volume pass

The volume pass is the smallest *and* the most self-contained (single fullscreen-triangle draw call, dedicated helpers).

- [ ] Move `_get_or_create_volume_pipeline`, `_get_or_create_color_lut_view`, `_get_or_create_opacity_lut_view`, `_get_or_create_volume_uniform_buffer`, `_write_volume_uniforms`, and the body of `_render_volume_pass` to `src/manifoldx/render/passes/volume.py` as module-level functions taking `rp` as first arg.
- [ ] In `renderer.py`, replace the `self._render_volume_pass(...)` call site in `render()` with `volume.render_volume_pass(self, ...)`. Delete the moved methods from `RenderPipeline`.
- [ ] `make test` → green.
- [ ] Commit: `refactor(render): extract volume pass to render/passes/volume.py`

## Task 3: Extract the mesh pass

- [ ] Move the body of `_render_mesh_batches` to `src/manifoldx/render/passes/mesh.py` as `render_mesh_batches(rp, engine, render_pass, mesh_batches, model_matrices, material_data)`.
- [ ] Update the call site in `RenderPipeline.render`. Delete the moved method.
- [ ] `make test` → green.
- [ ] Commit: `refactor(render): extract mesh pass to render/passes/mesh.py`

## Task 4: Extract the sprite pass

- [ ] Move `_render_sprite_batches`, `_make_sprite_bind_group`, `_get_or_create_lut_texture` to `src/manifoldx/render/passes/sprite.py`.
- [ ] Update the call site. Delete moved methods.
- [ ] `make test` → green.
- [ ] Commit: `refactor(render): extract sprite pass to render/passes/sprite.py`

## Task 5: Extract the label pass

- [ ] Move `_render_label_pass` and `_make_label_bind_group` to `src/manifoldx/render/passes/label.py`.
- [ ] Update the call site. Delete moved methods.
- [ ] `make test` → green.
- [ ] Commit: `refactor(render): extract label pass to render/passes/label.py`

## Task 6: Extract the axis pass

- [ ] Move `_render_axis_pass` to `src/manifoldx/render/passes/axis.py`.
- [ ] Update the call site. Delete moved method.
- [ ] `make test` → green.
- [ ] Commit: `refactor(render): extract axis pass to render/passes/axis.py`

## Task 7: Final tidy

- [ ] Verify `RenderPipeline.render` is a thin orchestrator: pass dispatch loop + per-pass function calls, no inlined pass bodies.
- [ ] Confirm `wc -l src/manifoldx/renderer.py` is dramatically smaller (target: under 800 lines, post-refactor).
- [ ] `make lint` clean. `make test` green. Run `examples/volume_demo.py --render --duration 1 --fps 15 --output /tmp/post-refactor.mp4` as a smoke test.
- [ ] Add a one-line entry under `[Unreleased] / Refactors` in CHANGELOG.md.
- [ ] Commit: `refactor(render): renderer.py shrunk to orchestrator + CHANGELOG`

---

## Risks & non-risks

- **Cache state ownership:** Caches (pipeline cache, LUT view caches) live on `RenderPipeline` instance fields. Functions read/write them through the `rp` parameter — same mutation, different syntactic access path. No semantic change.
- **Import cycles:** `render/passes/*` imports from `manifoldx.viz.materials` (already imported by `renderer.py`); `renderer.py` imports from `manifoldx.render.passes.*`. One-way dependency, no cycle.
- **Test gap:** No new tests. The full existing suite covers every pass via integration tests against an offscreen wgpu device. Each task ends with `make test` green; that's the gate.
- **Behavior change risk:** Zero, by construction. We are moving lexical scopes, not changing logic.
