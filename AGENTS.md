# ManifoldX — Agent Instructions

## What is ManifoldX?

A pure-Python real-time 3D rendering engine on top of `wgpu`, with an Entity Component System (ECS) architecture. The goal is to let researchers run large-N simulations and visualize them in 3D from the same NumPy-based codebase — no graphics expertise required.

Published as `manifold-gfx` on PyPI. Beta / academic project — expect breaking changes.

Key concepts:
- **ECS with SoA layout.** Each component is a `(max_entities, total_cols)` numpy array; per-frame physics runs as vectorized numpy ops over entity arrays, no Python loops in the hot path.
- **Material = pipeline.** Each material type compiles its own WGSL shader and owns a per-batch uniform layout. The renderer batches entities by `(geometry_id, material_type[, subtype])` and issues one instanced draw call per batch.
- **Three render paths:** mesh (3D geometry + transforms), sprite (camera-facing point sprites with sphere imposters), label (camera-facing billboards textured from a shared atlas). All three coexist in one render pass.

## Quick Commands

```bash
make test           # full test suite (uses offscreen wgpu backend)
make lint           # ruff check
make format         # ruff format

# Run an example interactively
uv run python examples/<name>.py

# Smoke-render an example to MP4 (good for visual regression)
uv run python examples/<name>.py --render --duration 2 --fps 30 --output /tmp/<name>.mp4
```

## Project Structure

- **Python**: 3.13+, managed via `uv`
- **Source**: `src/manifoldx/` — `engine.py`, `ecs.py`, `components.py`, `renderer.py`, `materials.py`, `camera.py`, `resources.py`, plus `viz/` for sci-viz primitives
- **Tests**: `tests/` — top-level for engine/ECS, `tests/viz/` for sci-viz primitives
- **Examples**: `examples/` — `cube.py`, `nbody.py`, `gas.py`, `boids.py`, `pbr_demo.py`, `point_cloud_demo.py`, `spheres.py`
- **Plans**: `.knowledge/plans/` — TDD-structured implementation plans, each task split into "write failing test → confirm fail → implement → confirm pass → commit" steps
- **Changelog**: `CHANGELOG.md` — Keep-a-Changelog format, `[Unreleased]` accumulates between version bumps

## Key Conventions

- Use `uv run` for all Python invocations (`uv run pytest`, `uv run python examples/...`).
- Conventional commit messages: `feat(scope):`, `refactor(scope):`, `test(scope):`, `fix(scope):`, `docs(scope):`, `chore(scope):`.
- ECS components inherit from `manifoldx.components.Component`; dtype + shape are derived from class-level annotations (`Float`, `Vector3`, `Vector4` from `manifoldx.types`). The Engine auto-registers them on first spawn — no manual `engine.store.register_component(...)` boilerplate.
- Built-in `Transform` / `Mesh` / `Material` use explicit `register(store)` static methods — they pre-date the `Component` base and have richer `get_data` semantics that don't fit the simple-field pattern.
- Materials expose `_compile()` (returns WGSL string), `uniform_type()` (dict of field → wgsl type), optional `pipeline_subtype` (string, included in pipeline cache key).
- Tests that need a real wgpu device gate on `get_offscreen_canvas` and `pytest.skip` if the backend is unavailable, so the suite stays runnable on machines without GPU.
- Plans live in `.knowledge/plans/<YYYY-MM-DD>-<slug>.md`. Sub-projects (e.g. sci-viz primitives v1) span multiple plans (`plan-1-foundation.md`, `plan-2-text-rendering.md`, ...). Each task in a plan ends with a commit step; each sub-project ends with a CHANGELOG entry under `[Unreleased]`.

## Know-how

Procedure docs in `know-how/` encode how to do things in this repo. Match the current task against the *when to reach for it* line and load the matched doc(s) before acting. Each doc has its own `triggers` frontmatter for finer-grained matching.

- [authoring-an-ecs-component](know-how/authoring-an-ecs-component.md) — adding a new ECS component (PointCloud-style marker, ScalarValue-style scalar, Velocity-style vector, etc.) using the `Component` base class with annotation-driven dtype + shape.

(More docs will land here as patterns crystallize. Empty bullet groups belong here, not in `know-how/`.)

## Sub-projects in flight

- **Sci-viz primitives v1** — `manifoldx.viz` subpackage. Plans 1–4 landed: foundation (PointCloud + ColormapMaterial + sprite path), text rendering (TextLabel + LabelMaterial + atlas), axes/scale-bar/legend (AxisFrame + screen-anchored labels), and the Altair-style declarative shim (`mxv.points/mesh/axes/legend/scale_bar/lights` + `Channel` wrappers + `+`-composed `Chart`). Plan 5 (visual regression infrastructure) is not yet specced. Plans live at `.knowledge/plans/2026-05-*-sci-viz-primitives-v1-*.md`.
- **Event-driven engine v1** — landed. `engine.emit(...)` / `@engine.on(...)` event bus with sync + async handlers. Replaces legacy `@engine.startup`/`@engine.shutdown`/`@engine.update`.
- **Input layer v1** — landed on top of the event bus. Keyboard, mouse, wheel, resize events + Bevy-style `engine.input` polling state.
- **Volume rendering v1** — landed. DVR via fragment-shader raymarching, `Volume` component + `VolumeMaterial` with reusable colormap LUT and per-material 256-sample opacity LUT.
- **Compute systems Phase 2** — landed. Python→WGSL transpiler so kernel authors write annotated Python instead of WGSL strings. Phase 3 (multi-pass / cross-kernel deps) not yet specced.
- **GUI v1 — Plan 1 (foundation)** — landed. `manifoldx.gui` with non-interactive widgets (`Panel`/`Text`/`ValueDisplay`), stack + flex layout, theme + named-class styling, `RectMaterial` (WGSL signed-distance rounded rect), and a new `gui` render pass at the end of the render order. Plan 2 (interaction: `_GuiBridge` + `Button`/`Slider`/`Toggle` + demo) not yet started. Plans at `.knowledge/plans/2026-05-11-gui-v1-plan-*.md`. Design at `.knowledge/analysis/2026-05-10-gui-v1-design.md`.
