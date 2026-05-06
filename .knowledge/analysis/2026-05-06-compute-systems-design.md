# Compute systems — first-class GPU work as ECS extension

**Status:** Design — approved through brainstorming on 2026-05-06.

**Goal:** Add `Compute` as a peer to `Component` and `Material` — a base class that lets users declare per-frame GPU work in a way that integrates with the existing Command-queue ECS. Phase 1 ships the architecture with raw-WGSL bodies; Phase 2 adds a Python-as-shader DSL on top of it without touching the architecture.

**Non-goal:** A hardcoded `gpu_nbody_gravity()` function. Every existing imperative demo (nbody, gas, boids, point_cloud_demo) should be expressible as a `Compute` subclass once Phase 2 ships, with the same architecture.

---

## Locked decisions (from brainstorming)

1. **Phase split.** Phase 1: `Compute` base class with `compile() → str` returning raw WGSL. Phase 2: Phase-1's API stays unchanged; users override a `main(self, i)` Python method instead of `compile()`, and the base class's default `compile()` traces `main()` to WGSL via a code generator. The decorator-shape and class structure are identical between phases — only the kernel-body language changes.

2. **Frame ordering.** Phase-separated. Single direction of data flow per frame:

   1. CPU `@engine.system` callbacks run → enqueue Commands.
   2. CPU Commands flush — apply to numpy mirrors AND `queue.write_buffer` to GPU storage buffers (for components any registered `Compute` reads).
   3. GPU `Compute` dispatches run in registration order. Reads see the post-flush state; Writes mutate GPU buffers directly.
   4. (Phase 1) Optionally download GPU-mutated buffers back to numpy mirrors for next-frame CPU access. Skipped for `gpu_only` components.
   5. Render — reads the GPU buffers compute just wrote.

   No CPU/GPU interleaving within a frame. Mixed-mode demos (GPU physics + CPU HUD) get last-frame data on the CPU side.

3. **API shape.** Class inheritance from `Compute`. Class-level annotations declare component bindings; class-level attributes declare uniforms and dispatch parameters; `compile()` returns the WGSL string.

4. **Residency.** Components are mirrored by default (CPU numpy + GPU storage buffer) — backwards compatible with every Plan 1-4 component. Authors opt high-frequency components into `gpu_only=True`, skipping the bidirectional sync. Mirrored components pay an upload-then-download cost per frame for any compute interaction; `gpu_only` components stay on the GPU permanently.

---

## API surface

### The `Compute` base class

```python
from manifoldx.compute import Compute, Reads, Writes, ReadsWrites, Uniform


class Gravity(Compute):
    """N-body gravitational acceleration + velocity integration."""

    # Component bindings via class-level annotations.
    # Reads → storage<read>, Writes / ReadsWrites → storage<read_write>.
    positions:  Reads[Transform.pos]
    masses:     Reads[Mass.value]
    velocities: ReadsWrites[Velocity.vector]

    # Uniforms — scalars packed into a single uniform buffer.
    # Class-level defaults can be literals (constant per pipeline lifetime)
    # or sentinel symbols (re-uploaded each frame).
    G:         Uniform[float] = 20.0           # constant
    softening: Uniform[float] = 0.05           # constant
    dt:        Uniform[float] = "frame_dt"     # auto-bound; reuploaded
    n:         Uniform[int]   = "entity_count" # auto-bound; reuploaded

    # Dispatch parameters.
    workgroup_size: int = 64
    dispatch = "entity_count"   # str symbol | int | callable(engine) → int

    def compile(self) -> str:
        """Return WGSL source. Phase 1 = raw string. Phase 2 = base class
        traces `self.main` to WGSL by default; users can still override
        `compile()` for hand-tuned kernels."""
        return _GRAVITY_WGSL


# Register with engine — same pattern as engine.system(func).
engine.compute(Gravity)
```

The `Compute` instance is created lazily by the engine on first frame; `compile()` is called once and the resulting pipeline + bind group layout are cached for the engine's lifetime.

### Marker types

`Reads[X]`, `Writes[X]`, `ReadsWrites[X]`, `Uniform[T]` are subscriptable type-marker classes (the same trick `Query[T]` already uses). The X in `Reads[Transform.pos]` is a *field reference* — `Transform.pos` is a `_FieldDescriptor` that the existing `_FieldView` machinery already understands. The `Compute` metaclass walks annotations at class-creation time and builds three lists: `_reads`, `_writes` (which contains both Writes and ReadsWrites), `_uniforms`. The lists become the bind-group layout.

### `gpu_only` components

```python
class Velocity(Component, gpu_only=True):
    vector: Vector3
```

Adds a class-level flag. The Engine's component registration path materializes a GPU storage buffer for `Velocity` but never allocates a numpy mirror. CPU `@engine.system` callbacks that try to `query[Velocity]` raise a clear error. CPU Commands targeting `Velocity` raise on enqueue.

For `gpu_only=True`, the spawn path uploads initial values straight to the GPU buffer (using a single transient numpy array) and discards the CPU side immediately.

### Auto-bound uniforms

Sentinel strings (`"frame_dt"`, `"entity_count"`, `"frame_index"`) resolve at frame start to current engine state. Future engine introspection points (camera position, viewport size) follow the same pattern. Documented set; user-defined sentinels not supported in v1.

### Dispatch sizes

- `int` — fixed thread count (e.g. `dispatch = 1024`).
- `str` symbol — resolved per frame: `"entity_count"` is the alive-entity count; same registry as auto-bound uniforms.
- callable — `dispatch = lambda engine: engine.entity_count() * 2` for advanced cases.

The engine computes workgroup count = `ceil(dispatch / workgroup_size)` each frame and dispatches.

---

## Architecture

### New module: `manifoldx/compute.py`

Holds:
- `Compute` base class with `__init_subclass__` walking annotations.
- `Reads`, `Writes`, `ReadsWrites`, `Uniform` marker classes.
- `_AUTO_BOUND_UNIFORMS` registry: `{"frame_dt": lambda e: e.frame_dt, ...}`.
- `_DISPATCH_SYMBOLS` registry: `{"entity_count": lambda e: e.entity_count(), ...}`.

### Engine extension

`Engine` gains:
- `_compute_systems: list[type[Compute]]` — registered classes (not instances).
- `_compute_pipelines: dict[type[Compute], _ComputePipeline]` — cached compiled pipelines.
- `_compute_buffers: dict[str, GPUBuffer]` — per-component GPU storage buffers (lazily materialized when any Compute references the component).
- `compute(cls)` — decorator/method to register a Compute class.

`Engine.run` (the per-frame loop) gets one new step between CPU command flush and rendering:

```python
# (existing) CPU systems run.
self.systems.run(self.store, dt)

# (existing) Commands flush — apply mutations to CPU mirrors.
self.commands.flush(self.store)

# (NEW) Upload mirrored components to GPU storage buffers if any Compute reads them.
self._upload_compute_inputs()

# (NEW) Dispatch each registered Compute in registration order.
for compute_cls in self._compute_systems:
    self._dispatch_compute(compute_cls, dt)

# (NEW) Download GPU-mutated mirrored components for next-frame CPU access.
self._download_compute_outputs()

# (existing) Render.
self._render_pipeline.run(...)
```

Each `_dispatch_compute` call:
1. Looks up the cached pipeline + bind group layout for this Compute class.
2. Resolves all Reads / Writes / ReadsWrites annotations to GPU buffers (creating any missing buffer for a `gpu_only` component or upload-mirror buffer).
3. Resolves all `Uniform` values (literal or sentinel) → packs into a single uniform buffer + uploads.
4. Resolves `dispatch` → workgroup count.
5. Encodes a compute pass with `set_pipeline` + `set_bind_group` + `dispatch_workgroups(N, 1, 1)`.

### Bind group layout (single binding group, group=0)

| Binding | Resource | WGSL declaration |
|---|---|---|
| 0 | Uniform buffer (packed scalars) | `var<uniform> uniforms: Uniforms` |
| 1..K | Reads buffers | `var<storage, read> X: array<...>` |
| K+1..K+W | Writes / ReadsWrites buffers | `var<storage, read_write> Y: array<...>` |

Bindings 1..K+W are assigned in annotation declaration order. Users reference them in WGSL by the same name as the Python annotation (`positions`, `velocities`, etc.). The `compile()` method's WGSL must match this layout — Phase 2's tracer will generate it; Phase 1 users write it by hand and we provide a header-template helper for the boilerplate.

### Component → buffer materialization

When the engine sees a `Compute.{Reads,Writes,ReadsWrites}[Component.field]` annotation:

1. If `Component` is `gpu_only`: ensure GPU buffer exists; size = `max_entities × stride(field)`.
2. Else (mirrored): ensure GPU buffer exists alongside the existing numpy mirror; mark for per-frame upload (Reads) or download (Writes / ReadsWrites).

The buffer is sized for `max_entities` (matching the component array). Inside the compute shader, the user reads `n = uniforms.n` to know how many entities are alive this frame.

### Ping-pong (deferred)

Phase 1 v1: `ReadsWrites` binds the same buffer for read and write. The shader author is responsible for ordering (typical compute pattern: each thread reads its own slot + neighbors, writes its own slot — no aliasing). For algorithms that need a true read-from-old / write-to-new pattern (cellular automata, some SPH solvers), a future opt-in `ping_pong = True` class attribute will allocate two buffers and rotate them between frames. **Out of scope for the first plan.**

---

## Phase 2 preview (designed now, built later)

The Phase 2 API is **identical** to Phase 1 from the user's perspective except for the kernel body:

```python
class Gravity(Compute):
    positions:  Reads[Transform.pos]
    velocities: ReadsWrites[Velocity.vector]
    G:  Uniform[float] = 20.0
    dt: Uniform[float] = "frame_dt"
    workgroup_size: int = 64
    dispatch = "entity_count"

    def main(self, i: int):
        if i >= self.n:
            return
        accel = vec3(0.0)
        for j in range(self.n):
            if i == j:
                continue
            diff = self.positions[j] - self.positions[i]
            r = length(diff) + self.softening
            accel += self.G * self.masses[j] * diff / (r * r * r)
        self.velocities[i] += accel * self.dt
```

The base class's default `compile()` calls a code generator on `type(self).main`. The generator:
- Walks the function's AST.
- Maps Python types: `int` → `i32`, `float` → `f32`, `vec3()` → `vec3<f32>(...)`.
- Maps component-proxy reads/writes: `self.positions[j]` → `positions[j]` storage-buffer access.
- Maps Python control flow: `if`/`for`/`while` → WGSL equivalents.
- Maps a small math library: `length`, `dot`, `cross`, `normalize`, `vec3`, `mat4`, etc.
- Emits the bind-group declarations from the class's `_reads`/`_writes`/`_uniforms` lists (same source the Phase 1 layout uses).
- Wraps `main` as `@compute @workgroup_size(N) fn vs_main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; <body> }`.

The code generator is a Phase 2 design and gets its own brainstorm + spec. Phase 1 just needs to **not preclude** it: the bind-group layout, marker types, residency model, and Engine integration are all identical between phases.

---

## What Phase 1 ships

### New code

- `src/manifoldx/compute.py` (~400 lines): `Compute`, `Reads`, `Writes`, `ReadsWrites`, `Uniform`, auto-bound registry, dispatch-symbol registry.
- `src/manifoldx/engine.py` patches: `_compute_systems`, `_compute_pipelines`, `_compute_buffers`, `compute(cls)`, `_upload_compute_inputs`, `_dispatch_compute`, `_download_compute_outputs`. Maybe ~150 lines.
- `src/manifoldx/components.py` patch: `Component` base accepts `gpu_only=True` class kwarg.

### New tests

- `tests/test_compute.py` (~250 lines): `Compute` subclass annotations populate `_reads`/`_writes`/`_uniforms`; bind group layout generation; auto-bound uniforms resolve correctly; gpu_only components skip CPU mirrors.
- `tests/test_compute_integration.py` (~150 lines): an end-to-end `Compute` subclass that doubles a position component, dispatched on a 16-entity engine, verifies post-frame component values match expectation.

### One example rewrite (proof of concept)

`examples/nbody_compute.py` — the existing `nbody.py` rewritten with a `Gravity(Compute)` subclass body in raw WGSL. Should produce the same n-body cluster collapse, render at the same fps with N=500, AND scale to N=10,000 where the CPU version chokes.

### Out of scope for Phase 1

- Phase 2 Python-DSL code generator.
- Ping-pong (`ping_pong = True`).
- Multi-pass / multi-kernel within one Compute (a Compute = exactly one shader).
- Compute → CPU readback for the same frame's CPU systems (always next-frame).
- User-extensible auto-bound uniform / dispatch symbols.

---

## Self-review

- **Placeholder scan:** none.
- **Internal consistency:** Phase 1 and Phase 2 share API shape, marker types, bind-group layout, frame ordering, residency model. The only thing that differs is `compile()`'s body source. Verified.
- **Scope check:** Phase 1 = ~700 lines new code + ~400 lines tests + 1 demo. Reasonable for a 12-15 task plan. Phase 2 is an independent follow-up brainstorm.
- **Ambiguity check:** "Reads see post-flush state" is locked — CPU commands fully flush before any compute dispatches. Compute → CPU readback is "next frame at earliest" — not within the same frame. Both documented.
