# Real-time Terrain Streaming + Background Tasks — Design

**Date:** 2026-07-28
**Scope:** Make CPU-generated procedural terrain run in real time by generating large patches on a background thread and swapping them into a fixed pool of recycled mesh slots — plus the clean, reusable primitives that make the demo read well. Six small additions across `modeling`, `engine`, `renderer`, and `camera`, and a streaming flyby demo. This is the pragmatic ~35× win (1.7 → ~60 fps) before the eventual GPU-fields sub-project; it *amortizes* the CPU cost off the hot path rather than eliminating it.

## Motivation

`examples/modeling_ridge_flyby.py` regenerates the whole heightfield every frame (~560 ms of pure-numpy noise = 92% of the frame → 1.7 fps). Profiling and a thread/process benchmark (1.97× / 2.61×) show: the cost is noise evaluation, and one background worker easily keeps up (one ~0.5 s patch covers ~8.7 s of flight → worker idle ~94%). So: generate discrete world-aligned patches on a worker, draw a small fixed pool of them, and the render loop becomes GPU-bound.

The engine has **no GPU-buffer free API** and `spawn()` returns no id, so naive spawn/despawn would leak buffers. The **recycled-slot pool** sidesteps both: all patches share a resolution → identical buffer size → overwrite in place. Zero allocation/leak after startup, no despawn needed.

## Target developer code (the API we are building toward)

```python
import manifoldx as mx
from manifoldx.modeling import Mesh, fields, Gradient

engine = mx.Engine("Terrain Stream", width=1024, height=768)
engine.background_color = (0.72, 0.78, 0.86)
engine.set_sun(mx.DirectionalLight(color="#fff0dc", intensity=3.1, direction=(-0.5, -0.55, -0.65)))
engine.enable_fog(start=30, end=95)

HEIGHT, DEPTH, SPEED, K = 8.0, 52.0, 8.0, 3
TEMPLATE = Mesh.plane(width=44, depth=DEPTH, segments=160)
terrain  = (fields.ridged(seed=5, freq=0.12) * 0.85
            + fields.fbm(seed=7, freq=0.5) * 0.15).warp(1.3, fx=fields.fbm(2, 0.15), fz=fields.fbm(9, 0.15))
palette  = Gradient([(0, "#3a5f8a"), (0.16, "#4a7a3a"), (0.65, "#6e5a44"), (0.96, "#fff")])

@engine.background
def patch_at(world_z):
    return (TEMPLATE
            .displace(terrain.shift((0, 0, world_z)), amount=HEIGHT)
            .color_by(fields.coord("y").remap(0, HEIGHT, 0, 1), palette))

slots = [engine.spawn(patch_at(z).wait(),
                      material=mx.StandardMaterial("#fff", roughness=0.92, vertex_colors=True),
                      pos=(0, 0, z))
         for z in (i * DEPTH for i in range(K))]
engine.camera.look_from((0, 6.5, -5), to=(0, 2.5, 22))

st = {"next_z": K * DEPTH, "pending": None, "rear": 0}

@engine.system
def stream(dt):
    engine.camera.move_by((0, 0, SPEED * dt))
    if st["pending"] is None and engine.camera.position[2] > (st["rear"] + 1) * DEPTH:
        st["pending"] = patch_at(st["next_z"])
    if st["pending"] and st["pending"].ready:
        slot = slots[st["rear"] % K]
        slot.set_geometry(st["pending"].result)
        slot.transform.pos = (0, 0, st["next_z"])
        st["next_z"] += DEPTH; st["rear"] += 1; st["pending"] = None
```

## Components

### 1. `Field.shift(offset)` — `modeling/fields.py`

`shift(offset) -> Field`: samples the field at `p + offset`. `Field(lambda p: self(p + np.asarray(offset)))`. Lets "the patch whose near edge is at world Z" be `terrain.shift((0, 0, world_z))`. Pure, trivial, generally useful.

### 2. `engine.enable_fog(start, end, color=None)` — shader distance fog

Linear distance fog in the `StandardMaterial` fragment shader.
- `Engine.enable_fog(start, end, color=None)` stores fog params; `color` defaults to `background_color`. `engine.fog_enabled` etc.
- The `Globals` uniform gains `fog_color: vec3`, `fog_start: f32`, `fog_end: f32`, `fog_enabled: u32` (following the established shadow/IBL Globals-extension pattern; sizes updated on both the WGSL struct and the CPU packer in `renderer.py`).
- Added to the **base** `_STANDARDMATERIAL_SHADER` `fs_main` at the very end, so all variants (scalar / textured / vcolor) inherit it: after tonemap+gamma, `let f = clamp((distance(globals.camera_pos, in.world_pos) - fog_start) / (fog_end - fog_start), 0, 1) * fog_enabled; color = mix(color, fog_color, f);`. Fogging **after** tonemap makes fully-fogged fragments equal `background_color` exactly → the horizon is seamless.
- Applies to `StandardMaterial` only in v1 (Basic/Phong unaffected); fine for terrain.

### 3. `@engine.background` + `Task` — `engine.py`

- The engine owns one `concurrent.futures.ThreadPoolExecutor(max_workers=1)` (created lazily).
- `@engine.background` decorates a **pure** function (must not touch the GPU or engine state; returns plain data). Calling the decorated function submits to the executor and returns a `Task`.
- `Task`: `.ready` (bool — the future is done), `.result` (the return value; raises if not ready), `.wait()` (block until done, return the value), `.on_ready(cb)` (optional; `cb(result)` fired on the **main thread** at the next frame boundary).
- The engine drains an internal pending-`on_ready` list once per frame on the main thread (in the frame step), so callbacks and the swap-in are GPU-safe. Poll style (`.ready`/`.result` in a `@engine.system`) is already main-thread because systems run in the loop.
- One shared worker in v1; a `background_workers` Engine kwarg can raise it later.

### 4. `engine.spawn(...) -> EntityHandle` — `engine.py` / `ecs.py`

- `spawn` currently returns `None`; it will return an `EntityHandle` for the spawned entity (for `n == 1`; for `n > 1` return a list or the first — decide in plan, default: handle to the first/only). Backward-compatible (callers ignoring the return are unaffected).
- Requires `store.spawn(...)` to report the allocated entity indices (small `ecs.py` change: return the index array).
- `EntityHandle(engine, index, geometry_id)` exposes:
  - `.transform` → a small proxy; `.transform.pos = (x, y, z)` (and `.rot`, `.scale`) writes the entity's `Transform` component row at `index`.
  - `.set_geometry(mesh)` → see (5).

### 5. `EntityHandle.set_geometry(mesh)` — in-place geometry update

- Rebuilds the interleaved vertex data from `mesh.to_geometry()` in the **same layout** as the slot's existing buffer and `queue.write_buffer`s it into the slot's `vertex_buffer` (the poke, hidden behind a method).
- **Requires equal vertex count** (asserts) — this is what makes the in-place overwrite safe and allocation-free; mismatch raises a clear error.
- Layout must match how the geometry was first registered (e.g. vcolor stride 36); `set_geometry` reads the slot's stored stride/flags to interleave correctly, and requires the new mesh to carry the same attributes (has-colors etc.).
- Retrofit: rewrite the `FFD` (`modeling_ffd.py`) and current flyby buffer-poke blocks to use `handle.set_geometry(...)`, removing the `engine._geometry_registry` / `_device.queue` reach-ins from example code.

### 6. `engine.camera.look_from(pos, to)` — `camera.py`

Sugar setting `position` + `target` in one call (over the existing setter). Trivial.

## The demo — `examples/modeling_terrain_stream.py`

The target code above, fleshed out: sun + fog + a K-slot recycled pool, camera flying forward through static world-aligned patches, next patch generated on the worker and swapped in on completion. `modeling_ridge_flyby.py` stays as the naive per-frame reference for contrast.

- **Seams:** patches sample the global field at world XZ → identical boundary heights (C0). Per-patch normals leave a faint edge crease; v1 **accepts it** (distant seams are fogged). A 1-row skirt is a noted follow-up if it shows.
- **Pop-in guard:** keep one patch of lead; cap `SPEED` so `DEPTH / SPEED > gen_time` with margin (here ~8.7 s vs ~0.5 s).

## Testing

- **`Field.shift`:** `shift(o)(pts) == field(pts + o)` on a known field.
- **`@engine.background` / `Task`:** a decorated fn runs off-thread; `.wait()` returns the value; `.ready` flips true after completion; `.on_ready` fires with the result (drained on the main thread). No GPU needed.
- **`EntityHandle`:** `spawn` returns a handle; `.transform.pos =` writes the component row; `.set_geometry` with a mismatched vertex count raises; matching count updates without error (GPU-gated smoke for the actual buffer write).
- **`enable_fog`:** the compiled StandardMaterial shader contains the fog mix and reads `globals.camera_pos`; `Globals` packs the new fields at the right size (unit check on the packer); GPU-gated render smoke of a fogged scene.
- **Demo:** GPU-gated render of `modeling_terrain_stream.py`; visual check that patches stream in seamlessly and the horizon dissolves into the fog. A perf note in the docstring (target ~60 fps vs the reference demo's ~1.7).

## Non-goals

- GPU-side field evaluation (the eventual sub-project that removes the cost entirely rather than amortizing it).
- Multi-worker background pools, task cancellation/priorities (one worker, fire-and-poll v1).
- General entity lifecycle (despawn/GPU-buffer-free) — the recycled pool avoids needing it.
- Fog on non-`StandardMaterial` materials; volumetric/height fog (linear distance fog only).
- Horizontal (X) streaming or a 2-D patch grid — forward (Z) flight only in this demo.
