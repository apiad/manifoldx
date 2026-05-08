# Event-Driven System for ManifoldX — Design

**Date:** 2026-05-08
**Status:** Approved, ready for implementation planning.
**Scope:** A first-class event bus parallel to the rendering loop, supporting sync and async handlers, with built-in lifecycle events replacing the existing `startup`/`shutdown`/`update` decorators.

## Goal

A general-purpose event bus that runs at the head of every frame, parallel to (and feeding into) the existing systems / commands / render pipeline. Sync handlers for immediate reactions; async handlers for long-horizon coroutines (animations, web fetches, scene loads). Both write to "globals" (camera, lights, etc.) immediately, and to ECS only via the existing command buffer.

The motivating use case is a long-running async coroutine such as a camera dolly that zooms in, holds, zooms out, all written as a single linear `while True` loop with `await engine.delay(...)` between phases — no state machines, no per-frame bookkeeping in user code.

## Non-goals (v1)

- Input events (keyboard, mouse, resize). They will eventually flow through this same bus, but the input layer is a separate, larger design.
- Wildcards / pattern subscriptions (`engine.on('camera:*')`).
- Typed event classes / dataclass payloads. Payloads remain arbitrary Python objects.
- Priority or handler ordering beyond registration order.
- `engine.on_error` hook. Errors propagate. We add a hook only when an actual incident motivates it.
- Re-entrance modes (replace / drop / parallel). Concurrent runs of the same handler are allowed; users coordinate with their own state if needed.
- A standalone `engine.spawn_task` API.
- Per-emit cancellation handles. `engine.emit` returns `None`.

## Architecture

A new `EventBus` and a driven `asyncio` event loop both owned by the `Engine`. Both stay in `src/manifoldx/`:

```
src/manifoldx/
├── engine.py    # adds: _event_bus, _aio_loop, emit(), on(),
│                #        tick(), delay(), elapsed_at(), run_blocking()
│                # removes: startup(), shutdown(), update() decorators and their
│                #          _*_callbacks lists
└── events.py    # NEW: EventBus, Handler, ReadOnlyView, frame-waiter primitives
```

### `EventBus`

- `_handlers: dict[str, list[Handler]]` — registration order preserved.
- `_pending: list[tuple[str, Any]]` — emits queued during the current frame, drained at the start of the next frame.
- Methods: `on(event, func)`, `emit(event, payload)`, `dispatch_pending(engine)`, `dispatch_immediate(engine, event, payload)` (for `'startup'` and `'shutdown'` only).

### `Handler`

A small dataclass:

- `func: Callable`
- `is_async: bool` — `inspect.iscoroutinefunction(func)`.
- `query_components: list[str] | None` — parsed from the function's `Query[...]` annotation using the same parser the existing `engine.system` decorator uses (extracted into a shared helper in `manifoldx.systems`).

One `Handler` per `@engine.on(name)` registration. A function may be registered multiple times under different names; that yields multiple `Handler` rows.

### `ReadOnlyView`

A wrapper around the existing `ComponentView`. Forwards reads (`__getitem__`, attribute access) to the underlying view; raises `RuntimeError` on every mutation path with the message:

```
Event handlers cannot mutate ECS data directly. Use engine.commands.append(...),
engine.spawn(...), or engine.destroy(...).
```

The mutation paths to guard:

- `view[Component] = ...`
- `view[Component].field = ...`
- `view[Component].field[idx] = ...` — this last one is delegated to numpy and cannot be intercepted at the Python level. v1 documents this caveat: `ReadOnlyView` blocks the obvious mutations (assignment to the view's own attributes and items); it does not freeze numpy arrays returned from reads. A future hardening pass could call `array.setflags(write=False)` on returned slices, but that has performance implications and is deferred.

### Frame-waiter primitives

The engine maintains three structures:

- `_tick_waiters: list[asyncio.Future]` — resolved at the next frame, unconditional.
- `_delay_waiters: list[tuple[asyncio.Future, float]]` — `(future, deadline_elapsed)`. Resolved when `engine.elapsed >= deadline_elapsed`.
- `_elapsed_waiters: list[tuple[asyncio.Future, float]]` — `(future, target_elapsed)`. Same predicate as `_delay_waiters`. The split exists for ergonomics at the call site (`delay(s)` vs `elapsed_at(t)`); internally, `delay(s)` simply registers a waiter on `_delay_waiters` with `deadline = self.elapsed + s`. Both lists could be a single list of pending deadlines without changing semantics.

## Engine API additions

```python
# Emission — fire-and-forget, queued for the next frame.
engine.emit(event: str, payload: Any = None) -> None

# Registration — works for sync and async functions; same decorator.
@engine.on(event: str)
def handler(payload, query: Query[...] = None): ...

@engine.on(event: str)
async def handler(payload, query: Query[...] = None): ...

# Async waiters — only meaningful inside async handlers.
await engine.tick()                     # resolves at the next frame boundary
await engine.delay(seconds: float)      # relative wait, anchored at call time
await engine.elapsed_at(target: float)  # absolute wait against engine.elapsed

# Convenience for blocking work — escape hatch for libraries that aren't async-clean.
await engine.run_blocking(fn, *args, **kwargs)
# → loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
```

### Removed

- `engine.startup(func)` decorator.
- `engine.shutdown(func)` decorator.
- `engine.update(func)` decorator (was registered but never invoked — dead code).
- The `_startup_callbacks`, `_shutdown_callbacks`, `_update_callbacks` lists.

All callers migrate to `@engine.on('startup' | 'shutdown' | 'frame')`.

### Built-in events

| Event | Delivery | Payload | When |
|---|---|---|---|
| `'startup'` | immediate | `{}` | once, after `_init_canvas` returns and before the first frame is drawn |
| `'shutdown'` | immediate | `{}` | once, after the rendercanvas loop exits, before the asyncio loop is torn down |
| `'frame'` | inline at step 3 | `{'dt': float, 'elapsed': float, 'frame': int}` | every frame |

`'startup'` and `'shutdown'` use immediate delivery — there is no frame loop running at those moments. Async handlers registered for `'startup'` are scheduled as tasks on the asyncio loop and begin running on the first pump of frame 1; async handlers for `'shutdown'` are scheduled and pumped once before the loop is closed (see "Shutdown" below).

`'frame'` is special: it is *not* enqueued via `_pending` (which would force a one-frame delay between the engine's bookkeeping and user code seeing it). At step 3, the engine constructs the frame payload from the just-computed `dt` / `elapsed` / `_frame_index` and prepends `('frame', payload)` to the drained list, so frame handlers see the current frame's data and can write to globals before any system runs that same frame. Emits issued *from* a `'frame'` handler still go into `_pending` and fire next frame, like any other emit.

## Frame-loop integration

`Engine._draw_frame` is rewritten to this exact order:

```
1.  dt = self._compute_dt()                              # advances self.elapsed
2.  self._resolve_frame_waiters()                        # tick + delay + elapsed_at
3.  self._event_bus.dispatch_pending(self)               # drain last frame's queue
4.  self._aio_loop.run_until_complete(asyncio.sleep(0))  # pump until quiescent
5.  self.systems.run_all(self, dt)                       # unchanged
6.  self.commands.execute(self.store)                    # unchanged
7.  self._compute_runner.run_all(dt)                     # unchanged
8.  self._render_pipeline.run(self, dt) + draw           # unchanged
```

### Step 2 — resolve waiters

Iterate the three waiter lists, set the result on every future whose predicate is satisfied, and remove satisfied entries from the lists. Setting `future.set_result(None)` does *not* run the resumed coroutine — it only marks the future ready. The coroutine resumes when the loop is pumped at step 4.

Each `await engine.tick()` / `delay(s)` / `elapsed_at(t)` call creates a *fresh* future and registers it on the appropriate list. After step 2, the resolved entries are gone; if a coroutine loops with another `await engine.tick()`, that next call registers a new future for the *following* frame. The lists are not standing subscriptions — they are per-frame inboxes.

### Step 3 — dispatch pending

Move `self._event_bus._pending` into a local list (`drained = self._pending; self._pending = []`) so emits from inside handlers go into a fresh queue for the *next* frame, not the current dispatch pass. For each `(event, payload)` in the drained list:

- For each registered handler, in registration order:
  - Sync handler: build args (`payload` plus `ReadOnlyView` if `query_components`) and call inline. Exceptions propagate immediately.
  - Async handler: build the coroutine object and `self._aio_loop.create_task(coro)`. The task is registered with the loop and will run when pumped.

### Step 4 — pump

`self._aio_loop.run_until_complete(asyncio.sleep(0))` runs the loop until there are no immediately-runnable callbacks left. Coroutines whose futures were resolved at step 2 advance to their next `await`. Freshly-scheduled tasks from step 3 run their synchronous prefix to first `await` (or to completion). I/O completions detected by the loop are processed.

The pump has no time budget in v1. If a user writes a coroutine that does heavy CPU work between awaits, that coroutine stalls the frame. This is documented as the user's discipline, mirroring how a long-running system would also stall the frame.

### Emit timing rule

Emits from any phase — including from sync handlers in step 3, async handlers in step 4, systems in step 5, compute in step 7, render in step 8, and async handlers running between frames — all push into `_event_bus._pending`, which is drained on the next call to step 3. Emits never deliver in the same frame they were issued.

## Handler invocation details

```python
def _invoke_sync(handler, payload, engine):
    if handler.query_components:
        view = engine.store.get_component_view(handler.query_components, engine)
        handler.func(payload, ReadOnlyView(view))
    else:
        handler.func(payload)


def _invoke_async(handler, payload, engine):
    if handler.query_components:
        view = engine.store.get_component_view(handler.query_components, engine)
        coro = handler.func(payload, ReadOnlyView(view))
    else:
        coro = handler.func(payload)
    engine._aio_loop.create_task(coro)
```

The view passed to an async handler is constructed at dispatch time and reflects the world state at the start of the current frame. If the coroutine `await engine.tick()`s and then reads the view again, it sees the *same* view object — i.e. potentially stale references into arrays that may have been resized by the spawn / destroy machinery. This is a known caveat: long-running async handlers should re-acquire views by re-emitting or by calling a (future) `engine.store.get_component_view(...)` helper themselves. v1 documents this; a more sophisticated "live view" abstraction is out of scope.

## Error handling

- Sync handler raises → propagates out of `dispatch_pending`, out of `_draw_frame`, out of the rendercanvas loop. Engine crashes. This is intentional for v1.
- Async handler raises (other than `CancelledError`) → the exception sits on the task. When the loop is pumped at step 4 of the next frame, `run_until_complete` re-raises it. Engine crashes. Also intentional.
- `await engine.tick / delay / elapsed_at / run_blocking` raising `CancelledError` is treated as normal asyncio cancellation; user code uses `try/finally` for cleanup.

If real use surfaces a need for "log and continue" semantics, we add `engine.on_error` then. Not now.

## Shutdown

`engine.quit()` (and the equivalent path on window close) does:

1. Set `self._running = False`.
2. Dispatch `'shutdown'` immediately. Sync handlers run inline. Async `'shutdown'` handlers are scheduled as tasks.
3. Cancel every pending task on `self._aio_loop`:
   ```python
   for task in asyncio.all_tasks(loop=self._aio_loop):
       task.cancel()
   ```
4. Pump the loop one final time so `CancelledError` propagates through `try/finally` blocks in user coroutines. Any unhandled exceptions other than `CancelledError` raised during cleanup propagate.
5. Close the loop (`self._aio_loop.close()`).
6. Stop the rendercanvas event loop and close the canvas (existing logic).

## Migration sweep

The replacement of the lifecycle decorators is part of this design's implementation, not a follow-up. Sites to update:

- All examples under `examples/`: `cube.py`, `nbody.py`, `gas.py`, `boids.py`, `pbr_demo.py`, `point_cloud_demo.py`, `spheres.py`, `axes_demo.py`, `gas_compute.py`, `nbody_compute.py`, `point_cloud_compute.py`, `volume_demo.py`, `scatter_plot.py`, `smoke_demo.py`, `hello_world.py` — replace `@engine.startup` with `@engine.on("startup")`.
- Tests under `tests/` that use the lifecycle decorators — same replacement.
- New example `examples/event_dolly.py` — async camera dolly, single `while True` loop with `await engine.delay(...)` between phases (zoom in → hold → zoom out → hold → repeat).
- New example `examples/event_pulse.py` — sync handler reacting to emitted events to spawn entities via the command buffer.

## Testing

All tests run under the existing offscreen canvas backend so they pass in CI without a GPU.

- Unit (`tests/test_event_bus.py`):
  - Registration: multiple handlers per event, registration order preserved on dispatch.
  - Payload pass-through: arbitrary Python objects round-trip unchanged.
  - Queue isolation: emits during dispatch go to next frame's queue, not the current drain.
  - `ReadOnlyView` raises on `view[C] = ...`, on attribute writes through the wrapper.
- Frame-loop (`tests/test_event_frame_integration.py`):
  - Emit from a system in frame N → handler fires in frame N+1, not N.
  - Emit from a handler in frame N's drain → handler fires in frame N+1.
  - Built-in `'frame'` event fires every frame with correct payload.
  - `'startup'` fires once, before the first frame's `'frame'`.
- Async (`tests/test_event_async.py`):
  - `await engine.tick()` resolves on exactly the next frame boundary.
  - `await engine.delay(s)` resolves on the first frame whose elapsed ≥ start+s, never earlier.
  - `await engine.elapsed_at(t)` resolves on the first frame whose elapsed ≥ t.
  - Async handler raising `RuntimeError` surfaces on the next pump and propagates out of `_draw_frame`.
- Shutdown (`tests/test_event_shutdown.py`):
  - A coroutine doing `while True: await engine.tick()` is cancelled cleanly on `engine.quit()`; its `finally` runs.
  - `'shutdown'` handlers (sync and async) get a chance to run before the loop closes.

## Open caveats explicitly accepted in v1

- `ReadOnlyView` does not freeze numpy slices it returns; in-place writes via `view[C].field[i] = ...` are not intercepted. Documented.
- A long-running async handler holding a view across `await engine.tick()` may see stale array references after spawns/destroys. Documented.
- The asyncio pump has no per-frame time budget; user discipline is required. Documented.
- A handler that awaits external blocking I/O (e.g. `requests.get`) will stall the frame. The escape hatch is `await engine.run_blocking(...)`. Documented.
