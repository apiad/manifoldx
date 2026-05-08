# Event-Driven System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the event-driven system specced in `.knowledge/analysis/2026-05-08-event-driven-system-design.md` — `EventBus`, driven asyncio loop, frame-loop integration, async waiter primitives, replacement of legacy lifecycle decorators, and two new examples.

**Architecture:** A new `manifoldx.events` module holds `EventBus`, `Handler`, `ReadOnlyView`, and frame-waiter primitives. The `Engine` owns one `EventBus` and one `asyncio.AbstractEventLoop`, both pumped synchronously at the head of every frame. Handlers register via `@engine.on(name)`, fire fire-and-forget via `engine.emit(name, payload)`, and async handlers can `await engine.tick() / delay(s) / elapsed_at(t) / run_blocking(fn)`. The legacy `engine.startup / shutdown / update` decorators are removed; everything routes through `@engine.on('startup' | 'shutdown' | 'frame')`.

**Tech Stack:** Python 3.13+, `asyncio` (stdlib), `wgpu`, `pytest`, `uv`. Component view + commands plumbing already in place.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/manifoldx/events.py` | **CREATE** | `EventBus`, `Handler`, `ReadOnlyView`, `_resolve_frame_waiters` helper. ~150 lines. |
| `src/manifoldx/engine.py` | **MODIFY** | Add `_event_bus`, `_aio_loop`, public methods (`on`, `emit`, `tick`, `delay`, `elapsed_at`, `run_blocking`); rewrite `_draw_frame` to the 8-step order; remove `startup` / `shutdown` / `update` decorators and their `_*_callbacks` lists; rewire `run()` / `render()` / `quit()` to use the bus. |
| `tests/test_events_readonly_view.py` | **CREATE** | Unit tests for `ReadOnlyView`. |
| `tests/test_event_bus.py` | **CREATE** | Unit tests for `EventBus` registration / dispatch / queue isolation. |
| `tests/test_frame_waiters.py` | **CREATE** | Unit tests for the waiter resolver. |
| `tests/test_engine_event_api.py` | **CREATE** | Tests that `engine.on / emit / tick / delay / elapsed_at` exist and behave at the API surface. |
| `tests/test_event_frame_integration.py` | **CREATE** | Integration: emit timing across frames, `'frame'` payload, sync handler with `Query` view. |
| `tests/test_event_async.py` | **CREATE** | Integration: async handlers, `await engine.tick / delay / elapsed_at`, async errors. |
| `tests/test_event_shutdown.py` | **CREATE** | Integration: `'shutdown'` dispatch, task cancellation, `try/finally` cleanup. |
| `examples/event_dolly.py` | **CREATE** | Async camera dolly demo. |
| `examples/event_pulse.py` | **CREATE** | Sync handler + `engine.spawn` from event demo. |
| `examples/*.py` | **MODIFY** | Sweep: replace `@engine.startup` with `@engine.on("startup")`. |
| `tests/test_engine.py`, others | **MODIFY** | Sweep tests that touch the legacy decorators. |
| `CHANGELOG.md` | **MODIFY** | New entry under `[Unreleased]`. |

---

### Task 1: `ReadOnlyView` wrapper

**Files:**
- Create: `src/manifoldx/events.py`
- Test: `tests/test_events_readonly_view.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_events_readonly_view.py`:

```python
"""ReadOnlyView wraps a ComponentView and forbids mutation."""
import pytest
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform
from manifoldx.events import ReadOnlyView


def _engine_with_one_entity():
    engine = mx.Engine("test", width=64, height=64)
    engine.spawn(Transform(pos=(1.0, 2.0, 3.0)), n=1)
    return engine


def test_read_passes_through():
    engine = _engine_with_one_entity()
    view = engine.store.get_component_view(["Transform"], engine)
    ro = ReadOnlyView(view)
    pos = ro[Transform].pos
    assert pos.shape[1] == 3
    np.testing.assert_allclose(pos[0], [1.0, 2.0, 3.0])


def test_setitem_on_view_raises():
    engine = _engine_with_one_entity()
    view = engine.store.get_component_view(["Transform"], engine)
    ro = ReadOnlyView(view)
    with pytest.raises(RuntimeError, match="cannot mutate ECS data"):
        ro[Transform] = "anything"


def test_attribute_assignment_through_accessor_raises():
    engine = _engine_with_one_entity()
    view = engine.store.get_component_view(["Transform"], engine)
    ro = ReadOnlyView(view)
    with pytest.raises(RuntimeError, match="cannot mutate ECS data"):
        ro[Transform].pos = (9.0, 9.0, 9.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_events_readonly_view.py -v`
Expected: FAIL with "ModuleNotFoundError: manifoldx.events" or "ImportError: ReadOnlyView".

- [ ] **Step 3: Write minimal implementation**

Create `src/manifoldx/events.py`:

```python
"""Event-driven system: bus, handlers, read-only views, frame waiters."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

_RO_MUTATION_MSG = (
    "Event handlers cannot mutate ECS data directly. Use "
    "engine.commands.append(...), engine.spawn(...), or engine.destroy(...)."
)


class _ReadOnlyAccessor:
    """Wraps a ComponentAccessor to forbid attribute writes."""

    def __init__(self, accessor):
        object.__setattr__(self, "_accessor", accessor)

    def __getattr__(self, name):
        return getattr(self._accessor, name)

    def __setattr__(self, name, value):
        raise RuntimeError(_RO_MUTATION_MSG)


class ReadOnlyView:
    """Wraps a ComponentView; reads pass through, writes raise.

    Caveat: numpy arrays returned by reads are not frozen — in-place
    mutations like `view[C].field[i] = x` cannot be intercepted at the
    Python level and remain undefined behavior in event handlers.
    """

    def __init__(self, view):
        object.__setattr__(self, "_view", view)

    def __getitem__(self, component):
        accessor = self._view[component]
        return _ReadOnlyAccessor(accessor)

    def __setitem__(self, component, value):
        raise RuntimeError(_RO_MUTATION_MSG)

    def __setattr__(self, name, value):
        raise RuntimeError(_RO_MUTATION_MSG)

    def __len__(self):
        return len(self._view)

    def __iter__(self):
        return iter(self._view)

    def get_component_data(self, component_name):
        return self._view.get_component_data(component_name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_events_readonly_view.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/events.py tests/test_events_readonly_view.py
git commit -m "feat(events): ReadOnlyView wrapping ComponentView for handlers"
```

---

### Task 2: `EventBus` skeleton — registration, emit, drain

**Files:**
- Modify: `src/manifoldx/events.py`
- Test: `tests/test_event_bus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_bus.py`:

```python
"""EventBus: registration, queued dispatch, queue isolation between frames."""
from manifoldx.events import EventBus, Handler


class _DummyEngine:
    """Minimal stand-in for tests that don't need a real engine."""

    def __init__(self):
        self.store = None
        self.commands = None


def test_register_and_dispatch_sync_handler():
    bus = EventBus()
    seen = []

    @bus.on("ping")
    def handler(payload):
        seen.append(payload)

    bus.emit("ping", {"v": 1})
    assert seen == []  # not delivered yet
    bus.dispatch_pending(_DummyEngine())
    assert seen == [{"v": 1}]


def test_dispatch_order_is_registration_order():
    bus = EventBus()
    seen = []

    @bus.on("e")
    def first(payload):
        seen.append("first")

    @bus.on("e")
    def second(payload):
        seen.append("second")

    bus.emit("e", None)
    bus.dispatch_pending(_DummyEngine())
    assert seen == ["first", "second"]


def test_emits_during_dispatch_defer_to_next_frame():
    bus = EventBus()
    seen = []

    @bus.on("a")
    def a(payload):
        seen.append("a")
        bus.emit("b", None)

    @bus.on("b")
    def b(payload):
        seen.append("b")

    bus.emit("a", None)
    bus.dispatch_pending(_DummyEngine())
    assert seen == ["a"]  # b was queued, not delivered

    bus.dispatch_pending(_DummyEngine())
    assert seen == ["a", "b"]


def test_dispatch_immediate_runs_inline():
    bus = EventBus()
    seen = []

    @bus.on("phase")
    def h(payload):
        seen.append(payload)

    bus.dispatch_immediate(_DummyEngine(), "phase", {"k": 1})
    assert seen == [{"k": 1}]


def test_unregistered_event_dispatch_is_noop():
    bus = EventBus()
    bus.emit("nobody-listens", None)
    bus.dispatch_pending(_DummyEngine())  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_bus.py -v`
Expected: FAIL with "ImportError: EventBus" or "ImportError: Handler".

- [ ] **Step 3: Write minimal implementation**

Append to `src/manifoldx/events.py`:

```python
@dataclass
class Handler:
    func: Callable
    is_async: bool
    query_components: list[str] | None = None


def _parse_query_components(func) -> list[str] | None:
    """Parse Query[A, B] annotation from a function's signature.

    Returns the list of component names if found, else None.
    Mirrors the logic in Engine.system().
    """
    from manifoldx.systems import Query

    hints = getattr(func, "__annotations__", {})
    for hint in hints.values():
        if isinstance(hint, Query):
            comps = hint.components
            if not isinstance(comps, tuple):
                comps = (comps,)
            names = []
            for c in comps:
                if isinstance(c, str):
                    names.append(c)
                elif hasattr(c, "__name__"):
                    names.append(c.__name__)
            return names
    return None


class EventBus:
    """Per-frame queued event bus.

    Emits go into `_pending`; `dispatch_pending` drains and delivers them.
    `dispatch_immediate` is for the engine's own 'startup' / 'shutdown'
    phase events, which must not wait a frame.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._pending: list[tuple[str, Any]] = []

    def on(self, event: str):
        def decorator(func):
            handler = Handler(
                func=func,
                is_async=inspect.iscoroutinefunction(func),
                query_components=_parse_query_components(func),
            )
            self._handlers.setdefault(event, []).append(handler)
            return func

        return decorator

    def emit(self, event: str, payload: Any = None) -> None:
        self._pending.append((event, payload))

    def dispatch_pending(self, engine) -> None:
        drained = self._pending
        self._pending = []
        for event, payload in drained:
            self._deliver(engine, event, payload)

    def dispatch_immediate(self, engine, event: str, payload: Any = None) -> None:
        self._deliver(engine, event, payload)

    def _deliver(self, engine, event: str, payload: Any) -> None:
        for handler in self._handlers.get(event, ()):
            if handler.is_async:
                _invoke_async(handler, payload, engine)
            else:
                _invoke_sync(handler, payload, engine)


def _invoke_sync(handler: Handler, payload, engine) -> None:
    if handler.query_components:
        view = engine.store.get_component_view(handler.query_components, engine)
        handler.func(payload, ReadOnlyView(view))
    else:
        handler.func(payload)


def _invoke_async(handler: Handler, payload, engine) -> None:
    if handler.query_components:
        view = engine.store.get_component_view(handler.query_components, engine)
        coro = handler.func(payload, ReadOnlyView(view))
    else:
        coro = handler.func(payload)
    task = engine._aio_loop.create_task(coro)
    task.add_done_callback(engine._on_task_done)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_event_bus.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/events.py tests/test_event_bus.py
git commit -m "feat(events): EventBus with registration, emit, drain, immediate dispatch"
```

---

### Task 3: Frame-waiter primitives

**Files:**
- Modify: `src/manifoldx/events.py`
- Test: `tests/test_frame_waiters.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frame_waiters.py`:

```python
"""Frame waiter resolution: tick (unconditional), delay (relative), elapsed_at (absolute)."""
import asyncio
import pytest

from manifoldx.events import FrameWaiters


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def test_tick_resolves_on_next_resolve_call(loop):
    w = FrameWaiters(loop)
    fut = w.add_tick()
    assert not fut.done()
    w.resolve(elapsed=0.0)
    assert fut.done()
    assert fut.result() is None


def test_tick_list_emptied_after_resolve(loop):
    w = FrameWaiters(loop)
    w.add_tick()
    w.resolve(elapsed=0.0)
    fut = w.add_tick()
    # this future should remain unresolved until the NEXT resolve call
    assert not fut.done()


def test_delay_resolves_only_when_deadline_passed(loop):
    w = FrameWaiters(loop)
    fut = w.add_delay(seconds=0.5, current_elapsed=10.0)
    w.resolve(elapsed=10.3)
    assert not fut.done()
    w.resolve(elapsed=10.5)
    assert fut.done()


def test_elapsed_at_uses_absolute_target(loop):
    w = FrameWaiters(loop)
    fut = w.add_elapsed_at(target=42.0)
    w.resolve(elapsed=41.9)
    assert not fut.done()
    w.resolve(elapsed=42.0)
    assert fut.done()


def test_resolve_keeps_unsatisfied_waiters(loop):
    w = FrameWaiters(loop)
    fut_short = w.add_delay(seconds=0.1, current_elapsed=0.0)
    fut_long = w.add_delay(seconds=10.0, current_elapsed=0.0)
    w.resolve(elapsed=0.2)
    assert fut_short.done()
    assert not fut_long.done()
    w.resolve(elapsed=10.1)
    assert fut_long.done()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_frame_waiters.py -v`
Expected: FAIL with "ImportError: FrameWaiters".

- [ ] **Step 3: Write minimal implementation**

Append to `src/manifoldx/events.py`:

```python
class FrameWaiters:
    """Per-frame inboxes for tick / delay / elapsed_at futures.

    Each call to add_* registers a fresh future on the appropriate list.
    `resolve(elapsed)` walks the lists, sets futures whose predicates hold,
    and removes them from their lists. The lists are not standing
    subscriptions — coroutines that loop must call add_* again next frame.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._tick: list[asyncio.Future] = []
        self._deadlines: list[tuple[asyncio.Future, float]] = []  # delay + elapsed_at share storage

    def add_tick(self) -> asyncio.Future:
        fut = self._loop.create_future()
        self._tick.append(fut)
        return fut

    def add_delay(self, seconds: float, current_elapsed: float) -> asyncio.Future:
        fut = self._loop.create_future()
        self._deadlines.append((fut, current_elapsed + seconds))
        return fut

    def add_elapsed_at(self, target: float) -> asyncio.Future:
        fut = self._loop.create_future()
        self._deadlines.append((fut, target))
        return fut

    def resolve(self, elapsed: float) -> None:
        # tick waiters resolve unconditionally
        for fut in self._tick:
            if not fut.done():
                fut.set_result(None)
        self._tick.clear()

        # deadline waiters: resolve those whose deadline is reached
        kept: list[tuple[asyncio.Future, float]] = []
        for fut, deadline in self._deadlines:
            if elapsed >= deadline:
                if not fut.done():
                    fut.set_result(None)
            else:
                kept.append((fut, deadline))
        self._deadlines = kept

    def cancel_all(self) -> None:
        """Cancel every outstanding future (used at shutdown)."""
        for fut in self._tick:
            if not fut.done():
                fut.cancel()
        for fut, _ in self._deadlines:
            if not fut.done():
                fut.cancel()
        self._tick.clear()
        self._deadlines.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_frame_waiters.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/events.py tests/test_frame_waiters.py
git commit -m "feat(events): FrameWaiters for tick/delay/elapsed_at futures"
```

---

### Task 4: Wire `EventBus` + asyncio loop onto `Engine`

**Files:**
- Modify: `src/manifoldx/engine.py:20-90` (`__init__`); add new public methods near existing decorators.
- Test: `tests/test_engine_event_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_event_api.py`:

```python
"""Engine-level event API: on, emit, tick, delay, elapsed_at, run_blocking."""
import asyncio
import inspect

import manifoldx as mx


def _engine():
    return mx.Engine("test", width=64, height=64)


def test_engine_has_event_bus_and_loop():
    e = _engine()
    assert e._event_bus is not None
    assert isinstance(e._aio_loop, asyncio.AbstractEventLoop)


def test_engine_on_emit_round_trip():
    e = _engine()
    seen = []

    @e.on("ping")
    def handler(payload):
        seen.append(payload)

    e.emit("ping", "hello")
    assert seen == []
    e._event_bus.dispatch_pending(e)
    assert seen == ["hello"]


def test_engine_tick_returns_awaitable_future():
    e = _engine()
    fut = e.tick()
    assert isinstance(fut, asyncio.Future)
    assert not fut.done()


def test_engine_delay_returns_future_with_correct_deadline():
    e = _engine()
    e.elapsed = 5.0
    fut = e.delay(2.0)
    assert isinstance(fut, asyncio.Future)
    e._frame_waiters.resolve(elapsed=6.9)
    assert not fut.done()
    e._frame_waiters.resolve(elapsed=7.0)
    assert fut.done()


def test_engine_elapsed_at_uses_absolute_target():
    e = _engine()
    fut = e.elapsed_at(target=12.5)
    e._frame_waiters.resolve(elapsed=12.4)
    assert not fut.done()
    e._frame_waiters.resolve(elapsed=12.5)
    assert fut.done()


def test_engine_on_decorator_returns_func():
    e = _engine()

    @e.on("e")
    def h(payload):
        pass

    assert callable(h)


def test_engine_run_blocking_is_coroutine_function():
    e = _engine()
    assert inspect.iscoroutinefunction(e.run_blocking)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_event_api.py -v`
Expected: FAIL with `AttributeError: 'Engine' object has no attribute '_event_bus'` (or similar).

- [ ] **Step 3: Modify `src/manifoldx/engine.py`**

Edit `__init__` to add the bus, loop, and waiters. Find:

```python
        self._update_callbacks = []

        # === ECS Infrastructure ===
```

Replace with:

```python
        # Event-driven system (replaces _startup/_shutdown/_update_callbacks)
        from manifoldx.events import EventBus, FrameWaiters

        self._event_bus = EventBus()
        self._aio_loop = asyncio.new_event_loop()
        self._frame_waiters = FrameWaiters(self._aio_loop)
        # Task error spool — populated by add_done_callback when an async
        # handler raises (other than CancelledError). Drained by
        # _pump_aio_loop, which re-raises the first error per the v1
        # "errors crash the engine" policy.
        self._task_errors: list[BaseException] = []

        # === ECS Infrastructure ===
```

Also remove the three lines:

```python
        self._startup_callbacks = []
        self._shutdown_callbacks = []
        self._update_callbacks = []
```

(Replace them by the block above — they are gone.)

Then replace the `startup`, `shutdown`, `update`, `system` decorators block. Find the existing block:

```python
    def startup(self, func):
        self._startup_callbacks.append(func)
        return func

    def shutdown(self, func):
        self._shutdown_callbacks.append(func)
        return func

    def update(self, func):
        self._update_callbacks.append(func)
        return func

    def system(self, func):
```

Replace with:

```python
    def on(self, event: str):
        """Register a sync or async handler for an event.

        Usage:
            @engine.on("startup")
            def init(payload): ...

            @engine.on("frame")
            async def each_frame(payload): ...
        """
        return self._event_bus.on(event)

    def emit(self, event: str, payload=None) -> None:
        """Queue an event for delivery at the start of the next frame."""
        self._event_bus.emit(event, payload)

    def tick(self):
        """Return a future that resolves at the next frame boundary."""
        return self._frame_waiters.add_tick()

    def delay(self, seconds: float):
        """Return a future that resolves after `seconds` of engine.elapsed."""
        return self._frame_waiters.add_delay(seconds, self.elapsed)

    def elapsed_at(self, target: float):
        """Return a future that resolves once engine.elapsed >= target."""
        return self._frame_waiters.add_elapsed_at(target)

    async def run_blocking(self, fn, *args, **kwargs):
        """Run a blocking callable in the default executor and await its result."""
        import functools

        return await self._aio_loop.run_in_executor(
            None, functools.partial(fn, *args, **kwargs)
        )

    def _on_task_done(self, task) -> None:
        """Done-callback wired onto every async-handler task.

        Called synchronously when the task completes. We surface non-cancel
        exceptions into _task_errors so _pump_aio_loop can re-raise them on
        the next pump (per the v1 "errors crash the engine" policy).
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._task_errors.append(exc)

    def system(self, func):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine_event_api.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/engine.py tests/test_engine_event_api.py
git commit -m "feat(events): Engine.on/emit/tick/delay/elapsed_at/run_blocking, drop legacy decorators"
```

---

### Task 5: Rewrite `_draw_frame` to the 8-step order

**Files:**
- Modify: `src/manifoldx/engine.py:350-380` (`_draw_frame`)
- Test: `tests/test_event_frame_integration.py`

This task only handles the deterministic, non-async parts of the new frame loop: waiter resolution, pending dispatch, and the inline `'frame'` event. Async handler scheduling + loop pump arrives in Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_frame_integration.py`:

```python
"""Frame-loop integration: emit timing, 'frame' event, sync handler with Query."""
import pytest


def _make_offscreen_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_frame_event_fires_every_frame_with_payload():
    engine = _make_offscreen_engine()
    seen = []

    @engine.on("frame")
    def each(payload):
        seen.append(dict(payload))

    engine._draw_frame()
    engine._draw_frame()
    engine._draw_frame()

    assert len(seen) == 3
    assert seen[0]["frame"] == 0
    assert seen[1]["frame"] == 1
    assert seen[2]["frame"] == 2
    assert all(set(p.keys()) == {"dt", "elapsed", "frame"} for p in seen)


def test_emit_from_handler_defers_one_frame():
    engine = _make_offscreen_engine()
    log = []

    @engine.on("a")
    def on_a(payload):
        log.append("a")
        engine.emit("b")

    @engine.on("b")
    def on_b(payload):
        log.append("b")

    engine.emit("a")
    engine._draw_frame()
    assert log == ["a"]  # 'b' queued by 'a' has not yet fired
    engine._draw_frame()
    assert log == ["a", "b"]


def test_emit_from_system_fires_next_frame():
    from manifoldx.systems import Query
    from manifoldx.components import Transform

    engine = _make_offscreen_engine()
    engine.spawn(Transform(pos=(0, 0, 0)), n=1)
    fired = []

    @engine.system
    def s(query: Query[Transform], dt: float):
        engine.emit("from_system", None)

    @engine.on("from_system")
    def on_emit(payload):
        fired.append(True)

    engine._draw_frame()
    assert fired == []  # not delivered same frame
    engine._draw_frame()
    assert fired == [True]


def test_handler_with_query_view_is_readonly():
    from manifoldx.events import ReadOnlyView
    from manifoldx.systems import Query
    from manifoldx.components import Transform

    engine = _make_offscreen_engine()
    engine.spawn(Transform(pos=(1, 2, 3)), n=1)

    captured = {}

    @engine.on("inspect")
    def on_inspect(payload, view: Query[Transform]):
        captured["view"] = view
        captured["pos0"] = view[Transform].pos[0].copy()

    engine.emit("inspect")
    engine._draw_frame()
    assert isinstance(captured["view"], ReadOnlyView)
    assert tuple(captured["pos0"]) == (1.0, 2.0, 3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_frame_integration.py -v`
Expected: failures because `'frame'` is not yet emitted by `_draw_frame` and the new dispatch-pending step is not yet wired.

- [ ] **Step 3: Modify `_draw_frame`**

In `src/manifoldx/engine.py`, find the existing body of `_draw_frame`. Replace its core (between the early-exit checks and the canvas-texture acquisition) so the order matches the spec.

Find:

```python
        dt = self._compute_dt()

        # 1. Clear command buffer for this frame
        self.commands.clear()

        # 2. Run all user systems (they emit commands)
        self.systems.run_all(self, dt)

        # 3. Execute command buffer (apply all spawn/destroy/update)
        self.commands.execute(self.store)

        # 3b. Dispatch GPU compute systems (after CPU flush, before render).
        self._compute_runner.run_all(dt)
        self._frame_index += 1

        # 4. RENDER PIPELINE
        self._render_pipeline.run(self, dt)
```

Replace with:

```python
        dt = self._compute_dt()
        self._last_dt = dt

        # Step 2: resolve frame waiters (tick / delay / elapsed_at)
        self._frame_waiters.resolve(self.elapsed)

        # Clear command buffer ONCE at the head of the frame so events,
        # async handlers, and systems all contribute to the same buffer
        # that gets flushed at step 6.
        self.commands.clear()

        # Step 3: drain pending events (frame N-1's emits + this frame's 'frame')
        frame_payload = {
            "dt": dt,
            "elapsed": self.elapsed,
            "frame": self._frame_index,
        }
        # Inject the inline 'frame' event ahead of the user-emitted queue,
        # so frame handlers see CURRENT-frame data (not last-frame's).
        self._event_bus._pending.insert(0, ("frame", frame_payload))
        self._event_bus.dispatch_pending(self)

        # Step 4: pump asyncio loop (async handlers + waiter wakers).
        # Wrapped in a sleep(0) trampoline so the loop processes ready
        # callbacks then returns. (Filled in by Task 6.)
        # PLACEHOLDER UNTIL TASK 6 LANDS:
        # self._aio_loop.run_until_complete(asyncio.sleep(0))

        # Step 5: run user systems (may emit commands).
        self.systems.run_all(self, dt)

        # Step 6: flush command buffer (events + handlers + systems).
        self.commands.execute(self.store)

        # Step 7: GPU compute systems.
        self._compute_runner.run_all(dt)
        self._frame_index += 1

        # Step 8: render pipeline.
        self._render_pipeline.run(self, dt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_event_frame_integration.py -v`
Expected: 4 passed.

- [ ] **Step 5: Verify nothing else broke**

Run: `uv run pytest tests/ -x --ignore=tests/test_event_async.py --ignore=tests/test_event_shutdown.py 2>&1 | tail -30`
Expected: all currently-existing tests pass except those that touch `engine.startup` / `engine.shutdown` / `engine.update` (those are addressed in Task 9).
If the test failure mode is **only** `AttributeError: 'Engine' object has no attribute 'startup'`, that is expected and addressed by Task 9.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/engine.py tests/test_event_frame_integration.py
git commit -m "feat(events): rewrite _draw_frame to 8-step order with waiters + inline 'frame'"
```

---

### Task 6: Async handlers + asyncio loop pump

**Files:**
- Modify: `src/manifoldx/engine.py:_draw_frame` (uncomment the loop pump).
- Test: `tests/test_event_async.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_async.py`:

```python
"""Async handlers, await engine.tick / delay / elapsed_at, error propagation."""
import pytest


def _make_offscreen_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_async_handler_runs_to_first_await_on_dispatch_frame():
    engine = _make_offscreen_engine()
    log = []

    @engine.on("go")
    async def go(payload):
        log.append("before")
        await engine.tick()
        log.append("after")

    engine.emit("go")
    engine._draw_frame()
    assert log == ["before"]
    engine._draw_frame()
    assert log == ["before", "after"]


def test_await_delay_resolves_after_seconds_elapsed():
    """Drive frames manually with a fixed timestep, advancing engine.elapsed
    between frames (since _compute_dt does not advance it in fixed-dt mode).
    """
    engine = _make_offscreen_engine()
    engine.set_fixed_timestep(0.1)  # so _compute_dt doesn't overwrite elapsed
    log = []

    @engine.on("start")
    async def start(payload):
        await engine.delay(0.5)
        log.append("done")

    engine.emit("start")
    engine.elapsed = 0.0
    engine._draw_frame()  # dispatch + first await; deadline set to 0.5
    assert log == []

    for tick in range(1, 11):
        engine.elapsed = tick * 0.1
        engine._draw_frame()
        if log:
            break

    assert log == ["done"]
    # Should have resolved on the frame where elapsed crossed 0.5, i.e. tick=5.
    assert tick == 5


def test_await_elapsed_at_uses_absolute_target():
    engine = _make_offscreen_engine()
    engine.set_fixed_timestep(0.1)
    log = []

    @engine.on("start")
    async def start(payload):
        await engine.elapsed_at(0.3)
        log.append("done")

    engine.emit("start")
    engine.elapsed = 0.0
    engine._draw_frame()  # dispatch handler, register waiter
    for tick in range(1, 11):
        engine.elapsed = tick * 0.1
        engine._draw_frame()
        if log:
            break
    assert log == ["done"]
    assert tick == 3


def test_async_handler_exception_propagates():
    engine = _make_offscreen_engine()

    @engine.on("boom")
    async def boom(payload):
        raise RuntimeError("kaboom")

    engine.emit("boom")
    engine._draw_frame()  # dispatches + creates task
    with pytest.raises(RuntimeError, match="kaboom"):
        engine._draw_frame()  # next pump surfaces the failed task


def test_sync_handler_exception_propagates_immediately():
    engine = _make_offscreen_engine()

    @engine.on("boom")
    def boom(payload):
        raise ValueError("sync-boom")

    engine.emit("boom")
    with pytest.raises(ValueError, match="sync-boom"):
        engine._draw_frame()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_async.py -v`
Expected: failures around `await engine.tick()` not advancing because the loop is not pumped, plus the "exception propagates" tests not firing.

- [ ] **Step 3: Wire the asyncio pump and propagate task errors**

In `src/manifoldx/engine.py`, replace the placeholder block in `_draw_frame`:

```python
        # Step 4: pump asyncio loop (async handlers + waiter wakers).
        # Wrapped in a sleep(0) trampoline so the loop processes ready
        # callbacks then returns. (Filled in by Task 6.)
        # PLACEHOLDER UNTIL TASK 6 LANDS:
        # self._aio_loop.run_until_complete(asyncio.sleep(0))
```

with:

```python
        # Step 4: pump asyncio loop (async handlers + waiter wakers).
        self._pump_aio_loop()
```

Add this method on `Engine` (place it near `_draw_frame`):

```python
    def _pump_aio_loop(self) -> None:
        """Drive the engine's asyncio loop until quiescent.

        Runs all currently-runnable callbacks: woken futures from waiter
        resolution, freshly-scheduled handler tasks, and I/O completions.
        If any async handler raised an exception during this pump, the
        first one is re-raised so the frame loop crashes per the v1
        error policy. (Done tasks are no longer in asyncio.all_tasks(),
        so we capture exceptions via add_done_callback in _invoke_async
        and spool them on engine._task_errors.)
        """
        self._aio_loop.run_until_complete(asyncio.sleep(0))
        if self._task_errors:
            exc = self._task_errors.pop(0)
            raise exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_event_async.py -v`
Expected: 5 passed.

- [ ] **Step 5: Verify other tests still pass**

Run: `uv run pytest tests/test_event_bus.py tests/test_events_readonly_view.py tests/test_frame_waiters.py tests/test_engine_event_api.py tests/test_event_frame_integration.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/engine.py tests/test_event_async.py
git commit -m "feat(events): pump asyncio loop in _draw_frame, propagate task errors"
```

---

### Task 7: Built-in `'startup'` immediate dispatch

**Files:**
- Modify: `src/manifoldx/engine.py` (`run`, `render`)

- [ ] **Step 1: Write the failing test (extend `test_event_frame_integration.py`)**

Append to `tests/test_event_frame_integration.py`:

```python
def test_startup_fires_once_before_first_frame():
    engine = _make_offscreen_engine()
    log = []

    @engine.on("startup")
    def on_start(payload):
        log.append("start")

    @engine.on("frame")
    def on_frame(payload):
        log.append(("frame", payload["frame"]))

    # Mimic what run()/render() do: call dispatch_immediate('startup') once
    # before the first _draw_frame.
    engine._event_bus.dispatch_immediate(engine, "startup", {})
    engine._draw_frame()

    assert log == ["start", ("frame", 0)]
```

- [ ] **Step 2: Run test to verify it fails or passes by accident**

Run: `uv run pytest tests/test_event_frame_integration.py::test_startup_fires_once_before_first_frame -v`
Expected: PASS (this test exercises the bus directly; it confirms the contract independently of `run()`/`render()`).

- [ ] **Step 3: Replace `_startup_callbacks` usage in `run()` and `render()`**

In `src/manifoldx/engine.py`, find in `run()`:

```python
        # Run startup callbacks
        for callback in self._startup_callbacks:
            callback()
```

Replace with:

```python
        # Fire built-in 'startup' event before the first frame.
        self._event_bus.dispatch_immediate(self, "startup", {})
```

Find the same pattern in `render()`:

```python
        # Run startup callbacks
        for callback in self._startup_callbacks:
            callback()
```

Replace with:

```python
        self._event_bus.dispatch_immediate(self, "startup", {})
```

- [ ] **Step 4: Run all event tests**

Run: `uv run pytest tests/test_event_bus.py tests/test_event_frame_integration.py tests/test_event_async.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/engine.py tests/test_event_frame_integration.py
git commit -m "feat(events): 'startup' fires via dispatch_immediate in run()/render()"
```

---

### Task 8: Built-in `'shutdown'` + clean asyncio teardown

**Files:**
- Modify: `src/manifoldx/engine.py` (`quit`, `run`, `render`)
- Test: `tests/test_event_shutdown.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_shutdown.py`:

```python
"""Shutdown: 'shutdown' fires, async tasks cancelled, try/finally runs."""
import asyncio
import pytest


def _make_offscreen_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_shutdown_fires_sync_handlers_immediately():
    engine = _make_offscreen_engine()
    log = []

    @engine.on("shutdown")
    def on_stop(payload):
        log.append("bye")

    engine.shutdown_events()
    assert log == ["bye"]


def test_long_running_async_handler_cancelled_with_finally_run():
    engine = _make_offscreen_engine()
    log = []

    @engine.on("go")
    async def loop_forever(payload):
        try:
            while True:
                await engine.tick()
                log.append("tick")
        except asyncio.CancelledError:
            log.append("cancelled")
            raise
        finally:
            log.append("finally")

    engine.emit("go")
    engine._draw_frame()  # dispatch + first await
    engine._draw_frame()  # one tick body iteration
    assert "tick" in log

    engine.shutdown_events()
    # cancellation propagated and finally ran
    assert "cancelled" in log
    assert log[-1] == "finally"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_shutdown.py -v`
Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'shutdown_events'`.

- [ ] **Step 3: Add `shutdown_events()` to Engine**

In `src/manifoldx/engine.py`, add this method near `quit()`:

```python
    def shutdown_events(self) -> None:
        """Fire 'shutdown', cancel async tasks, drain the loop, close it.

        Idempotent: safe to call from quit() and from finalization paths.
        """
        if self._aio_loop.is_closed():
            return

        # 1. Sync 'shutdown' handlers run inline; async ones get scheduled
        #    as tasks on the loop.
        self._event_bus.dispatch_immediate(self, "shutdown", {})

        # 2. Cancel every outstanding task so while-True coroutines get
        #    CancelledError on their next await.
        for task in list(asyncio.all_tasks(loop=self._aio_loop)):
            task.cancel()

        # 3. Cancel any outstanding waiter futures so coroutines blocked
        #    on engine.tick / delay / elapsed_at unblock with CancelledError.
        self._frame_waiters.cancel_all()

        # 4. Pump the loop one final time to let try/finally cleanup run.
        #    Done callbacks fire here, populating _task_errors for any
        #    non-cancel exceptions raised during cleanup.
        try:
            self._aio_loop.run_until_complete(asyncio.sleep(0))
        finally:
            pending = self._task_errors
            self._task_errors = []
            self._aio_loop.close()

        # 5. Surface the first non-cancel exception (if any) per the v1
        #    "errors propagate" policy.
        if pending:
            raise pending[0]
```

Wire it into `quit()`. Find:

```python
    def quit(self):
        self._running = False
        # Stop the event loop
        if hasattr(self, "_event_loop") and self._event_loop is not None:
            self._event_loop.stop()
        # Also close the canvas
        if self._render_canvas is not None:
            try:
                self._render_canvas.close()
            except Exception:
                pass
```

Replace with:

```python
    def quit(self):
        self._running = False
        self.shutdown_events()
        # Stop the rendercanvas event loop
        if hasattr(self, "_event_loop") and self._event_loop is not None:
            self._event_loop.stop()
        # Also close the canvas
        if self._render_canvas is not None:
            try:
                self._render_canvas.close()
            except Exception:
                pass
```

Also rewire the `finally` blocks in `run()` and `render()` to call `shutdown_events()` instead of looping over `_shutdown_callbacks`. Find in `run()`:

```python
        finally:
            self._running = False
            for callback in self._shutdown_callbacks:
                callback()
```

Replace with:

```python
        finally:
            self._running = False
            self.shutdown_events()
```

Find in `render()`:

```python
        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            callback()
```

Replace with:

```python
        self.shutdown_events()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_event_shutdown.py -v`
Expected: 2 passed.

- [ ] **Step 5: Verify all event tests still pass together**

Run: `uv run pytest tests/test_event_bus.py tests/test_events_readonly_view.py tests/test_frame_waiters.py tests/test_engine_event_api.py tests/test_event_frame_integration.py tests/test_event_async.py tests/test_event_shutdown.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/engine.py tests/test_event_shutdown.py
git commit -m "feat(events): 'shutdown' dispatch + asyncio task cancellation in quit/run/render"
```

---

### Task 9: Sweep examples and tests for legacy decorators

**Files:**
- Modify: every file under `examples/` that uses `@engine.startup` (or `@engine.shutdown` / `@engine.update`).
- Modify: every file under `tests/` that references those names.

- [ ] **Step 1: Find every callsite**

Run from `repos/manifoldx`:

```bash
grep -rn "@engine\.startup\|@engine\.shutdown\|@engine\.update\|engine\.startup(\|engine\.shutdown(\|engine\.update(" examples/ tests/ src/
```

Record the list. Expected hits: most files in `examples/` (cube.py, nbody.py, gas.py, boids.py, pbr_demo.py, point_cloud_demo.py, spheres.py, axes_demo.py, gas_compute.py, nbody_compute.py, point_cloud_compute.py, volume_demo.py, scatter_plot.py, smoke_demo.py, hello_world.py, smoke_demo.py); plus any tests.

- [ ] **Step 2: Replace `@engine.startup` with `@engine.on("startup")` in each file**

For every example like `examples/cube.py:17`:

```python
@engine.startup
def create_cubes():
    engine.spawn(...)
```

becomes:

```python
@engine.on("startup")
def create_cubes(_payload):
    engine.spawn(...)
```

Note: the new signature requires accepting the `payload` arg. Use `_payload` to signal "ignored" by convention.

For `@engine.shutdown` and `@engine.update` (rare): same treatment, using `"shutdown"` / `"frame"`.

Edit each file with `Edit` tool, one occurrence at a time. Do not refactor anything else in those files.

- [ ] **Step 3: Run a smoke render on cube.py and nbody.py to confirm the migration**

```bash
uv run python examples/cube.py --render --duration 1 --fps 10 --output /tmp/cube_smoke.mp4
uv run python examples/nbody.py --render --duration 1 --fps 10 --output /tmp/nbody_smoke.mp4
```

Expected: both produce a `.mp4` file without errors.

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/ -x 2>&1 | tail -20
```

Expected: all tests pass. If any test references the old decorators, update it the same way.

- [ ] **Step 5: Commit**

```bash
git add examples/ tests/
git commit -m "refactor(events): migrate startup/shutdown/update decorators to engine.on(...)"
```

---

### Task 10: Add `examples/event_dolly.py` (async camera dolly)

**Files:**
- Create: `examples/event_dolly.py`

- [ ] **Step 1: Write the example**

Create `examples/event_dolly.py`:

```python
"""Event-driven camera dolly: zoom in, hold, zoom out, hold — in a single
async while-True loop using `await engine.delay(...)` between phases."""

import math
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query


engine = mx.Engine("Event Dolly")

cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.RED)


@engine.on("startup")
def setup(_payload):
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(0, 0, 0)),
        n=1,
    )
    engine.emit("dolly")


@engine.on("dolly")
async def dolly(_payload):
    """Loop forever: dolly camera between (0, 1, 4) and (0, 1, 1.5)."""
    far = np.array([0, 1, 4], dtype=np.float32)
    near = np.array([0, 1, 1.5], dtype=np.float32)
    while True:
        # Zoom in over 2 seconds.
        t0 = engine.elapsed
        while engine.elapsed - t0 < 2.0:
            t = (engine.elapsed - t0) / 2.0
            engine.camera.position = (1 - t) * far + t * near
            await engine.tick()

        # Hold for 1 second.
        await engine.delay(1.0)

        # Zoom out over 2 seconds.
        t0 = engine.elapsed
        while engine.elapsed - t0 < 2.0:
            t = (engine.elapsed - t0) / 2.0
            engine.camera.position = (1 - t) * near + t * far
            await engine.tick()

        # Hold for 1 second.
        await engine.delay(1.0)


@engine.system
def rotate(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=0, y=dt * math.pi * 0.5, z=0)


if __name__ == "__main__":
    engine.cli()
```

- [ ] **Step 2: Smoke render**

```bash
uv run python examples/event_dolly.py --render --duration 6 --fps 30 --output /tmp/event_dolly.mp4
```

Expected: produces `/tmp/event_dolly.mp4`, ~6 seconds, with camera dolly visible. No exceptions.

- [ ] **Step 3: Commit**

```bash
git add examples/event_dolly.py
git commit -m "feat(examples): event_dolly.py — async camera dolly via engine.delay/tick"
```

---

### Task 11: Add `examples/event_pulse.py` (sync handler + spawn)

**Files:**
- Create: `examples/event_pulse.py`

- [ ] **Step 1: Write the example**

Create `examples/event_pulse.py`:

```python
"""Sync handler reacting to a periodic event: spawn a new cube every second."""

import math

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query


engine = mx.Engine("Event Pulse")

cube_mesh = mx.geometry.cube(0.4, 0.4, 0.4)
cube_material = mx.material.phong(mx.colors.GREEN)

_state = {"next_pulse_at": 1.0, "count": 0}


@engine.on("frame")
def emit_pulses(payload):
    if payload["elapsed"] >= _state["next_pulse_at"]:
        engine.emit("pulse", {"index": _state["count"]})
        _state["count"] += 1
        _state["next_pulse_at"] += 1.0


@engine.on("pulse")
def on_pulse(payload):
    i = payload["index"]
    angle = i * 0.7
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(math.cos(angle) * 2, 0, math.sin(angle) * 2)),
        n=1,
    )


@engine.system
def rotate(query: Query[Transform], dt: float):
    query[Transform].rot += Transform.rotation(x=0, y=dt * math.pi * 0.5, z=0)


if __name__ == "__main__":
    engine.cli()
```

- [ ] **Step 2: Smoke render**

```bash
uv run python examples/event_pulse.py --render --duration 5 --fps 30 --output /tmp/event_pulse.mp4
```

Expected: produces a video with cubes appearing one per second; no exceptions.

- [ ] **Step 3: Commit**

```bash
git add examples/event_pulse.py
git commit -m "feat(examples): event_pulse.py — sync handler firing engine.spawn from emitted events"
```

---

### Task 12: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the entry**

Open `CHANGELOG.md`. Under the existing `## [Unreleased]` heading, in the `### Features` block, prepend (or insert at the top of the block) the following:

```markdown
- **Event-driven system v1** — new `engine.emit(name, payload)` / `@engine.on(name)` event bus running parallel to the frame loop. Sync handlers run inline at the head of the next frame; async handlers are scheduled on a per-engine asyncio loop pumped synchronously each frame, with `await engine.tick()`, `await engine.delay(seconds)`, and `await engine.elapsed_at(target)` waiter primitives, plus `await engine.run_blocking(fn, ...)` as the escape hatch for genuinely blocking work. Globals (camera, lights) are written immediately; ECS reads come from a `ReadOnlyView` (mutations route through `engine.commands` / `engine.spawn` / `engine.destroy`). The legacy `@engine.startup` / `@engine.shutdown` / `@engine.update` decorators have been replaced by `@engine.on("startup" | "shutdown" | "frame")`; the `'frame'` event payload carries `dt`, `elapsed`, and `frame`. `engine.quit()` cancels all in-flight async tasks, allowing `try/finally` cleanup to run before the loop closes. Errors from handlers propagate and crash the frame loop in v1 by design. New `examples/event_dolly.py` (async camera dolly with `await engine.delay(...)`) and `examples/event_pulse.py` (sync handler emitting `engine.spawn`). Design: `.knowledge/analysis/2026-05-08-event-driven-system-design.md`.
```

- [ ] **Step 2: Verify the file parses**

```bash
head -40 CHANGELOG.md
```

Expected: the new bullet shows under `[Unreleased] / Features`, before earlier entries.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(events): CHANGELOG entry for event-driven system v1"
```

---

## Final verification

- [ ] **Step 1: Full test suite**

Run: `uv run pytest tests/ -v 2>&1 | tail -30`
Expected: all tests pass. No `RuntimeWarning` about the asyncio loop.

- [ ] **Step 2: Lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 3: Smoke-render a sample of existing examples**

Run a quick batch to confirm the legacy-decorator sweep didn't break visual output:

```bash
uv run python examples/cube.py --render --duration 1 --fps 10 --output /tmp/cube_post.mp4
uv run python examples/point_cloud_demo.py --render --duration 1 --fps 10 --output /tmp/pcd_post.mp4
uv run python examples/event_dolly.py --render --duration 3 --fps 30 --output /tmp/dolly_post.mp4
uv run python examples/event_pulse.py --render --duration 3 --fps 30 --output /tmp/pulse_post.mp4
```

Expected: all four MP4s produced cleanly.

- [ ] **Step 4: Confirm no leftover legacy decorators**

```bash
grep -rn "@engine\.startup\|@engine\.shutdown\|@engine\.update\|_startup_callbacks\|_shutdown_callbacks\|_update_callbacks" src/ examples/ tests/
```

Expected: no matches. If any remain, fix them and re-commit.
