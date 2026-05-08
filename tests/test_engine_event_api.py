"""Engine-level event API: on, emit, tick, delay, elapsed_at, run_blocking."""
import asyncio
import inspect

import manifoldx as mx


def _engine():
    return mx.Engine("test", width=64, height=64)


def test_engine_has_event_bus_and_loop():
    e = _engine()
    assert e._event_bus is not None
    # The loop is lazy: None until something async happens or _get_active_loop
    # is called. _get_active_loop returns either a running loop or a private
    # fallback.
    loop = e._get_active_loop()
    assert isinstance(loop, asyncio.AbstractEventLoop)


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


def test_engine_task_errors_spool_exists_and_empty():
    e = _engine()
    assert e._task_errors == []
