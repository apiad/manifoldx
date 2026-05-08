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
