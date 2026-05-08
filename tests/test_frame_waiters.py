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


@pytest.fixture
def waiters(loop):
    """FrameWaiters wired to a fixed loop via a constant-returning provider."""
    return FrameWaiters(lambda: loop)


def test_tick_resolves_on_next_resolve_call(waiters):
    fut = waiters.add_tick()
    assert not fut.done()
    waiters.resolve(elapsed=0.0)
    assert fut.done()
    assert fut.result() is None


def test_tick_list_emptied_after_resolve(waiters):
    waiters.add_tick()
    waiters.resolve(elapsed=0.0)
    fut = waiters.add_tick()
    # this future should remain unresolved until the NEXT resolve call
    assert not fut.done()


def test_delay_resolves_only_when_deadline_passed(waiters):
    fut = waiters.add_delay(seconds=0.5, current_elapsed=10.0)
    waiters.resolve(elapsed=10.3)
    assert not fut.done()
    waiters.resolve(elapsed=10.5)
    assert fut.done()


def test_elapsed_at_uses_absolute_target(waiters):
    fut = waiters.add_elapsed_at(target=42.0)
    waiters.resolve(elapsed=41.9)
    assert not fut.done()
    waiters.resolve(elapsed=42.0)
    assert fut.done()


def test_resolve_keeps_unsatisfied_waiters(waiters):
    fut_short = waiters.add_delay(seconds=0.1, current_elapsed=0.0)
    fut_long = waiters.add_delay(seconds=10.0, current_elapsed=0.0)
    waiters.resolve(elapsed=0.2)
    assert fut_short.done()
    assert not fut_long.done()
    waiters.resolve(elapsed=10.1)
    assert fut_long.done()
