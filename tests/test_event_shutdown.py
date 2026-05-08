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
