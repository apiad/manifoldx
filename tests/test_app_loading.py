"""An app loads without freezing the window: startup fires inside the running loop, and
engine.loading(label) marks work the scene still waits for (manifoldx#28)."""

import asyncio

import pytest


def _engine(width=64, height=64):
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


def test_run_fires_startup_on_the_first_frame_not_before(monkeypatch):
    import manifoldx as mx
    import manifoldx.backends as backends
    from manifoldx.backends import get_offscreen_canvas

    try:
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    monkeypatch.setattr(backends, "get_desktop_canvas", lambda **kw: get_offscreen_canvas(width=64, height=64))
    engine = mx.Engine("test", width=64, height=64)
    fired = []

    @engine.on("startup")
    def init(payload):
        fired.append(engine._frame_index)

    engine.run()  # the glfw loop is a mock in tests (conftest): it returns without a frame
    assert fired == []  # not before the loop runs
    engine._running = True
    engine._run_loop()
    engine._run_loop()
    assert fired == [0]  # on the first frame, once


def test_async_startup_runs_between_frames_on_the_running_loop():
    engine = _engine()
    done, seen_loading = [], []

    @engine.on("startup")
    async def load(payload):
        with engine.loading("model"):
            for i in range(3):
                await asyncio.sleep(0)
                done.append(i)
                seen_loading.append("model" in engine._loading())

    async def frames():
        engine._fire_startup()  # what the first frame of run() does, with the loop running
        for _ in range(10):
            engine._draw_frame()
            await asyncio.sleep(0)

    asyncio.run(frames())
    assert done == [0, 1, 2] and all(seen_loading)
    assert "model" not in engine._loading()


def test_loading_raises_the_spinner_with_its_label():
    engine = _engine()
    with engine.loading("model"):
        engine._draw_frame()
        assert engine._loading_panel in list(engine.gui)
        assert "model" in engine._loading_line()
    engine._draw_frame()
    assert engine._loading_panel is None


def test_loading_works_as_an_async_context_manager():
    engine = _engine()

    async def go():
        async with engine.loading("textures"):
            assert "textures" in engine._loading()
        assert "textures" not in engine._loading()

    asyncio.run(go())


def test_render_frame_waits_for_what_startup_is_loading():
    import manifoldx as mx
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube

    engine = mx.Engine("test", width=64, height=64)
    done = []

    @engine.on("startup")
    async def load(payload):
        async with engine.loading("model"):
            for i in range(3):
                await engine.tick()  # frames nobody watches still advance
                done.append(i)
            engine.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial(color="#ffffff")), Transform(pos=(0, 0, 0)))

    try:
        engine.render_frame(supersample=1)
    except Exception as e:  # noqa: BLE001  no offscreen canvas on this machine
        if "canvas" in str(e).lower() or "adapter" in str(e).lower():
            pytest.skip(str(e))
        raise
    assert done == [0, 1, 2]
    assert engine.store._alive.sum() >= 1


def test_run_blocking_works_under_a_running_loop():
    import manifoldx as mx

    engine = mx.Engine("test", width=64, height=64)

    async def go():
        return await engine.run_blocking(lambda a, b: a + b, 40, 2)

    assert asyncio.run(go()) == 42
