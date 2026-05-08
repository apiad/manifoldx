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
        captured["pos0"] = view[Transform].pos.data[0].copy()

    engine.emit("inspect")
    engine._draw_frame()
    assert isinstance(captured["view"], ReadOnlyView)
    assert tuple(captured["pos0"]) == (1.0, 2.0, 3.0)
