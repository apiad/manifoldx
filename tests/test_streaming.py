"""manifoldx.streaming: the Streamer builds, draws and frees cells around a focus."""

import time

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.streaming import MeshData, Streamer


def _offscreen_engine():
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=32, height=32)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("streaming", width=32, height=32)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def build_cell(i, j):
    """Top-level, so a worker process can import it."""
    pos = np.array([[0, 0, 0], [8, 0, 0], [0, 0, 8], [8, 0, 8]], np.float32)
    return MeshData(pos, np.tile([0, 1, 0], (4, 1)).astype(np.float32),
                    np.full((4, 3), 0.5, np.float32), np.array([0, 2, 1, 1, 2, 3], np.uint32))


def build_sparse(i, j):
    return None if (i + j) % 2 else build_cell(i, j)


def _settle(engine, streamer, x, z, frames=400):
    """Focus at (x, z) frame after frame until the drawn ring is full."""
    want = 9 if streamer.radius == 1 else (2 * streamer.radius + 1) ** 2
    for _ in range(frames):
        streamer.focus(x, z)
        engine._draw_frame()
        if len(streamer.drawn()) == want and not streamer._pending:
            return
        time.sleep(0.005)
    raise AssertionError(f"ring never filled: drawn {sorted(streamer.drawn())}")


@pytest.mark.parametrize("workers", ["thread", "process"])
def test_streamer_draws_the_ring_and_builds_the_ready_ring(workers):
    engine = _offscreen_engine()
    s = Streamer(engine, cell=8.0, build=build_cell, workers=workers)
    try:
        _settle(engine, s, 4.0, 4.0)
        assert s.drawn() == {(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
        st = s.stats()
        assert st["built"] == 25 and st["uploaded"] == 9 and st["dropped"] == 0
        h = s._drawn[(1, -1)]
        assert np.allclose(h.transform.pos, [8.0, 0.0, -8.0])  # placed at its corner
    finally:
        s.close()


def test_moving_away_drops_cells_and_frees_their_buffers():
    engine = _offscreen_engine()
    reg = engine._geometry_registry
    s = Streamer(engine, cell=8.0, build=build_cell)
    _settle(engine, s, 4.0, 4.0)
    start = len(reg._gpu_buffers)
    _settle(engine, s, 4.0 + 8 * 40, 4.0)   # 40 cells east
    engine._draw_frame()
    assert s.drawn() == {(i, j) for i in (39, 40, 41) for j in (-1, 0, 1)}
    assert s.stats()["dropped"] >= 25
    assert len(reg._gpu_buffers) == start  # nine drawn before, nine now
    s.close()
    engine._draw_frame()
    assert len(reg._gpu_buffers) == start - 9


def test_uploads_are_bounded_per_frame():
    engine = _offscreen_engine()
    s = Streamer(engine, cell=8.0, build=build_cell, uploads_per_frame=2)
    s.focus(4.0, 4.0)  # submits all 25 builds
    while not all(f.done() for f in s._pending.values()):
        time.sleep(0.005)
    seen = []
    for _ in range(6):
        s.focus(4.0, 4.0)
        seen.append(len(s.drawn()))
    assert seen[:5] == [2, 4, 6, 8, 9]
    s.close()


def test_empty_cells_count_as_drawn():
    engine = _offscreen_engine()
    s = Streamer(engine, cell=8.0, build=build_sparse)
    _settle(engine, s, 4.0, 4.0)
    assert sum(h is None for h in s._drawn.values()) == 4
    s.close()


def test_ready_radius_below_radius_is_refused():
    with pytest.raises(ValueError):
        Streamer(mx.Engine("streaming"), cell=8.0, build=build_cell, radius=2, ready_radius=1)
