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
    return MeshData(
        pos,
        np.tile([0, 1, 0], (4, 1)).astype(np.float32),
        np.full((4, 3), 0.5, np.float32),
        np.array([0, 2, 1, 1, 2, 3], np.uint32),
    )


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
    _settle(engine, s, 4.0 + 8 * 40, 4.0)  # 40 cells east
    engine._draw_frame()
    assert s.drawn() == {(i, j) for i in (39, 40, 41) for j in (-1, 0, 1)}
    assert s.stats()["dropped"] >= 25
    assert len(reg._gpu_buffers) == start  # the 5x5 ready ring before, the 5x5 ready ring now
    s.close()
    engine._draw_frame()
    assert len(reg._gpu_buffers) == start - 25


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


def build_big(i, j):
    """About 1.2 MB of vertices: a strip of 30 000 triangles, each with its own corners."""
    n = 30_000
    pos = np.zeros((3 * n, 3), np.float32)
    pos[:, 0] = np.repeat(np.arange(n, dtype=np.float32) % 8, 3)
    pos[1::3, 2] = 1.0
    pos[2::3, 0] += 1.0
    nrm = np.tile(np.array([0, 1, 0], np.float32), (3 * n, 1))
    return MeshData(
        pos, nrm, np.full((3 * n, 3), 0.5, np.float32), np.arange(3 * n, dtype=np.uint32)
    )


def test_the_ready_ring_is_on_the_gpu_before_it_is_drawn():
    engine = _offscreen_engine()
    reg = engine._geometry_registry
    s = Streamer(engine, cell=8.0, build=build_cell)
    try:
        _settle(engine, s, 4.0, 4.0)
        outer = (2, 0)
        assert outer not in s.drawn()
        assert reg.get_gpu_buffers(s._gpu[outer]["id"]) is not None
        # Stepping east draws column 2 without building a buffer on the frame.
        calls = []
        orig = reg.create_buffers
        reg.create_buffers = lambda *a, **k: calls.append(a[0]) or orig(*a, **k)
        _settle(engine, s, 12.0, 4.0)
        assert outer in s.drawn() and calls == []
    finally:
        s.close()


def test_gpu_uploads_stay_within_the_byte_budget():
    engine = _offscreen_engine()
    budget = 256 * 1024
    s = Streamer(
        engine, cell=8.0, build=build_big, radius=0, ready_radius=0, upload_bytes_per_frame=budget
    )
    q = engine._device.queue
    orig, written = q.write_buffer, []
    try:
        s.focus(4.0, 4.0)
        while s._pending:
            time.sleep(0.005)
            s.focus(4.0, 4.0)
        q.write_buffer = lambda buf, off, data, *a: (
            written.append(len(data)) or orig(buf, off, data, *a)
        )
        frames = 0
        while (0, 0) not in s.drawn():
            written.clear()
            s.focus(4.0, 4.0)
            assert sum(written) <= budget
            frames += 1
            assert frames < 100
        assert frames > 4  # 1.4 MB at 256 KB a frame
        engine._draw_frame()
    finally:
        q.write_buffer = orig
        s.close()


def test_undrawn_cells_keep_their_buffers_until_dropped():
    engine = _offscreen_engine()
    reg = engine._geometry_registry
    s = Streamer(engine, cell=8.0, build=build_cell)
    try:
        _settle(engine, s, 4.0, 4.0)
        gid = s._gpu[(-1, 0)]["id"]
        _settle(engine, s, 12.0, 4.0)  # one cell east: column -1 stops drawing
        engine._draw_frame()
        assert (-1, 0) not in s.drawn() and reg.get_gpu_buffers(gid) is not None
        _settle(engine, s, 28.0, 4.0)  # two more east: column -1 leaves the ready ring
        engine._draw_frame()
        assert reg.get_gpu_buffers(gid) is None
    finally:
        s.close()


def test_a_streamer_warmed_before_the_device_uploads_once_it_has_one():
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=32, height=32)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("streaming", width=32, height=32)
    s = Streamer(engine, cell=8.0, build=build_cell)
    try:
        for _ in range(400):  # warm: the device does not exist yet
            s.focus(4.0, 4.0)
            if len(s.drawn()) == 9 and not s._pending:
                break
            time.sleep(0.005)
        assert len(s.drawn()) == 9
        engine._init_canvas(canvas)
        engine._running = True
        engine._draw_frame()  # the nine drawn so far get their buffers here
        reg = engine._geometry_registry
        calls = []
        orig = reg.create_buffers
        reg.create_buffers = lambda *a, **k: calls.append(a[0]) or orig(*a, **k)
        _settle(engine, s, 4.0, 4.0)  # the rest of the ready ring goes up
        _settle(engine, s, 12.0, 4.0)
        assert (2, 0) in s.drawn() and calls == []
    finally:
        s.close()


def test_a_streamed_cell_carries_its_surface_pattern_ids():
    from manifoldx.streaming import MeshData, _pack

    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float32)
    data = MeshData(pos, np.tile([0, 1, 0], (4, 1)).astype(np.float32),
                    np.full((4, 3), 0.5, np.float32), np.array([0, 1, 2, 0, 2, 3], np.uint32),
                    surfaces=np.array([1, 1, 2, 2]))
    cell = _pack(lambda i, j: data, 0, 0)
    assert cell.geo["surfaces"].tolist() == [1, 1, 2, 2]
    packed = np.frombuffer(cell.vbytes, np.float32).reshape(4, cell.info["stride"] // 4)
    assert packed[:, 9].tolist() == [1, 1, 2, 2]  # after position, normal and colour
    plain = _pack(lambda i, j: data._replace(surfaces=None), 0, 0)
    assert "surfaces" not in plain.geo
    assert np.frombuffer(plain.vbytes, np.float32).reshape(4, -1)[:, 9].tolist() == [0, 0, 0, 0]


def test_streamers_share_the_engines_cell_draw_cap():
    engine = _offscreen_engine()
    engine.cell_draws_per_frame = 1
    pair = [Streamer(engine, cell=8.0, build=build_cell, radius=1, ready_radius=1, uploads_per_frame=4)
            for _ in range(2)]
    try:
        most = 0
        for _ in range(300):
            before = sum(len(s.drawn()) for s in pair)
            for s in pair:
                s.focus(4, 4)
            most = max(most, sum(len(s.drawn()) for s in pair) - before)
            engine._draw_frame()
            if sum(len(s.drawn()) for s in pair) == 18:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("the two rings never filled")
        assert most == 1  # each would draw up to 4 a frame on its own
    finally:
        for s in pair:
            s.close()


def test_a_streamed_cell_brings_its_culling_sphere():
    from manifoldx.streaming import _pack

    cell = _pack(build_cell, 0, 0)
    centre, radius = cell.geo["bounds"]
    assert np.allclose(centre, [4, 0, 4]) and radius == pytest.approx(np.sqrt(32))
