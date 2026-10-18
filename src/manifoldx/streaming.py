"""Cell streaming: keep meshes built and drawn around a moving focus point.

A Streamer knows nothing of what it streams. It cuts the ground plane into
square cells of side `cell`; cell (i, j) covers x in [i*cell, (i+1)*cell) and
z in [j*cell, (j+1)*cell). `build(i, j)` returns the cell's MeshData (or None for
a cell with nothing to draw) and runs in a worker, never on the frame.

Two rings sit around the focus cell. The cells within `ready_radius` are built
in the background; the cells within `radius` are drawn. Stepping into a new
cell uploads cells that are already built, at most `uploads_per_frame` a frame,
nearest first, so a frame never waits for a builder. A cell that leaves
`radius` stops drawing and frees its GPU buffers; one that leaves
`ready_radius` is dropped and its data forgotten.

    streamer = Streamer(engine, cell=32.0, build=build_cell)

    @engine.system
    def stream(query, dt):
        streamer.focus(player_x, player_z)   # float64 world position, once a frame

`workers="thread"` builds in one background thread; `"process"` builds in one
worker process, which shares no GIL with the frame but needs a picklable,
importable `build` (a top-level function, with the engine set up under
`if __name__ == "__main__":`) and ships each MeshData back over IPC.
"""

from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter
from typing import Callable, NamedTuple

import numpy as np


class MeshData(NamedTuple):
    """A cell's mesh: positions relative to the cell's corner (i*cell, 0, j*cell)."""

    positions: np.ndarray  # f32[N, 3]
    normals: np.ndarray  # f32[N, 3]
    colors: np.ndarray  # f32[N, 3], linear
    indices: np.ndarray  # u32[M]


class Streamer:
    """Builds, uploads and frees cell meshes around a focus point."""

    def __init__(
        self,
        engine,
        cell: float,
        build: Callable[[int, int], MeshData | None],
        radius: int = 1,
        ready_radius: int = 2,
        uploads_per_frame: int = 2,
        workers: str = "thread",
        material=None,
    ):
        if ready_radius < radius:
            raise ValueError(f"ready_radius ({ready_radius}) must be at least radius ({radius})")
        if workers == "thread":
            self._pool = ThreadPoolExecutor(max_workers=1)
        elif workers == "process":
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        else:
            raise ValueError(f'workers must be "thread" or "process", got {workers!r}')
        if material is None:
            from manifoldx.resources import StandardMaterial

            material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True)
        self.engine, self.cell, self.build = engine, float(cell), build
        self.radius, self.ready_radius, self.uploads_per_frame = radius, ready_radius, uploads_per_frame
        self.material = material
        self._pending = {}  # cell -> Future
        self._ready = {}  # cell -> MeshData | None, built and kept
        self._drawn = {}  # cell -> EntityHandle | None (None: an empty cell)
        self._stats = {"built": 0, "uploaded": 0, "dropped": 0, "max_frame_ms": 0.0}
        self._last = None

    def focus(self, x: float, z: float) -> None:
        """Move the focus to world position (x, z). Call it once a frame."""
        now = perf_counter()
        if self._last is not None:
            self._stats["max_frame_ms"] = max(self._stats["max_frame_ms"], (now - self._last) * 1000.0)
        self._last = now
        fi, fj = math.floor(x / self.cell), math.floor(z / self.cell)

        def ring(cell):
            return max(abs(cell[0] - fi), abs(cell[1] - fj))

        for c, fut in list(self._pending.items()):
            if fut.done():
                del self._pending[c]
                self._ready[c] = fut.result()
                self._stats["built"] += 1
        for c in [c for c in self._drawn if ring(c) > self.radius]:
            self._undraw(c)
        for c in [c for c in self._ready if ring(c) > self.ready_radius]:
            del self._ready[c]
            self._stats["dropped"] += 1
        for c in [c for c in self._pending if ring(c) > self.ready_radius]:
            self._pending.pop(c).cancel()

        r = self.ready_radius
        wanted = sorted(
            ((fi + di, fj + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)),
            key=lambda c: (c[0] - fi) ** 2 + (c[1] - fj) ** 2,
        )
        for c in wanted:
            if c not in self._ready and c not in self._pending:
                self._pending[c] = self._pool.submit(self.build, *c)
        budget = self.uploads_per_frame
        for c in wanted:
            if budget == 0:
                break
            if ring(c) <= self.radius and c not in self._drawn and c in self._ready:
                self._draw(c)
                budget -= 1

    def _draw(self, c):
        data = self._ready[c]
        if data is None:
            self._drawn[c] = None
            return
        from manifoldx.components import Material, Mesh, Transform

        geo = {
            "positions": np.asarray(data.positions, np.float32),
            "normals": np.asarray(data.normals, np.float32),
            "colors": np.asarray(data.colors, np.float32),
            "indices": np.asarray(data.indices, np.uint32),
        }
        self._drawn[c] = self.engine.spawn(
            Mesh(geo), Material(self.material), Transform(pos=(c[0] * self.cell, 0.0, c[1] * self.cell))
        )
        self._stats["uploaded"] += 1

    def _undraw(self, c):
        handle = self._drawn.pop(c)
        if handle is not None:
            handle.destroy()

    def drawn(self) -> set[tuple[int, int]]:
        """The cells drawn now, empty cells (build returned None) included."""
        return set(self._drawn)

    def stats(self) -> dict:
        """built, uploaded and dropped cell counts, and the longest gap between
        two focus() calls in ms (one frame, when focus runs once a frame)."""
        return dict(self._stats)

    def close(self) -> None:
        """Stop the worker and destroy every drawn cell."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        for c in list(self._drawn):
            self._undraw(c)
        self._pending.clear()
        self._ready.clear()


__all__ = ["MeshData", "Streamer"]
