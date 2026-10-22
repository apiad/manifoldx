"""Cell streaming: keep meshes built and drawn around a moving focus point.

A Streamer knows nothing of what it streams. It cuts the ground plane into
square cells of side `cell`; cell (i, j) covers x in [i*cell, (i+1)*cell) and
z in [j*cell, (j+1)*cell). `build(i, j)` returns the cell's MeshData (or None for
a cell with nothing to draw) and runs in a worker, never on the frame.

Two rings sit around the focus cell. The cells within `ready_radius` are built
in the background and uploaded to the GPU hidden; the cells within `radius` are
drawn. The worker also packs each cell's vertices the way its vertex buffer
holds them, and the frame copies at most `upload_bytes_per_frame` of them to
the GPU, nearest cell first, so a big cell takes several frames and none of
them stalls. Stepping into a new cell then draws cells that are already on the
GPU, at most `uploads_per_frame` a frame, and costs the frame next to nothing.
A cell that leaves `radius` stops drawing and keeps its buffers; one that
leaves `ready_radius` is dropped, its buffers freed and its data forgotten.

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

from manifoldx.resources import pack_vertices


class MeshData(NamedTuple):
    """A cell's mesh: positions relative to the cell's corner (i*cell, 0, j*cell)."""

    positions: np.ndarray  # f32[N, 3]
    normals: np.ndarray  # f32[N, 3]
    colors: np.ndarray  # f32[N, 3], linear
    indices: np.ndarray  # u32[M]
    surfaces: np.ndarray | None = None  # i[N]: surface pattern ids (resources.SURFACE_PATTERNS)


class _Cell:
    """A built cell: its geometry dict, and its packed bytes until they are on the GPU."""

    __slots__ = ("geo", "info", "vbytes", "ibytes")

    def __init__(self, geo, info, vbytes, ibytes):
        self.geo, self.info, self.vbytes, self.ibytes = geo, info, vbytes, ibytes


def _pack(build, i, j):
    """The worker's job: build cell (i, j) and pack it for its GPU buffers."""
    data = build(i, j)
    if data is None:
        return None
    geo = {
        "positions": np.asarray(data.positions, np.float32),
        "normals": np.asarray(data.normals, np.float32),
        "colors": np.asarray(data.colors, np.float32),
        "indices": np.asarray(data.indices, np.uint32),
    }
    if data.surfaces is not None:
        geo["surfaces"] = np.asarray(data.surfaces, np.int32)
    vbytes, info = pack_vertices(geo)
    return _Cell(geo, info, vbytes, geo["indices"].tobytes())


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
        upload_bytes_per_frame: int = 2 << 20,
        workers: str = "thread",
        material=None,
    ):
        if ready_radius < radius:
            raise ValueError(f"ready_radius ({ready_radius}) must be at least radius ({radius})")
        if workers == "thread":
            self._pool = ThreadPoolExecutor(max_workers=1)
        elif workers == "process":
            self._pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        else:
            raise ValueError(f'workers must be "thread" or "process", got {workers!r}')
        if material is None:
            from manifoldx.resources import StandardMaterial

            material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True)
        self.engine, self.cell, self.build = engine, float(cell), build
        self.radius, self.ready_radius, self.uploads_per_frame = (
            radius,
            ready_radius,
            uploads_per_frame,
        )
        self.material = material
        self.upload_bytes_per_frame = max(4, int(upload_bytes_per_frame) // 4 * 4)
        self._pending = {}  # cell -> Future
        self._ready = {}  # cell -> _Cell | None, built and kept
        self._gpu = {}  # cell -> its upload: geometry id, buffers, bytes written, done
        self._drawn = {}  # cell -> EntityHandle | None (None: an empty cell)
        self._stats = {"built": 0, "uploaded": 0, "dropped": 0, "max_frame_ms": 0.0}
        self._last = None

    def focus(self, x: float, z: float) -> None:
        """Move the focus to world position (x, z). Call it once a frame."""
        now = perf_counter()
        if self._last is not None:
            self._stats["max_frame_ms"] = max(
                self._stats["max_frame_ms"], (now - self._last) * 1000.0
            )
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
            self._unload(c)
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
                self._pending[c] = self._pool.submit(_pack, self.build, *c)
        room = self.upload_bytes_per_frame
        for c in wanted:
            if room <= 0:
                break
            if self._ready.get(c) is not None and not self._on_gpu(c):
                room -= self._upload(c, room)
        budget = self.uploads_per_frame
        for c in wanted:
            if budget == 0:
                break
            if (
                ring(c) <= self.radius
                and c not in self._drawn
                and c in self._ready
                and self._on_gpu(c)
            ):
                self._draw(c)
                budget -= 1

    def _on_gpu(self, c) -> bool:
        """Whether cell c can be drawn without building a buffer on the frame."""
        if self._ready.get(c) is None or self.engine._device is None:
            return True  # nothing to upload, or no device yet: the render pass builds the buffers
        return self._gpu.get(c, {}).get("done", False)

    def _upload(self, c, room: int) -> int:
        """Copy up to `room` bytes of cell c to its GPU buffers; returns the bytes copied."""
        import wgpu

        cell, reg, dev = self._ready[c], self.engine._geometry_registry, self.engine._device
        if dev is None:
            return 0
        up = self._gpu.get(c)
        if up is None:
            gid = reg.register(cell.geo)
            up = self._gpu[c] = {
                "id": gid,
                "done": False,
                "at": [0, 0],
                "bufs": [
                    dev.create_buffer(
                        size=len(cell.vbytes),
                        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
                    ),
                    dev.create_buffer(
                        size=len(cell.ibytes),
                        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
                    ),
                ],
            }
        spent = 0
        for k, data in enumerate((cell.vbytes, cell.ibytes)):
            at = up["at"][k]
            n = min(len(data) - at, room - spent)
            if n > 0:
                dev.queue.write_buffer(up["bufs"][k], at, memoryview(data)[at : at + n])
                up["at"][k] += n
                spent += n
        if up["at"] == [len(cell.vbytes), len(cell.ibytes)]:
            vb, ib = up.pop("bufs")
            reg.adopt_buffers(
                up["id"],
                {
                    **cell.info,
                    "vertex_buffer": vb,
                    "index_buffer": ib,
                    "index_count": len(cell.geo["indices"]),
                },
            )
            up["done"] = True
            cell.vbytes = cell.ibytes = None
        return spent

    def _unload(self, c):
        """Free cell c's GPU buffers, whole or half written."""
        up = self._gpu.pop(c, None)
        if up is None:
            return
        for buf in up.get("bufs", ()):
            buf.destroy()
        self.engine._geometry_registry.release(up["id"])

    def _draw(self, c):
        cell = self._ready[c]
        if cell is None:
            self._drawn[c] = None
            return
        from manifoldx.components import Material, Mesh, Transform

        self._drawn[c] = self.engine.spawn(
            Mesh(cell.geo),
            Material(self.material),
            Transform(pos=(c[0] * self.cell, 0.0, c[1] * self.cell)),
        )
        # Drawn before any device: its buffers come from the render pass, and are still ours to free.
        self._gpu.setdefault(c, {"id": self._drawn[c]._geometry_id, "done": True})
        self._stats["uploaded"] += 1

    def _undraw(self, c):
        handle = self._drawn.pop(c)
        if handle is not None:
            handle.destroy(free_geometry=False)

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
        for c in list(self._gpu):
            self._unload(c)
        self._pending.clear()
        self._ready.clear()


__all__ = ["MeshData", "Streamer"]
