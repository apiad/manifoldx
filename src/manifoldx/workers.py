"""Heavy pure-Python work in a child process, its results handed back one at a time.

A thread running pure Python shares the GIL with the frame loop, and C calls that hold it
(building big numpy arrays, say) stall frames for hundreds of milliseconds. A child process
shares nothing, so the loop keeps its frames; the price is pickling the results. Yielding
them in pieces lets the frame loop unpickle one per frame:

    stream = ProcessStream(build_world, seed)   # build_world: a generator, importable by name
    ...
    item = stream.take()                        # each frame: the next piece, or None
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import queue
import threading
import traceback

_ITEM, _ERROR, _END = "item", "error", "end"


def _serve(send, fn, args):
    try:
        for item in fn(*args):
            send.send_bytes(pickle.dumps((_ITEM, item), protocol=pickle.HIGHEST_PROTOCOL))
        send.send_bytes(pickle.dumps((_END, None)))
    except BaseException:  # noqa: BLE001  the parent raises it, with the child's traceback
        send.send_bytes(pickle.dumps((_ERROR, traceback.format_exc())))
    finally:
        send.close()


class ProcessStream:
    """Run the generator function `fn(*args)` in a spawned child process and hand back what it yields.

    `fn` and its arguments must be picklable (a module-level function). A reader thread
    receives each item's bytes (I/O, no GIL held); take() unpickles at most one per call, so
    a caller that takes once per frame spends one item's unpickling per frame.
    """

    def __init__(self, fn, *args):
        ctx = mp.get_context("spawn")  # never fork a process holding a GPU device and threads
        recv, send = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(target=_serve, args=(send, fn, args), daemon=True)
        self._proc.start()
        send.close()
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.done = False  # True once every item has been taken
        threading.Thread(target=self._read, args=(recv,), daemon=True).start()

    def _read(self, recv):
        try:
            while True:
                self._inbox.put(recv.recv_bytes())
        except (EOFError, OSError):
            self._inbox.put(None)  # the child died without saying why
        finally:
            recv.close()

    def take(self, timeout: float | None = 0.0):
        """The next item, or None when none has arrived (within `timeout` seconds; None waits).

        Raises RuntimeError with the child's traceback if `fn` raised, or if the child died.
        After the last item, take() returns None and `done` is True.
        """
        if self.done:
            return None
        try:
            data = self._inbox.get(timeout=timeout) if timeout != 0.0 else self._inbox.get_nowait()
        except queue.Empty:
            return None
        if data is None:
            self.done = True
            raise RuntimeError(f"the worker process exited ({self._proc.exitcode}) before finishing")
        kind, payload = pickle.loads(data)
        if kind == _END:
            self.done = True
            self._proc.join(timeout=5)
            return None
        if kind == _ERROR:
            self.done = True
            raise RuntimeError(f"the worker process failed:\n{payload}")
        return payload

    def close(self):
        """Stop the child if it still runs."""
        if self._proc.is_alive():
            self._proc.terminate()
        self.done = True
