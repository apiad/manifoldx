"""Heavy pure-Python work in a child process, its results handed back one at a time.

A thread running pure Python shares the GIL with the frame loop, and C calls that hold it
(building big numpy arrays, say) stall frames for hundreds of milliseconds. A child process
shares nothing, so the loop keeps its frames; the price is pickling the results. Yielding
them in pieces lets the frame loop unpickle one per frame:

    stream = ProcessStream(build_world, seed)   # build_world: a generator, importable by name
    ...
    item = stream.take()                        # each frame: the next piece, or None

The child is a fresh `python -m manifoldx.workers`, not a multiprocessing spawn, which would
run the parent's main script again (an app's window, a test's GPU lock) before the work.
"""

from __future__ import annotations

import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import traceback

_ITEM, _ERROR, _END = "item", "error", "end"
_LEN = struct.Struct("<Q")


def _send(out, message) -> None:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    out.write(_LEN.pack(len(data)))
    out.write(data)
    out.flush()


def _child() -> None:
    """The child's side: read (fn, args) from stdin, send each item fn yields on stdout."""
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # what the work prints goes to stderr, never into the item stream
    try:
        try:
            fn, args = pickle.load(sys.stdin.buffer)
            for item in fn(*args):
                _send(out, (_ITEM, item))
            _send(out, (_END, None))
        except BrokenPipeError:
            raise
        except BaseException:  # noqa: BLE001  the parent raises it, with the child's traceback
            _send(out, (_ERROR, traceback.format_exc()))
        out.close()
    except BrokenPipeError:  # the parent exited or closed the stream: nobody wants the rest
        pass
    # Skip the interpreter's teardown: freeing a big heap object by object takes seconds.
    os._exit(0)


class ProcessStream:
    """Run the generator function `fn(*args)` in a child process and hand back what it yields.

    `fn` and its arguments must be picklable (a module-level function), and the child finds
    them on the parent's sys.path. A reader thread receives each item's bytes (I/O, no GIL
    held); take() unpickles at most one per call, so a caller that takes once per frame
    spends one item's unpickling per frame.
    """

    def __init__(self, fn, *args):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        self._proc = subprocess.Popen([sys.executable, "-m", "manifoldx.workers"],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self._proc.stdin.write(pickle.dumps((fn, args)))
        self._proc.stdin.close()
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.done = False  # True once every item has been taken
        threading.Thread(target=self._read, args=(self._proc.stdout,), daemon=True).start()

    def _read(self, pipe):
        try:
            while head := pipe.read(_LEN.size):
                self._inbox.put(pipe.read(_LEN.unpack(head)[0]))
        except (OSError, ValueError):  # the pipe was closed under the read
            pass
        finally:
            self._inbox.put(None)  # end of the stream; take() tells a clean end from a death
            pipe.close()

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
            raise RuntimeError(f"the worker process exited ({self._proc.poll()}) before finishing")
        kind, payload = pickle.loads(data)
        if kind == _END:
            self.done = True
            self._proc.poll()  # never wait: the child exits on its own right after
            return None
        if kind == _ERROR:
            self.done = True
            raise RuntimeError(f"the worker process failed:\n{payload}")
        return payload

    def close(self):
        """Stop the child if it still runs."""
        if self._proc.poll() is None:
            self._proc.terminate()
        self.done = True


if __name__ == "__main__":
    _child()
