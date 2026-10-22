"""The dev channel of `manifoldx run`: what an agent working beside a person can read and ask for.

Its files live in `.manifoldx/` under the folder `manifoldx run` was started from:

- `log.txt`, rotated at 256 KB with one old file kept as `log.txt.1`: reloads, pauses,
  failed loads, keys pressed, and the app's own lines from `engine.log()`.
- `requests/look` asks for `look.png`: the next frame presented, GUI included.
- `requests/state` asks for `state.json`: `engine.state` as JSON, with `repr` for what JSON cannot hold.

`manifoldx look`, `manifoldx state` and `manifoldx log` write the request and wait for the answer.
"""

import json
import logging
import os
import struct
import time
import zlib
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

FOLDER = ".manifoldx"
MAX_BYTES = 256_000

# engine.log writes here. Only manifoldx run gives it a file; otherwise its lines go nowhere.
LOG = logging.getLogger("manifoldx.app")
LOG.addHandler(logging.NullHandler())
LOG.setLevel(logging.INFO)
LOG.propagate = False


class Channel:
    """The open `.manifoldx/` folder of one `manifoldx run`."""

    def __init__(self, root):
        self.dir = Path(root) / FOLDER
        self.requests = self.dir / "requests"
        self.requests.mkdir(parents=True, exist_ok=True)
        for stale in self.requests.iterdir():  # asked of a run that ended
            stale.unlink(missing_ok=True)
        self.handler = RotatingFileHandler(self.dir / "log.txt", maxBytes=MAX_BYTES, backupCount=1, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        LOG.addHandler(self.handler)
        self.keys = frozenset()

    def close(self):
        LOG.removeHandler(self.handler)
        self.handler.close()

    def frame(self, engine):
        """After a frame: log the keys pressed since the last one."""
        keys = engine.input.pressed_keys
        if down := sorted(keys - self.keys):
            LOG.info("keys %s", " ".join(down))
        self.keys = keys

    def poll(self, engine):
        """Answer the requests waiting: a look is taken by the next frame, the state at once."""
        look, state = self.requests / "look", self.requests / "state"
        if look.exists():
            look.unlink(missing_ok=True)
            engine._look_to = str(self.dir / "look.png")
        if state.exists():
            state.unlink(missing_ok=True)
            _replace(self.dir / "state.json", json.dumps(engine.state, default=repr, indent=1).encode())


def ask(what, timeout=5.0, root="."):
    """Ask the app running under manifoldx run in `root` for its look or its state; the answer's path.

    Raises TimeoutError when no answer comes, as when no app runs there."""
    folder = Path(root) / FOLDER
    out = folder / {"look": "look.png", "state": "state.json"}[what]
    before = _mtime(out)
    (folder / "requests").mkdir(parents=True, exist_ok=True)
    request = folder / "requests" / what
    request.touch()
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if _mtime(out) != before:
            return out
        time.sleep(0.05)
    request.unlink(missing_ok=True)
    raise TimeoutError(f"no {what} from an app under manifoldx run in {Path(root).resolve()}")


def save_texture(device, texture, texture_format, path):
    """Write a presented frame's texture to `path` as a PNG."""
    w, h = texture.size[0], texture.size[1]
    data = device.queue.read_texture({"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
                                     {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    pixels = np.frombuffer(data, np.uint8).reshape(h, w, 4)
    rgb = pixels[..., [2, 1, 0]] if str(texture_format).startswith("bgra") else pixels[..., :3]
    _replace(Path(path), png(rgb))


def png(rgb):
    """PNG bytes of an (h, w, 3) uint8 image."""
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), np.ascontiguousarray(rgb).reshape(h, w * 3)], axis=1)

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def _replace(path, data):
    """Write whole, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None
