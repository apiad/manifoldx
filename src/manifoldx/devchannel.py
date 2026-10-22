"""The dev channel of `manifoldx run`: a page and an HTTP API beside the running app, for a person and an agent.

`manifoldx run` serves, on 127.0.0.1 only (default port 8770, the next free one if taken):

- `GET /`: a page with the frame on screen, the log as it grows, engine.state, and the app's events.
- `GET /api/state`: engine.state as JSON (`repr` for what JSON cannot hold).
- `PATCH /api/state`: a JSON merge patch (RFC 7386) applied to engine.state: objects merge, `null` deletes.
- `GET /api/events`: the events the app handles, each with its handlers' docstrings.
- `POST /api/events/<name>`: emit the event with the JSON body as its payload; it runs on the next frame.
- `GET /api/log?after=N&match=RE&wait=S&last=K`: log lines after sequence number N (or the last K), those
  matching the regex RE; with `wait`, block up to S seconds until one arrives.
- `GET /api/look`: the next frame presented, GUI included, as a PNG.

Everything that reads or changes the app runs on the engine's thread between frames, so a request
never races the app. The log is also written to `.manifoldx/log.txt` (rotated at 256 KB, one old file
kept), and `.manifoldx/server.json` holds the URL the `manifoldx` commands use.
"""

import collections
import json
import logging
import os
import queue
import re
import struct
import threading
import time
import zlib
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

FOLDER = ".manifoldx"
PORT = 8770
MAX_BYTES = 256_000
KEEP_LINES = 5000   # log lines the API keeps in memory
WAIT_MAX = 300.0    # seconds a log request may wait
LOOK_WAIT = 5.0     # seconds a look waits for a frame (none come while the app is paused)

# engine.log writes here. Only manifoldx run gives it somewhere to go; otherwise its lines are dropped.
LOG = logging.getLogger("manifoldx.app")
LOG.addHandler(logging.NullHandler())
LOG.setLevel(logging.INFO)
LOG.propagate = False


class Lines(logging.Handler):
    """The last KEEP_LINES log lines, numbered from 1, and a way to wait for the next."""

    def __init__(self):
        super().__init__()
        self.lines = collections.deque(maxlen=KEEP_LINES)
        self.seq = 0
        self.cv = threading.Condition()

    def emit(self, record):
        with self.cv:
            self.seq += 1
            self.lines.append({"seq": self.seq, "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                               "text": record.getMessage()})
            self.cv.notify_all()

    def read(self, after=None, match=None, wait=0.0, last=None):
        """Lines after `after` (or the last `last`), only those matching `match`, waiting up to `wait` for one."""
        pattern = re.compile(match) if match else None
        end = time.monotonic() + min(wait, WAIT_MAX)
        with self.cv:
            while True:
                if after is None:
                    found = list(self.lines)[-last:] if last else []
                    after = self.seq
                else:
                    found = [line for line in self.lines if line["seq"] > after]
                found = [line for line in found if pattern is None or pattern.search(line["text"])]
                left = end - time.monotonic()
                if found or left <= 0:
                    return {"lines": found, "next": self.seq}
                self.cv.wait(left)


def merge_patch(target, patch):
    """Apply a JSON merge patch (RFC 7386) to the dict `target` in place."""
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_patch(target[key], value)
        else:
            target[key] = value


class Channel:
    """The dev channel of one `manifoldx run`: the log, the server, and the jobs the engine's thread runs."""

    def __init__(self, root, hot, port=PORT):
        self.dir = Path(root) / FOLDER
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hot = hot  # the HotReloader: its engine is the one running now, across reloads
        self.file = RotatingFileHandler(self.dir / "log.txt", maxBytes=MAX_BYTES, backupCount=1, encoding="utf-8")
        self.file.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        self.lines = Lines()
        LOG.addHandler(self.file)
        LOG.addHandler(self.lines)
        self.jobs = queue.SimpleQueue()
        self.keys = frozenset()
        self.server = _bind(port, self)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        (self.dir / "server.json").write_text(json.dumps({"url": self.url, "pid": os.getpid()}))
        threading.Thread(target=self.server.serve_forever, name="manifoldx-dev", daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        (self.dir / "server.json").unlink(missing_ok=True)
        LOG.removeHandler(self.file)
        LOG.removeHandler(self.lines)
        self.file.close()

    # -- the engine's thread --

    def run_jobs(self):
        """Run what requests asked of the app; the reloader calls it every frame, paused or not."""
        while True:
            try:
                fn, future = self.jobs.get_nowait()
            except queue.Empty:
                return
            try:
                future.set_result(fn(self.hot.engine))
            except Exception as e:  # noqa: BLE001  the request gets the error, the app goes on
                future.set_exception(e)

    def frame(self, engine):
        """After a frame: log the keys pressed since the last one."""
        keys = engine.input.pressed_keys
        if down := sorted(keys - self.keys):
            LOG.info("keys %s", " ".join(down))
        self.keys = keys

    # -- the server's threads --

    def on_engine(self, fn, timeout=LOOK_WAIT):
        """Run fn(engine) on the engine's thread and return its result."""
        future = Future()
        self.jobs.put((fn, future))
        return future.result(timeout)

    def look(self):
        """PNG bytes of the next frame presented."""
        shot = Future()
        self.on_engine(lambda engine: engine._looks.append(shot))
        return shot.result(LOOK_WAIT)


def events(engine):
    """{event: [first docstring line of each handler]} for the events the app handles."""
    return {name: [((h.func.__doc__ or "").strip().splitlines() or [""])[0] for h in handlers]
            for name, handlers in sorted(engine._event_bus._handlers.items()) if handlers}


class _Handler(BaseHTTPRequestHandler):
    channel = None  # set per server

    def log_message(self, *args):  # requests stay out of the app's terminal
        pass

    def _send(self, code, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=repr, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"null") if n else None

    def _answer(self, fn):
        try:
            self._send(*fn())
        except (FutureTimeout, TimeoutError):
            self._send(504, {"error": "the app did not answer: is it paused on an error?"})
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"the body is not JSON: {e}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        ch = self.channel
        if url.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self._answer(lambda: (200, json.loads(ch.on_engine(
                lambda e: json.dumps(e.state, default=repr)))))
        if url.path == "/api/events":
            return self._answer(lambda: (200, ch.on_engine(events)))
        if url.path == "/api/log":
            return self._answer(lambda: (200, ch.lines.read(
                after=int(q["after"]) if "after" in q else None, match=q.get("match"),
                wait=float(q.get("wait", 0)), last=int(q.get("last", 50)))))
        if url.path == "/api/look":
            return self._answer(lambda: (200, ch.look(), "image/png"))
        self._send(404, {"error": f"no {url.path}"})

    def do_PATCH(self):
        if urlparse(self.path).path != "/api/state":
            return self._send(404, {"error": f"no {self.path}"})

        def patch():
            body = self._body()
            if not isinstance(body, dict):
                return 400, {"error": "a merge patch is a JSON object"}

            def apply(engine):
                merge_patch(engine.state, body)
                LOG.info("state patched: %s", json.dumps(body, default=repr)[:200])
                return json.loads(json.dumps(engine.state, default=repr))
            return 200, self.channel.on_engine(apply)
        self._answer(patch)

    def do_POST(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/events/"):
            return self._send(404, {"error": f"no {path}"})
        name = unquote(path.removeprefix("/api/events/"))

        def emit():
            payload = self._body()

            def fire(engine):
                known = events(engine)
                if name not in known:
                    return 404, {"error": f"no handler for {name!r}", "events": sorted(known)}
                LOG.info("event %s %s", name, json.dumps(payload, default=repr)[:200])
                engine.emit(name, payload)
                return 200, {"emitted": name, "handlers": len(known[name])}
            return self.channel.on_engine(fire)
        self._answer(emit)


def _bind(port, channel):
    handler = type("Handler", (_Handler,), {"channel": channel})
    for p in [port + i for i in range(10)] + [0] if port else [0]:
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), handler)
        except OSError:
            continue
    raise OSError("no free port for the dev server")


def server_url(root="."):
    """The URL of the dev server of the app running under manifoldx run in `root`, or None."""
    try:
        return json.loads((Path(root) / FOLDER / "server.json").read_text())["url"]
    except (OSError, ValueError, KeyError):
        return None


def save_png(device, texture, texture_format):
    """PNG bytes of a presented frame's texture."""
    w, h = texture.size[0], texture.size[1]
    data = device.queue.read_texture({"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
                                     {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    pixels = np.frombuffer(data, np.uint8).reshape(h, w, 4)
    return png(pixels[..., [2, 1, 0]] if str(texture_format).startswith("bgra") else pixels[..., :3])


def png(rgb):
    """PNG bytes of an (h, w, 3) uint8 image."""
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), np.ascontiguousarray(rgb).reshape(h, w * 3)], axis=1)

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>manifoldx run</title>
<style>
body{margin:0;font:14px/1.4 system-ui,sans-serif;background:#16161c;color:#e8e6f0}
header{padding:10px 16px;background:#1f1e27;display:flex;gap:16px;align-items:center}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:12px;padding:12px}
section{background:#1f1e27;border-radius:8px;padding:10px;min-width:0}
h2{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#a9a4c0}
img{width:100%;border-radius:4px;background:#000;min-height:120px}
pre,textarea{font:12px/1.35 ui-monospace,monospace;background:#121217;color:#e8e6f0;border:0;border-radius:4px;
  padding:8px;width:100%;box-sizing:border-box;margin:0}
#log{height:46vh;overflow:auto;white-space:pre-wrap}#state{max-height:40vh;overflow:auto}
button{background:#5b4fd6;color:#fff;border:0;border-radius:4px;padding:5px 10px;cursor:pointer;margin:4px 4px 0 0}
.ev{margin:6px 0}.ev small{color:#a9a4c0;display:block}.err{color:#ff8a8a}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><b>manifoldx run</b><span id="status"></span>
<label><input type="checkbox" id="auto" checked> live frame</label></header>
<main><div>
<section><h2>Frame</h2><img id="frame" alt="the frame on screen"><button onclick="look()">refresh</button></section>
<section><h2>Log</h2><pre id="log"></pre></section>
</div><div>
<section><h2>State</h2><pre id="state"></pre>
<textarea id="patch" rows="4" placeholder='merge patch, e.g. {"cam": {"yaw": 0.5}}'></textarea>
<button onclick="patch()">patch state</button><span id="perr" class="err"></span></section>
<section><h2>Events</h2><textarea id="payload" rows="3" placeholder="payload as JSON (optional)"></textarea>
<div id="events"></div></section>
</div></main>
<script>
const $ = id => document.getElementById(id);
let after = null;
async function get(u, o) { const r = await fetch(u, o); return r.json(); }
async function look() {
  const r = await fetch('/api/look');
  if (r.ok) { const old = $('frame').src; $('frame').src = URL.createObjectURL(await r.blob());
    if (old) URL.revokeObjectURL(old); $('status').textContent = ''; }
  else $('status').textContent = (await r.json()).error;
}
async function logLoop() {
  for (;;) {
    try {
      const q = after === null ? 'last=200' : 'after=' + after + '&wait=25';
      const d = await get('/api/log?' + q);
      after = d.next;
      const box = $('log'), end = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
      for (const l of d.lines) box.textContent += l.seq + ' ' + l.time + ' ' + l.text + '\\n';
      if (end) box.scrollTop = box.scrollHeight;
    } catch (e) { $('status').textContent = 'the app is gone'; await new Promise(r => setTimeout(r, 2000)); }
  }
}
async function state() {
  try { $('state').textContent = JSON.stringify(await get('/api/state'), null, 1); } catch (e) {}
}
async function patch() {
  $('perr').textContent = '';
  try { const d = await get('/api/state', {method: 'PATCH', body: $('patch').value});
    if (d.error) $('perr').textContent = d.error; else state(); }
  catch (e) { $('perr').textContent = String(e); }
}
async function events() {
  const d = await get('/api/events'), box = $('events'); box.textContent = '';
  for (const [name, docs] of Object.entries(d)) {
    const div = document.createElement('div'), b = document.createElement('button'), s = document.createElement('small');
    div.className = 'ev'; b.textContent = name; s.textContent = docs.filter(Boolean).join(' / ');
    b.onclick = async () => { const p = $('payload').value.trim();
      const r = await get('/api/events/' + encodeURIComponent(name), {method: 'POST', body: p || 'null'});
      if (r.error) alert(r.error); };
    div.append(b, s); box.append(div);
  }
}
logLoop(); state(); events(); look();
setInterval(state, 1000); setInterval(events, 5000);
setInterval(() => { if ($('auto').checked) look(); }, 1500);
</script></body></html>
"""
