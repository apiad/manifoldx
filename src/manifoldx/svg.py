"""SVG rasterization for GUI images and textures.

An SVG source is a path (str or os.PathLike), SVG markup (a str starting with
"<"), or bytes. `rasterize_svg` draws it at an exact pixel size with thorvg
(the optional extra `manifold-gfx[svg]`) and caches the result by
(content hash, width, height), so a frame that asks again costs a dict lookup.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from importlib.resources import files
from pathlib import Path
from typing import Callable, Union

import numpy as np

SvgSource = Union[str, bytes, os.PathLike]

# Bytes of rasterized pixels the cache keeps before it drops the least recently used.
CACHE_BUDGET = 64 * 1024 * 1024

_FONT_PATH = files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf")


class LruCache:
    """A dict with a byte budget: least recently used entries go first, each
    through `on_evict(value)`."""

    def __init__(self, budget: int, on_evict: Callable | None = None) -> None:
        self.budget = budget
        self.on_evict = on_evict
        self.nbytes = 0
        self._items: OrderedDict = OrderedDict()

    def get(self, key):
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value[0] if value is not None else None

    def put(self, key, value, nbytes: int) -> None:
        self._items[key] = (value, nbytes)
        self.nbytes += nbytes
        while self.nbytes > self.budget and len(self._items) > 1:
            _key, (old, size) = self._items.popitem(last=False)
            self.nbytes -= size
            if self.on_evict is not None:
                self.on_evict(old)

    def clear(self) -> None:
        for value, _size in self._items.values():
            if self.on_evict is not None:
                self.on_evict(value)
        self._items.clear()
        self.nbytes = 0

    def __contains__(self, key) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


_cache = LruCache(CACHE_BUDGET)
_engine = None


def _tvg():
    """The thorvg module and its process-wide engine, created on first use."""
    global _engine
    try:
        import thorvg_python as tvg
    except ImportError as e:
        raise ImportError(
            "SVG support needs thorvg-python: pip install 'manifold-gfx[svg]'"
        ) from e
    if _engine is None:
        _engine = tvg.Engine()
        load_font(str(_FONT_PATH))
    return tvg, _engine


def svg_bytes(source: SvgSource) -> bytes:
    """The SVG document behind `source`, as bytes."""
    if isinstance(source, bytes):
        return source
    if isinstance(source, str) and source.lstrip().startswith("<"):
        return source.encode("utf-8")
    return Path(source).read_bytes()


def svg_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _picture(data: bytes):
    tvg, engine = _tvg()
    picture = tvg.Picture(engine)
    if picture.load_data(data, "svg", None, True) != tvg.Result.SUCCESS:
        raise ValueError("could not parse SVG")
    return picture


def svg_size(source: SvgSource) -> tuple[float, float]:
    """The SVG's own size in pixels, from its width/height or its viewBox."""
    _result, w, h = _picture(svg_bytes(source)).get_size()
    return (float(w), float(h))


def load_font(path: str | os.PathLike) -> None:
    """Make a font file available to SVG <text>, matched by its family name."""
    tvg, engine = _tvg()
    if tvg.Text(engine).font_load(str(path)) != tvg.Result.SUCCESS:
        raise ValueError(f"could not load font {path}")


def _render(data: bytes, width: int, height: int) -> np.ndarray:
    """Draw the SVG stretched to width x height: RGBA uint8, straight alpha."""
    tvg, engine = _tvg()
    canvas = tvg.SwCanvas(engine)
    # ABGR8888S: RGBA bytes with straight alpha, as textures and the GUI shader take them.
    canvas.set_target(width, height, None, tvg.Colorspace.ABGR8888S)
    picture = _picture(data)
    picture.set_size(width, height)
    canvas.add(picture)
    canvas.update()
    canvas.draw(True)
    canvas.sync()
    rgba = np.frombuffer(canvas.buffer_arr, np.uint8).reshape(height, width, 4).copy()
    canvas.destroy()
    return rgba


def rasterize_digest(data: bytes, digest: str, width: int, height: int) -> np.ndarray:
    """`rasterize_svg` for a caller that already holds the bytes and their hash."""
    if width < 1 or height < 1:
        raise ValueError(f"SVG size must be at least 1x1, got {width}x{height}")
    key = (digest, width, height)
    rgba = _cache.get(key)
    if rgba is None:
        rgba = _render(data, width, height)
        _cache.put(key, rgba, rgba.nbytes)
    return rgba


def rasterize_svg(source: SvgSource, width: int, height: int) -> np.ndarray:
    """The SVG stretched to exactly width x height pixels, as a (height, width, 4)
    uint8 array: sRGB colour with straight alpha. Cached by content and size;
    keep the SVG's aspect ratio in the size you ask for (see `svg_size`)."""
    data = svg_bytes(source)
    return rasterize_digest(data, svg_digest(data), int(width), int(height))


def clear_cache() -> None:
    _cache.clear()
