"""Text in a font file, drawn by Pillow at its exact pixel size.

A Text widget with a `font` style key goes through the image path instead of the
label atlas: no 256-pixel tile limit, Pillow's kerning, and a raster per pixel
size, cached like SVG rasters.
"""

from __future__ import annotations

import functools
import hashlib
import math

import numpy as np

from manifoldx.svg import CACHE_BUDGET, LruCache

_cache = LruCache(CACHE_BUDGET)


@functools.lru_cache(maxsize=64)
def _font(path: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size=size)


@functools.lru_cache(maxsize=4096)
def measure(text: str, font: str, font_size: int) -> tuple[float, float]:
    """(width, height) of one line: the advance width, and ascent plus descent."""
    f = _font(font, font_size)
    ascent, descent = f.getmetrics()
    return (float(math.ceil(f.getlength(text))), float(ascent + descent))


def key(text: str, font: str, font_size: int) -> str:
    return "text:" + hashlib.sha1(f"{font}\0{font_size}\0{text}".encode()).hexdigest()


def rasterize(text: str, font: str, font_size: int, width: int, height: int) -> np.ndarray:
    """White text with straight alpha in a (height, width, 4) uint8 array, at the
    font size that makes the line `height` pixels tall. Cached by content and size."""
    k = (key(text, font, font_size), width, height)
    rgba = _cache.get(k)
    if rgba is None:
        from PIL import Image, ImageDraw

        _w, h = measure(text, font, font_size)
        size = max(1, round(font_size * height / h))
        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).text((0, 0), text, font=_font(font, size), fill=255)
        rgba = np.zeros((height, width, 4), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = np.asarray(img)
        _cache.put(k, rgba, rgba.nbytes)
    return rgba
