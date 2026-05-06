"""Text rasterization for sci-viz labels.

LabelTextureAtlas owns a 2D texture array of rasterized strings. Each unique
(text, font_size) pair occupies one slice (256x64 RGBA8). The host-side cache
holds the pixel data; upload_dirty pushes new slices to the GPU.
"""

from importlib.resources import files
from typing import Dict, Tuple

import numpy as np

# Fixed slice geometry for v1 — design §8.
TILE_WIDTH = 256
TILE_HEIGHT = 64
MAX_LABELS = 256

_FONT_PATH = files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf")


class AtlasOverflowError(RuntimeError):
    """Raised when a 257th unique label is requested in v1."""


class LabelTextureAtlas:
    """Host-side cache + lazy GPU texture array for rasterized labels."""

    def __init__(self):
        self._slices: np.ndarray = np.zeros(
            (MAX_LABELS, TILE_HEIGHT, TILE_WIDTH, 4), dtype=np.uint8
        )
        self._slice_count: int = 0
        self._index: Dict[Tuple[str, int], int] = {}
        self._dirty_slices: set[int] = set()
        self._gpu_texture = None
        self._gpu_sampler = None

    @staticmethod
    def rasterize_string(text: str, font_size: int = 14) -> np.ndarray:
        """Rasterize `text` via PIL into an RGBA8 (TILE_HEIGHT, TILE_WIDTH, 4) tile.

        White glyphs on transparent background. Text is left-aligned with a
        small left margin and vertically centered.
        """
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(str(_FONT_PATH), size=font_size)
        img = Image.new("RGBA", (TILE_WIDTH, TILE_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        x = 4
        y = (TILE_HEIGHT - text_h) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        return np.array(img, dtype=np.uint8)

    @property
    def slice_count(self) -> int:
        return self._slice_count

    @property
    def dirty_slices(self) -> set[int]:
        return self._dirty_slices

    def clear_dirty(self) -> None:
        self._dirty_slices.clear()

    def get_or_create(self, text: str, font_size: int = 14) -> int:
        key = (text, font_size)
        if key in self._index:
            return self._index[key]
        if self._slice_count >= MAX_LABELS:
            raise AtlasOverflowError(
                f"label atlas full at {MAX_LABELS} unique labels (v1 cap). "
                f"Refused to add {key!r}."
            )
        idx = self._slice_count
        self._slices[idx] = self.rasterize_string(text, font_size=font_size)
        self._index[key] = idx
        self._dirty_slices.add(idx)
        self._slice_count += 1
        return idx
