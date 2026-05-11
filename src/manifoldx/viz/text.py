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
    def measure_string(
        text: str, font_size: int = 14
    ) -> tuple[int, int, float, float, float, float]:
        """Return (glyph_w_px, glyph_h_px, u0, v0, u1, v1) for `text` at `font_size`.

        glyph_w_px, glyph_h_px: actual pixel extents PIL rasterizes inside the tile.
        u0, v0, u1, v1: normalized UV bounds within the tile (suitable for clamping
        shader sampling to the live glyph region).

        Mirrors the placement logic in `rasterize_string` so callers can compute
        the glyph's location in the tile without re-rasterizing.
        """
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(str(_FONT_PATH), size=font_size)
        img = Image.new("RGBA", (TILE_WIDTH, TILE_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = 4
        y = (TILE_HEIGHT - text_h) // 2 - bbox[1]
        # Pixel bbox of the actual glyph within the tile.
        px_x0 = x + bbox[0]
        px_y0 = y + bbox[1]
        px_x1 = px_x0 + text_w
        px_y1 = px_y0 + text_h
        return (
            text_w,
            text_h,
            px_x0 / TILE_WIDTH,
            px_y0 / TILE_HEIGHT,
            px_x1 / TILE_WIDTH,
            px_y1 / TILE_HEIGHT,
        )

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

    def register_colormap_legend(
        self, cmap_name: str, *, orientation: str = "horizontal"
    ) -> int:
        """Rasterize a colormap LUT as an atlas slice and return its index.

        The atlas tile is 256×64 (4:1 aspect), so a horizontal legend maps
        directly: column i = LUT[i] for i in [0, 255]. A vertical legend
        subsamples the 256-entry LUT to 64 rows.

        Plan 4's `colormap_legend(...)` shim composes this with TextLabel
        tick annotations. Plan 3 just ships the rendering capability —
        Spawn a screen-anchored TextLabel pointing at the returned slot and
        the legend draws via the standard label render path; no new
        material or pipeline is needed.

        Idempotent: same cmap_name + orientation returns the same slot.
        """
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
            )
        # Use a sentinel key so colormap legends never collide with text labels.
        key = (f"\x00cmap_legend\x00{cmap_name}\x00{orientation}", 0)
        if key in self._index:
            return self._index[key]
        if self._slice_count >= MAX_LABELS:
            raise AtlasOverflowError(
                f"label atlas full at {MAX_LABELS} unique labels (v1 cap). "
                f"Refused to add {key!r}."
            )

        from manifoldx.viz import colormaps

        lut = colormaps.get_colormap(cmap_name)  # (256, 4) uint8
        if orientation == "horizontal":
            # Tile each LUT row across all 64 atlas rows: column i → LUT[i].
            tile = np.broadcast_to(lut[None, :, :], (TILE_HEIGHT, TILE_WIDTH, 4)).copy()
        else:
            # Subsample LUT to 64 entries; bottom row = LUT[0] (low value),
            # top row = LUT[255] (high value). Then broadcast across columns.
            sample_idx = np.linspace(255, 0, TILE_HEIGHT, dtype=np.int32)
            sampled = lut[sample_idx]  # (64, 4)
            tile = np.broadcast_to(sampled[:, None, :], (TILE_HEIGHT, TILE_WIDTH, 4)).copy()

        idx = self._slice_count
        self._slices[idx] = tile
        self._index[key] = idx
        self._dirty_slices.add(idx)
        self._slice_count += 1
        return idx

    @property
    def gpu_texture(self):
        return self._gpu_texture

    @property
    def gpu_sampler(self):
        return self._gpu_sampler

    def upload_dirty(self, device, queue) -> None:
        """Create the texture array on first call; upload each dirty slice.

        The texture is allocated once at MAX_LABELS slices so subsequent
        `get_or_create` calls just write into already-allocated slots.
        """
        import wgpu

        if not self._dirty_slices and self._gpu_texture is not None:
            return

        if self._gpu_texture is None:
            self._gpu_texture = device.create_texture(
                size=(TILE_WIDTH, TILE_HEIGHT, MAX_LABELS),
                format=wgpu.TextureFormat.rgba8unorm_srgb,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                dimension=wgpu.TextureDimension.d2,
            )
            self._gpu_sampler = device.create_sampler(
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
                address_mode_v=wgpu.AddressMode.clamp_to_edge,
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
            )

        for idx in sorted(self._dirty_slices):
            slice_data = self._slices[idx]
            queue.write_texture(
                {
                    "texture": self._gpu_texture,
                    "origin": (0, 0, idx),
                },
                slice_data.tobytes(),
                {
                    "bytes_per_row": TILE_WIDTH * 4,
                    "rows_per_image": TILE_HEIGHT,
                },
                (TILE_WIDTH, TILE_HEIGHT, 1),
            )
        self._dirty_slices.clear()
