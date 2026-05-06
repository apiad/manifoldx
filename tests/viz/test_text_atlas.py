"""Unit tests for LabelTextureAtlas (host-side rasterization + slot allocation)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from manifoldx.viz.text import LabelTextureAtlas


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_rasterize_string_shape_and_dtype():
    """A rasterized string returns a (64, 256, 4) uint8 RGBA tile."""
    tile = LabelTextureAtlas.rasterize_string("Hello", font_size=14)
    assert tile.shape == (64, 256, 4)
    assert tile.dtype == np.uint8


def test_rasterize_string_alpha_nonzero_for_glyphs():
    """Pixels covered by glyphs must have alpha > 0; transparent background must be alpha == 0."""
    tile = LabelTextureAtlas.rasterize_string("Hi", font_size=14)
    assert tile[0, 250, 3] == 0, "background pixel is not transparent"
    assert (tile[:, :40, 3] > 0).any(), "no visible glyphs found in left band"


def test_rasterize_string_text_is_white():
    """Visible pixels are white (255, 255, 255, alpha)."""
    tile = LabelTextureAtlas.rasterize_string("X", font_size=14)
    visible = tile[tile[..., 3] > 0]
    assert np.all(visible[:, :3] == 255), "non-white pixels found in glyph"


def test_rasterize_string_idempotent():
    """Calling rasterize_string twice with the same inputs returns identical bytes."""
    a = LabelTextureAtlas.rasterize_string("abc 123", font_size=14)
    b = LabelTextureAtlas.rasterize_string("abc 123", font_size=14)
    assert np.array_equal(a, b)


def test_rasterize_string_golden_hello_world():
    """Hash of `Hello, world` at 14pt matches committed golden hash.

    Detects silent font / PIL regressions. If this fails after a Pillow upgrade,
    review the diff and update the golden hash if intended.
    """
    tile = LabelTextureAtlas.rasterize_string("Hello, world", font_size=14)
    digest = hashlib.sha256(tile.tobytes()).hexdigest()
    golden_path = GOLDEN_DIR / "label_helloworld.png.sha256"
    expected = golden_path.read_text().strip()
    assert digest == expected, (
        f"rasterized 'Hello, world' hash drifted: {digest} != {expected}. "
        f"If this is intentional (font / PIL upgrade), regenerate the golden."
    )
