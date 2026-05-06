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


def test_get_or_create_returns_int_slice_index():
    atlas = LabelTextureAtlas()
    idx = atlas.get_or_create("alpha")
    assert isinstance(idx, int)
    assert idx == 0


def test_get_or_create_idempotent_per_string():
    atlas = LabelTextureAtlas()
    a = atlas.get_or_create("alpha")
    b = atlas.get_or_create("alpha")
    assert a == b
    assert atlas.slice_count == 1


def test_get_or_create_distinct_strings_get_distinct_slices():
    atlas = LabelTextureAtlas()
    a = atlas.get_or_create("alpha")
    b = atlas.get_or_create("beta")
    c = atlas.get_or_create("gamma")
    assert {a, b, c} == {0, 1, 2}
    assert atlas.slice_count == 3


def test_get_or_create_distinct_font_sizes_get_distinct_slices():
    atlas = LabelTextureAtlas()
    a = atlas.get_or_create("alpha", font_size=14)
    b = atlas.get_or_create("alpha", font_size=18)
    assert a != b
    assert atlas.slice_count == 2


def test_get_or_create_marks_slice_dirty():
    atlas = LabelTextureAtlas()
    atlas.get_or_create("alpha")
    assert 0 in atlas.dirty_slices
    atlas.clear_dirty()
    atlas.get_or_create("alpha")
    assert atlas.dirty_slices == set()


def test_get_or_create_overflow_at_max_labels():
    """The 257th unique label raises AtlasOverflowError."""
    from manifoldx.viz.text import AtlasOverflowError, MAX_LABELS

    atlas = LabelTextureAtlas()
    for i in range(MAX_LABELS):
        atlas.get_or_create(f"label_{i}")
    assert atlas.slice_count == MAX_LABELS
    with pytest.raises(AtlasOverflowError):
        atlas.get_or_create("one_too_many")


def test_rasterize_string_writes_into_slice_buffer():
    """get_or_create stores the rasterized bytes in the host slice array."""
    atlas = LabelTextureAtlas()
    idx = atlas.get_or_create("X", font_size=14)
    expected = LabelTextureAtlas.rasterize_string("X", font_size=14)
    assert np.array_equal(atlas._slices[idx], expected)
