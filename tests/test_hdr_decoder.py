"""Pinned output of the Radiance RGBE decoder (manifoldx#117) on small files."""

import numpy as np
import pytest

from manifoldx.ibl import _decode_rgbe


def _expected(img):
    """RGBE bytes (H, W, 4) to linear float32 RGB: mantissa * 2^(E - 136), 0 where E is 0."""
    e = img[..., 3].astype(int)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return (img[..., :3] * scale[..., None]).astype(np.float32)


def _rle_channel(line, runs):
    """Encode one channel of one scanline; runs is a list of ("run"|"lit", length)."""
    out, x = bytearray(), 0
    for kind, n in runs:
        if kind == "run":
            out += bytes([128 + n, line[x]])
        else:
            out += bytes([n]) + bytes(line[x : x + n])
        x += n
    assert x == len(line)
    return bytes(out)


def _write(path, img, rows):
    """rows[r] is "flat" or a list of runs used for every channel of row r."""
    h, w, _ = img.shape
    out = bytearray(b"#?RADIANCE\n# made by a test\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1.0\n\n")
    out += f"-Y {h} +X {w}\n".encode()
    for r, mode in enumerate(rows):
        if mode == "flat":
            out += img[r].tobytes()
        else:
            out += bytes([2, 2, w >> 8, w & 255])
            for ch in range(4):
                out += _rle_channel(img[r, :, ch].tobytes(), mode)
    path.write_bytes(bytes(out))


def _image(h, w, seed=0):
    img = np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)
    img[..., 0] = np.where(img[..., 0] == 2, 3, img[..., 0])  # a flat row must not start like a marker
    img[0, 0, 3] = 0  # E = 0 decodes to black
    return img


def test_flat_scanlines_narrow_image(tmp_path):
    img = _image(3, 5)
    _write(tmp_path / "a.hdr", img, ["flat"] * 3)
    out = _decode_rgbe(str(tmp_path / "a.hdr"))
    assert out.dtype == np.float32 and out.shape == (3, 5, 3)
    np.testing.assert_array_equal(out, _expected(img))
    assert (out[0, 0] == 0).all()


def test_new_style_rle_runs_and_literals(tmp_path):
    img = _image(4, 20, seed=1)
    for r in range(4):  # make the run stretches constant so they can be run-coded
        img[r, 3:10] = img[r, 3]
        img[r, 14:20] = img[r, 14]
    runs = [("lit", 3), ("run", 7), ("lit", 1), ("lit", 3), ("run", 6)]
    _write(tmp_path / "b.hdr", img, [runs] * 4)
    np.testing.assert_array_equal(_decode_rgbe(str(tmp_path / "b.hdr")), _expected(img))


def test_runs_of_the_longest_lengths(tmp_path):
    img = _image(2, 300, seed=2)
    img[:, 128:255] = img[:, 128:129]
    runs = [("lit", 128), ("run", 127), ("lit", 45)]
    _write(tmp_path / "c.hdr", img, [runs] * 2)
    np.testing.assert_array_equal(_decode_rgbe(str(tmp_path / "c.hdr")), _expected(img))


@pytest.mark.parametrize("rows", [["flat", "rle", "flat"], ["rle", "flat", "rle"]])
def test_flat_and_rle_scanlines_in_one_file(tmp_path, rows):
    img = _image(3, 16, seed=3)
    rows = [[("lit", 16)] if r == "rle" else r for r in rows]
    _write(tmp_path / "d.hdr", img, rows)
    np.testing.assert_array_equal(_decode_rgbe(str(tmp_path / "d.hdr")), _expected(img))
