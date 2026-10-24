"""Asset loading: Wavefront OBJ and Radiance HDR, from synthetic files in a temp folder."""

import os
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.harness import case, measure


def write_grid_obj(path: Path, n_tris: int) -> int:
    """A k x k vertex grid with v/vt/vn faces, about n_tris triangles. Returns the count."""
    k = max(2, int((n_tris / 2) ** 0.5) + 1)
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    p = np.stack([i.ravel() / k, np.sin(j.ravel() / 7.0), j.ravel() / k], 1)
    uv = np.stack([i.ravel() / k, j.ravel() / k], 1)
    a = (i[:-1, :-1] * k + j[:-1, :-1]).ravel() + 1
    b, c, d = a + 1, a + k, a + k + 1
    tris = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    with open(path, "w") as f:
        f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in p))
        f.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in uv))
        f.write("vn 0 1 0\n")
        f.write("".join(f"f {x}/{x}/1 {y}/{y}/1 {z}/{z}/1\n" for x, y, z in tris))
    return len(tris)


def write_rgbe(path: Path, width: int) -> None:
    """A width x width/2 new-style RLE RGBE file, each channel as literal runs of 128
    bytes: the fewest runs a decoder can meet, so a lower bound on its cost."""
    h = max(1, width // 2)
    img = np.random.default_rng(0).integers(0, 255, (h, width, 4), dtype=np.uint8)
    out = bytearray(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {width}\n".encode())
    for row in range(h):
        out += bytes([2, 2, width >> 8, width & 255])
        for ch in range(4):
            line = img[row, :, ch].tobytes()
            for s in range(0, width, 128):
                chunk = line[s : s + 128]
                out += bytes([len(chunk)]) + chunk
    path.write_bytes(bytes(out))


@case("loaders", "load_obj", sizes=[10_000, 100_000, 300_000], smoke=200)
def load_obj(n):
    """load_obj on a v/vt/vn triangle grid of about n triangles."""
    from manifoldx.assets.obj import load_obj as load

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grid.obj"
        tris = write_grid_obj(path, n)
        mb = os.path.getsize(path) / 1e6
        return {"ms": measure(lambda: load(path), repeat=3, number=1), "tris": tris, "MB": round(mb, 1)}


@case("loaders", "decode_rgbe", sizes=[512, 1_024, 2_048], smoke=64)
def decode_rgbe(n):
    """Decode an n x n/2 RLE HDR (best case for the decoder: literal runs of 128)."""
    from manifoldx.ibl import _decode_rgbe

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "env.hdr"
        write_rgbe(path, n)
        return measure(lambda: _decode_rgbe(str(path)), repeat=3, number=1)
