"""VolumeRegistry: CPU-side handle bookkeeping + numpy validation.

These tests exercise registry shape only (no GPU). Render-pipeline
integration is covered separately in tests/test_volume_render.py.
"""
import numpy as np
import pytest

from manifoldx.resources import VolumeRegistry


def _gaussian_blob(n=8):
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.exp(-(X**2 + Y**2 + Z**2) / 0.1).astype(np.float32)


def test_register_returns_sequential_handles_starting_at_one():
    reg = VolumeRegistry(device=None)
    a = reg.register(_gaussian_blob(), name="a")
    b = reg.register(_gaussian_blob(), name="b")
    assert (a, b) == (1, 2)


def test_register_rejects_non_3d_array():
    reg = VolumeRegistry(device=None)
    with pytest.raises(ValueError, match="3D"):
        reg.register(np.zeros((8, 8), dtype=np.float32))


def test_register_rejects_non_float32_with_hint():
    reg = VolumeRegistry(device=None)
    with pytest.raises(ValueError, match="float32"):
        reg.register(np.zeros((4, 4, 4), dtype=np.float64))


def test_register_rejects_non_contiguous():
    reg = VolumeRegistry(device=None)
    base = np.zeros((8, 8, 8, 2), dtype=np.float32)
    non_contig = base[..., 0]
    assert not non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        reg.register(non_contig)


def test_get_returns_resource_with_data_dirty_name():
    reg = VolumeRegistry(device=None)
    arr = _gaussian_blob()
    vol_id = reg.register(arr, name="blob")
    res = reg.get(vol_id)
    assert res.name == "blob"
    assert res.data is arr
    assert res.dirty is True   # newly registered → needs upload


def test_update_requires_matching_shape():
    reg = VolumeRegistry(device=None)
    vol_id = reg.register(_gaussian_blob(8))
    with pytest.raises(ValueError, match="shape"):
        reg.update(vol_id, _gaussian_blob(16))


def test_update_swaps_data_and_sets_dirty():
    reg = VolumeRegistry(device=None)
    vol_id = reg.register(_gaussian_blob(8))
    reg.get(vol_id).dirty = False     # simulate post-upload
    new = _gaussian_blob(8)
    reg.update(vol_id, new)
    res = reg.get(vol_id)
    assert res.data is new
    assert res.dirty is True


def test_unknown_handle_raises():
    reg = VolumeRegistry(device=None)
    with pytest.raises(KeyError, match="999"):
        reg.get(999)
