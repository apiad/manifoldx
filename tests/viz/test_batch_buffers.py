"""Unit test for _BatchBuffers extension to label_indices."""
import numpy as np
import pytest


def _get_offscreen_device():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter.request_device_sync()


def test_batch_buffers_upload_label_indices():
    from manifoldx.renderer import _BatchBuffers

    device = _get_offscreen_device()
    bufs = _BatchBuffers(device)
    data = np.array([0, 1, 2, 3], dtype=np.float32)
    bufs.upload_label_indices(data)
    assert bufs.label_indices_buf is not None
    assert bufs.label_indices_capacity >= data.nbytes


def test_batch_buffers_label_indices_grows_capacity():
    from manifoldx.renderer import _BatchBuffers

    device = _get_offscreen_device()
    bufs = _BatchBuffers(device)
    bufs.upload_label_indices(np.zeros(4, dtype=np.float32))
    cap_small = bufs.label_indices_capacity
    bufs.upload_label_indices(np.zeros(64, dtype=np.float32))
    assert bufs.label_indices_capacity > cap_small
