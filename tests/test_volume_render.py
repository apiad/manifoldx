"""Volume render-pass integration tests (require an offscreen wgpu device)."""
import numpy as np
import pytest


def _make_offscreen_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_volume_shader_compiles_via_wgpu_create_shader_module():
    """The hand-written WGSL volume shader must pass wgpu validation."""
    from manifoldx.renderer import VOLUME_SHADER_SOURCE

    engine = _make_offscreen_engine()
    engine._device.create_shader_module(code=VOLUME_SHADER_SOURCE)


def test_volume_registry_creates_r32float_3d_texture_on_first_upload():
    """First call to upload_to_gpu creates a texture_3d r32float; clears dirty."""
    engine = _make_offscreen_engine()
    arr = np.zeros((4, 8, 16), dtype=np.float32)   # (Nz=4, Ny=8, Nx=16)
    arr[2, 4, 8] = 1.0
    handle = engine.register_volume(arr)
    res = engine._volume_registry.get(handle)
    assert res.dirty is True
    assert res.texture is None

    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)

    assert res.dirty is False
    assert res.texture is not None
    # Verify size mapping: numpy (Nz, Ny, Nx) → texture (Nx, Ny, Nz).
    assert res.texture.size == (16, 8, 4) or res.texture.size == [16, 8, 4]


def test_volume_registry_reuploads_only_when_dirty():
    """A second upload call without update_volume() is a no-op."""
    engine = _make_offscreen_engine()
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    handle = engine.register_volume(arr)
    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    tex_before = engine._volume_registry.get(handle).texture
    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    tex_after = engine._volume_registry.get(handle).texture
    assert tex_before is tex_after   # texture object not recreated


def test_volume_registry_update_triggers_reupload():
    """update_volume() flips dirty; upload_to_gpu writes new bytes; same texture object."""
    engine = _make_offscreen_engine()
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    handle = engine.register_volume(arr)
    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    res = engine._volume_registry.get(handle)
    tex_before = res.texture

    new = np.ones((4, 4, 4), dtype=np.float32)
    engine.update_volume(handle, new)
    assert res.dirty is True

    engine._volume_registry.upload_to_gpu(handle, engine._device.queue)
    assert res.dirty is False
    assert res.texture is tex_before    # reused, not reallocated
