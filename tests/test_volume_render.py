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


def _render_one_frame(engine):
    """Drive one frame and read back the framebuffer."""
    engine._draw_frame()
    return engine._render_canvas.draw()   # (H, W, 4) uint8


def _capture_background(engine):
    """Render the current scene without volumes and return a frame copy."""
    return _render_one_frame(engine).copy()


def test_centered_blob_renders_visible_pixels_at_origin():
    """Spawn a 32^3 Gaussian blob centered at the origin; render one frame;
    the framebuffer's central pixel must show the volume (non-clear color),
    while the corner must remain unchanged from the clear color (`discard`
    preserves the framebuffer outside the projected box bounds).
    """
    from manifoldx.components import Material, Transform
    from manifoldx.viz import Volume, VolumeMaterial

    engine = _make_offscreen_engine(width=64, height=64)
    engine.camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)

    bg = _capture_background(engine)   # empty-scene background

    n = 32
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    density = np.exp(-(X**2 + Y**2 + Z**2) / 0.1).astype(np.float32)

    handle = engine.register_volume(density)
    engine.spawn(
        Volume(volume_id=handle),
        Material(VolumeMaterial(
            cmap="inferno",
            opacity_stops=[(0.0, 0.0), (0.3, 0.05), (1.0, 0.6)],
            step_size=0.02,
        )),
        Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
        n=1,
    )

    rgba = _render_one_frame(engine)
    assert not np.array_equal(rgba[32, 32], bg[32, 32])   # center modified
    assert np.array_equal(rgba[0, 0], bg[0, 0])           # corner unchanged


def test_two_entities_share_volume_handle():
    """Same vol_id with different Transform.pos → two visually separate regions."""
    from manifoldx.components import Material, Transform
    from manifoldx.viz import Volume, VolumeMaterial

    engine = _make_offscreen_engine(width=128, height=64)
    engine.camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)

    bg = _capture_background(engine)

    arr = np.ones((4, 4, 4), dtype=np.float32)   # uniform-1 box
    handle = engine.register_volume(arr)
    for px in (-3.5, +3.5):
        engine.spawn(
            Volume(volume_id=handle),
            Material(VolumeMaterial(
                cmap="viridis",
                opacity_stops=[(0.0, 0.0), (1.0, 1.0)],
                step_size=0.05,
            )),
            Transform(pos=(px, 0, 0), scale=(1.0, 1.0, 1.0)),
            n=1,
        )

    rgba = _render_one_frame(engine)
    # Left and right halves must each have at least one modified pixel;
    # the center column must be unchanged from the empty-scene background.
    left_changed = (rgba[32, :32] != bg[32, :32]).any(axis=-1)
    right_changed = (rgba[32, -32:] != bg[32, -32:]).any(axis=-1)
    assert left_changed.any()
    assert right_changed.any()
    assert np.array_equal(rgba[32, 64], bg[32, 64])
