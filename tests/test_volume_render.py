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
