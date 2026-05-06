"""Unit tests for LabelMaterial."""
import numpy as np

from manifoldx.viz import LabelMaterial


def test_label_material_defaults():
    mat = LabelMaterial()
    assert mat.pixel_width == 256.0
    assert mat.pixel_height == 64.0
    assert mat.anchor_mode == "world"


def test_label_material_uniform_data_shape():
    mat = LabelMaterial(pixel_width=128, pixel_height=32)
    data = mat.get_data(n=5)
    assert data.shape == (5, 4)
    assert data.dtype == np.float32
    # Row 0 is the canonical row used by the renderer.
    np.testing.assert_array_equal(data[0], [128.0, 32.0, 0.0, 0.0])


def test_label_material_pipeline_subtype_is_world_or_screen():
    """The cache key includes anchor_mode so world and screen pipelines diverge."""
    a = LabelMaterial(anchor_mode="world").pipeline_subtype
    # `screen` is reserved for Plan 3; LabelMaterial accepts the literal but
    # keeps the renderer-side fallback to world for v2 (Plan 2).
    assert a == "world"


def test_label_material_compile_returns_wgsl_source():
    src = LabelMaterial._compile()
    assert "@vertex" in src
    assert "@fragment" in src
    assert "atlas_texture" in src


def test_label_material_uniform_type_layout():
    fields = LabelMaterial.uniform_type()
    assert list(fields.keys()) == ["pixel_width", "pixel_height", "anchor_mode", "_pad"]
    assert all(t == "f32" for t in fields.values())


def test_label_material_invalid_anchor_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        LabelMaterial(anchor_mode="bogus")


def test_label_material_shader_module_creates_without_error():
    """The shader source must compile in a real wgpu shader module."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    src = LabelMaterial._compile()
    module = device.create_shader_module(code=src)
    assert module is not None


def test_renderer_creates_label_pipeline_with_alpha_blend():
    """The pipeline factory must accept label=True and configure alpha blending."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    from manifoldx.renderer import RenderPipeline
    from manifoldx.viz import LabelMaterial

    rp = RenderPipeline.__new__(RenderPipeline)
    rp._pipelines = {}
    rp._bind_group_layouts = {}
    rp._pipeline_layouts = {}
    rp._material_buffers = {}

    mat = LabelMaterial()
    pipeline, layout = rp._get_or_create_pipeline(
        device,
        wgpu.TextureFormat.rgba8unorm_srgb,
        geometry_id=1,  # any non-zero
        material=mat,
        registry=None,
        label=True,
    )
    assert pipeline is not None
    assert layout is not None
    # Same call returns cached pipeline.
    pipeline_again, _ = rp._get_or_create_pipeline(
        device,
        wgpu.TextureFormat.rgba8unorm_srgb,
        geometry_id=1,
        material=mat,
        registry=None,
        label=True,
    )
    assert pipeline_again is pipeline
