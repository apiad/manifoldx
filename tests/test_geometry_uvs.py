import numpy as np
from manifoldx.resources import sphere, plane


def test_sphere_emits_uvs():
    geo = sphere(1.0, segments=8)
    assert "uvs" in geo
    uvs = geo["uvs"]
    assert uvs.dtype == np.float32
    assert uvs.shape == (geo["positions"].shape[0], 2)
    assert uvs.min() >= 0.0
    assert uvs.max() <= 1.0


def test_plane_emits_uvs():
    geo = plane(2.0, 2.0)
    assert "uvs" in geo
    uvs = geo["uvs"]
    assert uvs.shape == (4, 2)
    expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    np.testing.assert_allclose(uvs, expected)


def test_create_buffers_records_stride_when_uvs_present():
    """Geometry with UVs gets stride=32 (pos+normal+uv); without UVs, stride=24."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    from manifoldx.resources import GeometryRegistry, sphere

    reg = GeometryRegistry(device=device)
    geo = sphere(1.0, segments=4)
    geom_id = reg.register(geo)
    buffers = reg.create_buffers(geom_id, geo, device.queue)
    assert buffers["stride"] == 32
    assert buffers["has_uvs"] is True
    assert buffers["has_normals"] is True


def test_scalar_standard_material_renders_on_uv_sphere():
    """Regression: sphere() now emits UVs (stride-32 buffer), but a scalar
    StandardMaterial pipeline must still render correctly on it — the
    pipeline reads positions+normals from the 32-byte stride and ignores
    the UV bytes."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    from manifoldx.resources import StandardMaterial, PointLight, sphere

    engine = mx.Engine("scalar-on-uv-sphere", width=64, height=64)
    engine._init_canvas(canvas)
    engine._running = True

    sphere_geo = sphere(1.0, segments=8)
    mat = StandardMaterial(color=(1, 0, 0), roughness=0.5, metallic=0.0)
    engine.spawn(Mesh(sphere_geo), Material(mat), Transform(pos=(0, 0, 0)))
    engine.add_light(PointLight(position=(2, 2, 2), color="#ffffff", intensity=20.0))

    # If _draw_frame raises, the stride mismatch (or similar) regressed.
    engine._draw_frame()
