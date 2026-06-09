import pytest


def _make_engine_with_canvas():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import manifoldx as mx
    engine = mx.Engine("cache-key-test", width=64, height=64)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_mesh_cache_key_includes_subtype():
    """Scalar StandardMaterial and a subtype-variant material on the same
    geometry must produce different pipeline-cache keys."""
    engine = _make_engine_with_canvas()
    rp = engine._render_pipeline

    from manifoldx.resources import StandardMaterial, sphere

    geo = sphere(1.0, segments=4)
    geom_id = engine._geometry_registry.register(geo)
    engine._geometry_registry.create_buffers(geom_id, geo, engine._device.queue)
    gpu_buffers = engine._geometry_registry.get_gpu_buffers(geom_id)

    # Now that StandardMaterial supports albedo_map, use a real TextureHandle.
    from manifoldx.textures import TextureHandle
    fake = TextureHandle(id=99, texture=object(), view=object(),
                         sampler=object(), size=(1, 1))
    mat_scalar = StandardMaterial(color="#ff0000")
    mat_variant = StandardMaterial(color="#ff0000", albedo_map=fake)
    assert mat_variant.pipeline_subtype == "textured"

    rp._get_or_create_pipeline(
        engine._device, engine._texture_format, geom_id, mat_scalar,
        engine._material_registry, geometry_buffers=gpu_buffers,
    )
    rp._get_or_create_pipeline(
        engine._device, engine._texture_format, geom_id, mat_variant,
        engine._material_registry, geometry_buffers=gpu_buffers,
    )

    keys = list(rp._pipelines.keys())
    matching = [k for k in keys if k[0] == geom_id and "StandardMaterial" in str(k)]
    # Two distinct cache entries for the same (geom_id, StandardMaterial) pair
    # with different subtypes.
    assert len(matching) >= 2, f"expected 2+ cache entries, got {len(matching)}: {keys}"
