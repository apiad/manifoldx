"""Depth-only shadow pass — renders mesh geometry from the sun's POV.

Runs before the main render pass (encoded at the command-encoder level in
Engine._draw_frame). Writes a depth-only shadow map that the StandardMaterial
fragment shader later samples. No-op unless both a sun (set_sun) and a shadow
config (enable_shadows) are present.
"""

import numpy as np
import wgpu


def _ensure_shadow_map(rp, resolution):
    if rp._shadow_map is not None and rp._shadow_map_size == resolution:
        return
    rp._shadow_map = rp._device.create_texture(
        size=(resolution, resolution, 1),
        format=wgpu.TextureFormat.depth24plus,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    rp._shadow_map_view = rp._shadow_map.create_view()
    rp._shadow_map_size = resolution


def _ensure_vs_layout(rp):
    if rp._shadow_bind_layout_vs is not None:
        return
    layout = rp._device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.VERTEX,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
        ]
    )
    rp._shadow_bind_layout_vs = layout
    rp._shadow_pipeline_layout = rp._device.create_pipeline_layout(bind_group_layouts=[layout])


def _get_pipeline(rp, stride):
    """One depth-only pipeline per vertex stride (24B pos+normal, 32B +uv)."""
    cache = getattr(rp, "_shadow_pipelines", None)
    if cache is None:
        cache = {}
        rp._shadow_pipelines = cache
    if stride in cache:
        return cache[stride]
    from manifoldx.renderer import _SHADOW_SHADER

    module = rp._device.create_shader_module(code=_SHADOW_SHADER)
    pipeline = rp._device.create_render_pipeline(
        layout=rp._shadow_pipeline_layout,
        vertex={
            "module": module,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": stride,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0}
                    ],
                }
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.back,
        },
        depth_stencil={
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        },
        fragment=None,
    )
    cache[stride] = pipeline
    return pipeline


def render_shadow_map(rp, engine, command_encoder):
    cfg = getattr(engine, "_shadow_config", None)
    sun = getattr(engine, "_sun", None)
    if cfg is None or sun is None or not rp._initialized:
        return

    _ensure_shadow_map(rp, cfg["resolution"])
    _ensure_vs_layout(rp)

    instances = rp._collect_mesh_instances(engine)
    if not instances:
        return

    # Upload all model matrices (column-major) once; record per-geometry offsets.
    all_mats, draw_info, offset = [], {}, 0
    for geom_id, mats in instances.items():
        mats_t = mats.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        draw_info[geom_id] = (offset, len(mats))
        all_mats.append(mats_t)
        offset += len(mats)
    rp._shadow_batch_buffers.upload_transforms(np.concatenate(all_mats).astype(np.float32))

    bind_group = rp._device.create_bind_group(
        layout=rp._shadow_bind_layout_vs,
        entries=[
            {"binding": 0, "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 352}},
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._shadow_batch_buffers.transforms_buf,
                    "offset": 0,
                    "size": rp._shadow_batch_buffers.transforms_capacity,
                },
            },
        ],
    )

    shadow_pass = command_encoder.begin_render_pass(
        color_attachments=[],
        depth_stencil_attachment={
            "view": rp._shadow_map_view,
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        },
    )
    shadow_pass.set_bind_group(0, bind_group)
    for geom_id, (first_instance, count) in draw_info.items():
        gpu = engine._geometry_registry.get_gpu_buffers(geom_id)
        if gpu is None:
            geom = engine._geometry_registry.get(geom_id)
            gpu = (
                engine._geometry_registry.create_buffers(geom_id, geom, rp._device.queue)
                if geom
                else None
            )
        if gpu is None:
            continue
        stride = gpu.get("stride", 6 * 4)
        shadow_pass.set_pipeline(_get_pipeline(rp, stride))
        shadow_pass.set_vertex_buffer(0, gpu["vertex_buffer"])
        shadow_pass.set_index_buffer(gpu["index_buffer"], wgpu.IndexFormat.uint32)
        shadow_pass.draw_indexed(
            gpu["index_count"], count, first_index=0, base_vertex=0, first_instance=first_instance
        )
    shadow_pass.end()
