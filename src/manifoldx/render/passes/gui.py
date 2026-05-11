"""GUI render pass — runs last in the pipeline.

Issues at most two batched draws per frame (in Plan 1, only the first — the
glyph path lands in Task 7):

1. Rects — one instanced draw via `RectMaterial`.
2. Glyphs — one instanced draw reusing the screen-anchored label pipeline.

Pass attributes (set at pipeline-creation time in `_ensure_gui_pipeline`):
- depth-test off, depth-write off
- alpha blend on (pre-multiplied alpha)
- no culling
"""

from __future__ import annotations

import numpy as np
import wgpu

from manifoldx.gui.layout import LayoutBox, compute_layout
from manifoldx.gui.material import RectMaterial
from manifoldx.gui.painter import Painter, paint
from manifoldx.gui.widgets import Panel


# Unit-quad geometry shared by all rect instances (positions in [-0.5, 0.5]).
_UNIT_QUAD_VERTS = np.array(
    [
        [-0.5, -0.5, 0.0],
        [+0.5, -0.5, 0.0],
        [+0.5, +0.5, 0.0],
        [-0.5, +0.5, 0.0],
    ],
    dtype=np.float32,
)
_UNIT_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


def render_gui_pass(rp, engine, render_pass) -> None:
    """Entry point — called from RenderPipeline.render after the axis pass."""
    if not engine.gui:
        return

    viewport = LayoutBox(0.0, 0.0, float(engine.w), float(engine.h))

    painter = Painter()
    for panel in engine.gui:
        spec = panel.build_layout_spec()
        boxes = compute_layout(spec, viewport=_anchored(panel, viewport))
        paint(panel, spec, boxes, painter)

    _ensure_gui_pipeline(rp, engine)

    # Guarantee viewport_size is populated even when the entity pass short-
    # circuited early (no alive entities).  We write only the two floats we
    # need (bytes 208-215); the rest of the globals struct is irrelevant for
    # the GUI shader, which only reads viewport_size from it.
    _ensure_globals_viewport(rp, engine)

    # --- Rect draw ---
    rect_data = RectMaterial.pack_instances(painter.rect_ops)
    if rect_data.size > 0:
        _upload_rect_instances(rp, rect_data)
        _draw_rects(rp, engine, render_pass, rect_data.shape[0])

    # Glyph draw lands in Task 7.


def _ensure_globals_viewport(rp, engine) -> None:
    """Write viewport_size into the globals uniform buffer.

    The entity render passes upload the full 224-byte globals struct, which
    includes viewport_size at offset 208.  When there are no alive entities
    those passes short-circuit before the upload, leaving stale bytes in the
    buffer.  The GUI shader only reads viewport_size, so we write just those
    8 bytes unconditionally to keep the rect NDC conversion correct.
    """
    viewport_size = np.array([float(engine.w), float(engine.h)], dtype=np.float32)
    rp._device.queue.write_buffer(rp._globals_buffer, 208, viewport_size.tobytes())


def _anchored(panel: Panel, viewport: LayoutBox) -> LayoutBox:
    """Resolve the panel's root slot from its anchor, offset, and explicit size.

    The returned box is the *layout slot* that compute_layout assigns to the
    panel root. For panels with explicit width/height the slot is bounded to
    those dimensions; otherwise the full available viewport area is used so
    flex children can fill it.

    For now we anchor top-left only; the full anchor map lands here as the
    GUI accumulates examples that need it (`top-right`, `bottom-right`, ...).
    """
    ox, oy = panel.offset
    s = panel.effective_style()
    w = float(s["width"]) if s.get("width") is not None else viewport.w
    h = float(s["height"]) if s.get("height") is not None else viewport.h
    return LayoutBox(viewport.x + ox, viewport.y + oy, w, h)


def _ensure_gui_pipeline(rp, engine) -> None:
    if getattr(rp, "_gui_pipeline", None) is not None:
        return
    device = rp._device
    texture_format = engine._texture_format
    shader = device.create_shader_module(code=RectMaterial._compile())

    # Bind group: 0 = globals uniform, 1 = instance storage buffer.
    bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

    pipeline = device.create_render_pipeline(
        layout=layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 12,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {
                            "shader_location": 0,
                            "offset": 0,
                            "format": wgpu.VertexFormat.float32x3,
                        },
                    ],
                },
            ],
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": texture_format,
                    "blend": {
                        "color": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                        "alpha": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                    },
                    "write_mask": wgpu.ColorWrite.ALL,
                }
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        },
        # Depth format must match the render pass attachment even when
        # depth-test is logically off. We declare always-pass + no write.
        depth_stencil={
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.always,
        },
        multisample={
            "count": 1,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        },
    )
    rp._gui_pipeline = pipeline
    rp._gui_bgl = bgl

    # Unit-quad vertex + index buffers.
    rp._gui_quad_vbuf = device.create_buffer_with_data(
        data=_UNIT_QUAD_VERTS.tobytes(),
        usage=wgpu.BufferUsage.VERTEX,
    )
    rp._gui_quad_ibuf = device.create_buffer_with_data(
        data=_UNIT_QUAD_INDICES.tobytes(),
        usage=wgpu.BufferUsage.INDEX,
    )
    rp._gui_instance_buf = None
    rp._gui_instance_capacity = 0


def _upload_rect_instances(rp, data: np.ndarray) -> None:
    needed = data.nbytes
    if rp._gui_instance_buf is None or needed > rp._gui_instance_capacity:
        rp._gui_instance_buf = rp._device.create_buffer(
            size=max(needed, 4096),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        rp._gui_instance_capacity = max(needed, 4096)
    rp._device.queue.write_buffer(rp._gui_instance_buf, 0, data.tobytes())


def _draw_rects(rp, engine, render_pass, instance_count: int) -> None:
    bind_group = rp._device.create_bind_group(
        layout=rp._gui_bgl,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": rp._globals_buffer,
                    "offset": 0,
                    "size": 224,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._gui_instance_buf,
                    "offset": 0,
                    "size": rp._gui_instance_capacity,
                },
            },
        ],
    )
    render_pass.set_pipeline(rp._gui_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 0)
    render_pass.set_vertex_buffer(0, rp._gui_quad_vbuf)
    render_pass.set_index_buffer(rp._gui_quad_ibuf, wgpu.IndexFormat.uint32)
    render_pass.draw_indexed(6, instance_count, 0, 0, 0)
