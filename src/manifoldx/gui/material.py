"""WGSL signed-distance rounded-rect material for the GUI render pass.

One instanced draw call → one rounded rect per instance, with optional 1 px
(or wider) border. Per-instance state is packed into a 16-float row that
mirrors the WGSL ``RectInstance`` struct layout (including implicit padding)::

    [xy.x, xy.y, size.x, size.y, radius, border, _pad, _pad,
     bg.r, bg.g, bg.b, bg.a,
     border_color.r, border_color.g, border_color.b, border_color.a]

Total: 16 floats = 64 bytes per instance (8 bytes implicit padding after
``border`` so that ``bg`` (vec4) aligns to 16 bytes). The vertex shader
expands a
unit quad in [-0.5, 0.5]^2 to the instance's pixel size at the instance's
top-left position; the fragment shader evaluates the rounded-rect SDF.
"""

from __future__ import annotations

from typing import Any

from manifoldx.resources import Material


_RECT_WGSL = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    viewport_size: vec2<f32>,
    _pad1: vec2<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct RectInstance {
    xy:           vec2<f32>,
    size:         vec2<f32>,
    radius:       f32,
    border:       f32,
    bg:           vec4<f32>,
    border_color: vec4<f32>,
};

@group(0) @binding(1) var<storage, read> instances: array<RectInstance>;

struct VsIn {
    @location(0) position: vec3<f32>,  // unit-quad corner in [-0.5, 0.5]
    @builtin(instance_index) instance_id: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,   // pixel offset from rect center
    @location(1) flat_id: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let inst = instances[in.instance_id];
    // Pixel-space center + corner.
    let center_px = inst.xy + inst.size * 0.5;
    let corner_px = center_px + vec2<f32>(in.position.x, in.position.y) * inst.size;
    // Convert pixel coords -> NDC. Viewport origin is top-left; flip Y.
    let ndc_x = (corner_px.x / globals.viewport_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (corner_px.y / globals.viewport_size.y) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local = vec2<f32>(in.position.x, in.position.y) * inst.size;
    out.flat_id = in.instance_id;
    return out;
}

fn rounded_rect_sdf(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) +
           min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let inst = instances[in.flat_id];
    let half = inst.size * 0.5;
    let d = rounded_rect_sdf(in.local, half, inst.radius);
    // 1 px smoothstep gives cheap AA at the outer edge.
    let outer = 1.0 - smoothstep(-1.0, 0.0, d);
    var color = inst.bg;
    if (inst.border > 0.0) {
        let inside_d = d + inst.border;
        let inner = 1.0 - smoothstep(-1.0, 0.0, inside_d);
        let border_mask = clamp(outer - inner, 0.0, 1.0);
        color = inst.bg * inner + inst.border_color * border_mask;
        // Pre-multiplied alpha for blending.
        color.a = outer;
    } else {
        color = inst.bg * outer;
        color.a = inst.bg.a * outer;
    }
    return color;
}
"""


class RectMaterial(Material):
    """Material for the GUI rect path. One instance per rounded rectangle."""

    binding_slot = 1  # storage buffer at @group(0) @binding(1)

    @classmethod
    def _compile(cls) -> str:
        return _RECT_WGSL

    @classmethod
    def uniform_type(cls) -> dict[str, str]:
        return {
            "xy": "vec2<f32>",
            "size": "vec2<f32>",
            "radius": "f32",
            "border": "f32",
            "bg": "vec4<f32>",
            "border_color": "vec4<f32>",
        }

    @property
    def pipeline_subtype(self) -> str:
        return "gui"

    @staticmethod
    def pack_instances(rect_ops: list[Any]) -> "np.ndarray":  # noqa: F821
        """Pack a list of painter.RectOp into a (N, 16) float32 buffer.

        Column order matches the WGSL `RectInstance` struct layout including
        implicit alignment padding.  WGSL aligns vec4 fields to 16 bytes, so
        the two f32 scalars (radius, border) at offset 16/20 are followed by
        8 bytes of implicit padding before `bg` (vec4, offset 32).  We
        represent the full 64-byte struct as 16 float32 values:

            [xy.x, xy.y, size.x, size.y, radius, border, _pad, _pad,
             bg.r, bg.g, bg.b, bg.a,
             border_color.r, border_color.g, border_color.b, border_color.a]
        """
        import numpy as np

        if not rect_ops:
            return np.zeros((0, 16), dtype=np.float32)
        rows = []
        for op in rect_ops:
            rows.append(
                [
                    op.box.x,
                    op.box.y,
                    op.box.w,
                    op.box.h,
                    op.radius,
                    op.border,
                    0.0,  # padding to align bg to 16-byte boundary
                    0.0,  # padding
                    op.fill[0],
                    op.fill[1],
                    op.fill[2],
                    op.fill[3],
                    op.border_color[0],
                    op.border_color[1],
                    op.border_color[2],
                    op.border_color[3],
                ]
            )
        return np.asarray(rows, dtype=np.float32)
