"""Sci-viz materials.

ColormapMaterial: maps a per-instance scalar value through a 1D LUT.
Camera-facing point sprites with sphere-imposter normal reconstruction.
"""

import numpy as np
from typing import Dict

from manifoldx.resources import Material
from manifoldx.viz import colormaps


# WGSL shader source for ColormapMaterial.
#
# Bindings (group 0):
#   0: Globals uniform   { vp: mat4x4, view: mat4x4, proj: mat4x4, camera_pos: vec3, _pad: f32 }
#   1: transforms        storage<read> array<mat4x4<f32>>
#   2: material uniform  { vmin: f32, vmax: f32, lit_flag: f32, _pad: f32 }
#   3: scalar_values     storage<read> array<f32>
#   4: radii             storage<read> array<f32>
#   5: lut_texture       texture_1d<f32>
#   6: lut_sampler       sampler
#
# Vertex inputs:
#   @location(0) position: vec3<f32>   — quad-local in [-1, 1]^2 (z = 0)
#
# Vertex outputs / fragment inputs:
#   @location(0) quad_uv: vec2<f32>    — passes the quad-local xy for normal reconstruction
#   @location(1) scalar:  f32          — per-instance scalar value passed to fragment

_COLORMAP_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct MaterialUniform {
    vmin: f32,
    vmax: f32,
    lit_flag: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> scalar_values: array<f32>;
@group(0) @binding(4) var<storage, read> radii: array<f32>;
@group(0) @binding(5) var lut_texture: texture_1d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) scalar: f32,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let radius = radii[iidx];
    let scalar = scalar_values[iidx];

    // Center of the sprite in world space
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

    // Transform center to view space, then offset in view-space XY by quad-local
    // position scaled by radius. This makes the quad always face the camera.
    let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
    let offset = vec2<f32>(in.position.x, in.position.y) * radius;
    let view_pos = vec4<f32>(view_center.x + offset.x, view_center.y + offset.y, view_center.z, 1.0);

    // Project view-space position directly to clip space.
    let clip = globals.proj * view_pos;

    var out: VSOut;
    out.clip_position = clip;
    out.quad_uv = in.position.xy;  // [-1, 1]^2
    out.scalar = scalar;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }
    let n_view = vec3<f32>(in.quad_uv.x, in.quad_uv.y, sqrt(1.0 - r2));
    let denom = max(material.vmax - material.vmin, 1e-6);
    let t = clamp((in.scalar - material.vmin) / denom, 0.0, 1.0);
    let base_color = textureSample(lut_texture, lut_sampler, t);
    let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
    let lambert = max(dot(n_view, light_dir), 0.0);
    let lit_factor = mix(1.0, 0.2 + 0.8 * lambert, material.lit_flag);
    return vec4<f32>(base_color.rgb * lit_factor, base_color.a);
}
""".strip()


# WGSL shader source for LabelMaterial.
#
# Bindings (group 0):
#   0: Globals uniform   { vp: mat4x4, view: mat4x4, proj: mat4x4,
#                          camera_pos: vec3, _pad0: f32,
#                          viewport_size: vec2, _pad1: vec2 }
#   1: transforms        storage<read> array<mat4x4<f32>>
#   2: material uniform  { pixel_width: f32, pixel_height: f32, anchor_mode: f32, _pad: f32 }
#   3: label_indices     storage<read> array<f32>     # cast to u32 in shader
#   4: atlas_texture     texture_2d_array<f32>
#   5: atlas_sampler     sampler
#
# Vertex inputs:
#   @location(0) position: vec3<f32>   — quad-local in [-1, 1]^2 (z = 0)
#
# Vertex outputs:
#   @location(0) uv:    vec2<f32>      — texture UV in [0, 1]^2
#   @location(1) slice: f32            — label slice index (f32-encoded u32)

_LABEL_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    viewport_size: vec2<f32>,
    _pad1: vec2<f32>,
};

struct MaterialUniform {
    pixel_width: f32,
    pixel_height: f32,
    anchor_mode: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> label_indices: array<f32>;
@group(0) @binding(4) var atlas_texture: texture_2d_array<f32>;
@group(0) @binding(5) var atlas_sampler: sampler;

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) slice: f32,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

    var clip: vec4<f32>;

    if (material.anchor_mode < 0.5) {
        // World-anchored: project the entity's world position through the
        // camera, then add a billboard offset in view space scaled so the
        // label takes up `pixel_width` × `pixel_height` pixels on screen.
        let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
        let half_w_pixels = material.pixel_width * 0.5;
        let half_h_pixels = material.pixel_height * 0.5;
        let view_z = max(-view_center.z, 1e-3);
        let view_dx = in.position.x * half_w_pixels * 2.0 * view_z / globals.proj[0][0] / globals.viewport_size.x;
        let view_dy = in.position.y * half_h_pixels * 2.0 * view_z / globals.proj[1][1] / globals.viewport_size.y;
        let view_pos = vec4<f32>(
            view_center.x + view_dx,
            view_center.y + view_dy,
            view_center.z,
            1.0,
        );
        clip = globals.proj * view_pos;
    } else {
        // Screen-anchored: bypass view/proj entirely. Treat
        // Transform.pos.xy as an NDC anchor in [-1, 1] and add the
        // quad-local offset converted from pixels to NDC. NDC width
        // is 2 units across the full viewport, so 1 pixel = 2 / viewport
        // units; a quad-local x in [-1, 1] spans the full pixel_width
        // → NDC offset = quad_local.x * pixel_width / viewport_size.x.
        let ndc_dx = in.position.x * material.pixel_width / globals.viewport_size.x;
        let ndc_dy = in.position.y * material.pixel_height / globals.viewport_size.y;
        clip = vec4<f32>(world_center.x + ndc_dx, world_center.y + ndc_dy, 0.0, 1.0);
    }

    var out: VSOut;
    out.clip_position = clip;
    // Map quad position [-1, 1]^2 to UV [0, 1]^2 with V flipped so PIL's
    // top-left origin lands at the top of the rendered quad.
    out.uv = vec2<f32>(in.position.x * 0.5 + 0.5, 0.5 - in.position.y * 0.5);
    out.slice = label_indices[iidx];
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let layer = i32(in.slice + 0.5);
    let texel = textureSample(atlas_texture, atlas_sampler, in.uv, layer);
    return texel;
}
""".strip()


class ColormapMaterial(Material):
    """Point-sprite material that colormaps a per-instance scalar.

    Per-batch uniform (4 floats):
        vmin, vmax, lit_flag, _pad

    Per-instance storage buffers (read in shader by instance_index):
        transforms (mat4x4)         — existing
        scalar_values (float)       — new, sourced from ScalarValue
        radii (float)               — new, sourced from Radius

    Per-batch texture: 256x1 RGBA8 1D LUT, sampled in fragment shader.
    """

    binding_slot = 2

    def __init__(self, cmap: str, vmin: float, vmax: float, lit: bool = False):
        # Validate cmap exists
        colormaps.get_colormap(cmap)
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.lit = bool(lit)

    @classmethod
    def _compile(cls) -> str:
        return _COLORMAP_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"vmin": "f32", "vmax": "f32", "lit_flag": "f32", "_pad": "f32"}

    @property
    def pipeline_subtype(self) -> str:
        """Used by RenderPipeline cache to share pipelines across cmaps."""
        return self.cmap

    def get_data(self, n: int, registry=None) -> np.ndarray:
        """Per-batch material uniform data, broadcast to n rows.

        The renderer reads row 0 as the uniform; n rows are produced for
        compatibility with the existing material registry's per-instance
        data convention.
        """
        row = np.array(
            [self.vmin, self.vmax, 1.0 if self.lit else 0.0, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 4)).copy()

    def get_lut(self) -> np.ndarray:
        """Return the (256, 4) uint8 LUT for this material's colormap."""
        return colormaps.get_colormap(self.cmap)


_VALID_ANCHOR_MODES = ("world", "screen")


class LabelMaterial(Material):
    """Camera-facing billboard material textured with a label-atlas slice.

    Per-batch uniform (4 floats):
        pixel_width, pixel_height, anchor_mode, _pad
        (anchor_mode: 0.0 = world, 1.0 = screen — Plan 2 ships world only)

    Per-instance storage buffer:
        transforms (mat4x4)  — existing
        label_indices (f32)  — atlas slice index, cast to u32 in shader

    Texture binding: 2D texture array (TILE_WIDTH x TILE_HEIGHT x MAX_LABELS).

    Pipeline cache key: includes pipeline_subtype = anchor_mode so the
    world-anchored and screen-anchored pipelines stay separate even though
    they share a shader. (Screen-anchored is reserved for Plan 3.)
    """

    binding_slot = 3

    def __init__(
        self,
        *,
        pixel_width: float = 256.0,
        pixel_height: float = 64.0,
        anchor_mode: str = "world",
    ):
        if anchor_mode not in _VALID_ANCHOR_MODES:
            raise ValueError(
                f"LabelMaterial.anchor_mode must be one of {_VALID_ANCHOR_MODES}, got {anchor_mode!r}"
            )
        self.pixel_width = float(pixel_width)
        self.pixel_height = float(pixel_height)
        self.anchor_mode = anchor_mode

    @classmethod
    def _compile(cls) -> str:
        return _LABEL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "pixel_width": "f32",
            "pixel_height": "f32",
            "anchor_mode": "f32",
            "_pad": "f32",
        }

    @property
    def pipeline_subtype(self) -> str:
        return self.anchor_mode

    def get_data(self, n: int, registry=None) -> np.ndarray:
        anchor_f = 0.0 if self.anchor_mode == "world" else 1.0
        row = np.array(
            [self.pixel_width, self.pixel_height, anchor_f, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 4)).copy()
