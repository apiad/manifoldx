"""Reusable sky helpers for space scenes."""

from __future__ import annotations

from typing import Dict

import numpy as np

from manifoldx import random as _random
from manifoldx.components import Transform, Material
from manifoldx.viz import PointCloud, ColormapMaterial, ScalarValue, Radius


# Point-sprite star shader with a day/night fade. Stars stay full-bright in space
# and on the night side; they dim toward black under a daylit sky (the additive
# atmosphere then paints sky over them), gated by altitude so space is unaffected.
_STARFIELD_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
    ibl_intensity: f32, ibl_enabled: u32, _pad_ibl: vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>, _pad_sun0: f32,
    sun_color: vec3<f32>, sun_intensity: f32,
};
struct MaterialUniform { vmin: f32, vmax: f32, ground_radius: f32, atmo_top: f32 };

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> scalar_values: array<f32>;
@group(0) @binding(4) var<storage, read> radii: array<f32>;
@group(0) @binding(5) var lut_texture: texture_1d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;

struct VSIn { @location(0) position: vec3<f32> };
struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) scalar: f32,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let radius = radii[iidx];
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
    let offset = vec2<f32>(in.position.x, in.position.y) * radius;
    let view_pos = vec4<f32>(view_center.x + offset.x, view_center.y + offset.y, view_center.z, 1.0);
    var out: VSOut;
    out.clip_position = globals.proj * view_pos;
    out.quad_uv = in.position.xy;
    out.scalar = scalar_values[iidx];
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) { discard; }
    let denom = max(material.vmax - material.vmin, 1e-6);
    let t = clamp((in.scalar - material.vmin) / denom, 0.0, 1.0);
    let base_color = textureSample(lut_texture, lut_sampler, t);

    // Day/night + altitude fade: only dim under a daylit sky near the surface.
    var fade = 1.0;
    if (material.atmo_top > 0.0) {
        let cam_alt = length(globals.camera_pos) - material.ground_radius;
        let in_atmo = 1.0 - smoothstep(material.atmo_top, material.atmo_top * 2.5, cam_alt);
        let sun = normalize(-globals.sun_direction);
        let day = smoothstep(-0.08, 0.12, dot(normalize(globals.camera_pos), sun));
        fade = 1.0 - 0.97 * day * in_atmo;
    }
    if (fade < 0.35) { discard; }                 // fully daylit: let the sky show through
    return vec4<f32>(base_color.rgb * fade, base_color.a * fade);
}
""".strip()


class StarfieldMaterial(ColormapMaterial):
    """Star point-sprites that fade under a daylit sky (altitude-gated, so space stays starry).

    `ground_radius` / `atmo_top` describe the planet the stars orbit; leave
    `atmo_top = 0` for a plain never-fading starfield (deep-space scenes).
    """

    def __init__(self, cmap: str = "gray", vmin: float = 0.0, vmax: float = 1.0,
                 ground_radius: float = 0.0, atmo_top: float = 0.0):
        super().__init__(cmap, vmin, vmax, lit=False)
        self.ground_radius = float(ground_radius)
        self.atmo_top = float(atmo_top)

    @classmethod
    def _compile(cls) -> str:
        return _STARFIELD_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"vmin": "f32", "vmax": "f32", "ground_radius": "f32", "atmo_top": "f32"}

    def get_data(self, n: int, registry=None) -> np.ndarray:
        row = np.array([self.vmin, self.vmax, self.ground_radius, self.atmo_top], dtype=np.float32)
        return np.broadcast_to(row, (n, 4)).copy()


def starfield(engine, count: int = 1500, radius: float = 400.0, seed: int = 42,
              ground_radius: float = 0.0, atmo_top: float = 0.0):
    """Spawn `count` unlit star points on a sky sphere of `radius`.

    Uses the point-sprite path (grey colormap, per-star brightness + size), so it
    is a cheap static backdrop for any space scene. Pass `ground_radius`/`atmo_top`
    to fade the stars out under a daylit sky near a planet's surface (space stays
    starry). Returns the entity handle.
    """
    pos = _random.positions_on_sphere(count, radius=radius, rng=seed)
    bright = _random.scalars_uniform(count, low=0.65, high=1.0, rng=seed + 1)
    radii = _random.scalars_uniform(count, low=radius * 0.0010, high=radius * 0.0028, rng=seed + 2)
    return engine.spawn(
        PointCloud(),
        Material(StarfieldMaterial(cmap="gray", vmin=0.0, vmax=1.0,
                                   ground_radius=ground_radius, atmo_top=atmo_top)),
        Transform(pos=pos.astype(np.float32)),
        ScalarValue(value=bright),
        Radius(radius=radii),
        n=count,
    )
