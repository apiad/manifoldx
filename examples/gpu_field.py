"""GPU fields — evaluate a composable `Field` live on the GPU.

The *same* field algebra that bakes terrain on the CPU (`Mesh.displace`) can be
transpiled to WGSL with `field_to_wgsl` and evaluated in a shader. Here a flat
plane is displaced in the vertex shader by a developer-composed terrain field
(ridged mountains + fbm, domain-warped), with normals from finite differences
of the same field — no CPU displacement, no baked heightmap.

    uv run python examples/gpu_field.py
    uv run python examples/gpu_field.py --render --duration 8 --output /tmp/gpu_field.mp4
"""

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields, field_to_wgsl
from manifoldx.resources import Material as _Material, DirectionalLight

# A developer-composed terrain field — exactly the kind you'd pass to Mesh.displace.
TERRAIN = (
    fields.ridged(seed=4, freq=0.16, octaves=5) * 0.75
    + fields.fbm(seed=8, freq=0.5, octaves=5) * 0.25
).warp(0.8, fx=fields.fbm(2, 0.3), fz=fields.fbm(9, 0.3)).remap(0.0, 1.0, -1.2, 3.2)

FIELD_WGSL = field_to_wgsl(TERRAIN, name="terrain_field")

_SHADER = FIELD_WGSL + """

struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
    ibl_intensity: f32, ibl_enabled: u32, _pad_ibl: vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>, _pad_sun0: f32,
    sun_color: vec3<f32>, sun_intensity: f32,
};
struct Transforms { models: array<mat4x4<f32>> };
struct FieldUniforms { params: vec4<f32> };   // x=amplitude, y=finite-diff epsilon

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: FieldUniforms;

// Baked in the vertex stage (binding 2 is fragment-visible only in the layout).
const _AMP: f32 = 1.0;
const _EPS: f32 = 0.03;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) height: f32,
};

fn height_at(p: vec3<f32>) -> f32 {
    return terrain_field(p) * _AMP;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let model = transforms.models[in.instance];
    let base = (model * vec4<f32>(in.position, 1.0)).xyz;
    let h = height_at(base);
    let world = vec3<f32>(base.x, base.y + h, base.z);

    // Normal from finite differences of the field (central differences in x/z).
    let e = _EPS;
    let hx = height_at(base + vec3<f32>(e, 0.0, 0.0)) - height_at(base - vec3<f32>(e, 0.0, 0.0));
    let hz = height_at(base + vec3<f32>(0.0, 0.0, e)) - height_at(base - vec3<f32>(0.0, 0.0, e));
    let n = normalize(vec3<f32>(-hx, 2.0 * e, -hz));

    var out: VertexOutput;
    out.world_pos = world;
    out.world_normal = n;
    out.height = h;
    out.position = globals.vp * vec4<f32>(world, 1.0);
    return out;
}

fn ramp(t: f32) -> vec3<f32> {
    let a = clamp(t, 0.0, 1.0);
    let water = vec3<f32>(0.09, 0.24, 0.42);
    let sand  = vec3<f32>(0.78, 0.71, 0.48);
    let grass = vec3<f32>(0.24, 0.52, 0.20);
    let rock  = vec3<f32>(0.42, 0.36, 0.29);
    let snow  = vec3<f32>(0.95, 0.96, 1.0);
    if (a < 0.28) { return mix(water, sand, smoothstep(0.18, 0.28, a)); }
    if (a < 0.5)  { return mix(sand, grass, smoothstep(0.28, 0.4, a)); }
    if (a < 0.72) { return mix(grass, rock, smoothstep(0.5, 0.72, a)); }
    return mix(rock, snow, smoothstep(0.72, 0.9, a));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let sun = normalize(-globals.sun_direction);
    let ndl = max(dot(N, sun), 0.0);
    let amp = material.params.x;
    let col = ramp((in.height / amp) * 0.26 + 0.34);
    let lit = col * (0.28 + 0.9 * ndl) * globals.sun_color * max(globals.sun_intensity, 0.6);
    let out = pow(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(out, 1.0);
}
"""


class GpuFieldMaterial(_Material):
    """Displaces + shades a mesh by a WGSL-transpiled `Field`, evaluated on the GPU."""

    binding_slot = 0

    def __init__(self, amplitude: float = 1.0, epsilon: float = 0.03):
        self.amplitude = amplitude
        self.epsilon = epsilon

    @property
    def pipeline_subtype(self):
        return "gpufield"

    @classmethod
    def _compile(cls) -> str:
        return _SHADER

    @classmethod
    def uniform_type(cls):
        return {"params": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        return np.tile(np.array([self.amplitude, self.epsilon, 0.0, 0.0], dtype=np.float32), (n, 1))


def main():
    engine = mx.Engine("GPU Field", width=1024, height=768)
    engine.background_color = (0.55, 0.68, 0.85)
    engine.set_sun(DirectionalLight(color="#fff2e0", intensity=2.4, direction=(-0.5, -0.7, -0.35)))

    # A flat plane — all the terrain shape comes from the field, live on the GPU.
    plane = GeoMesh.plane(width=40.0, depth=40.0, segments=256).to_geometry()
    engine.spawn(Mesh(plane), Material(GpuFieldMaterial(amplitude=1.0, epsilon=0.03)), Transform())

    engine.camera.set_pose((0.0, 9.0, 17.0), (0.0, 0.0, 0.0))

    clock = {"f": 0}

    @engine.system
    def spin(query, dt):
        clock["f"] += 1
        a = 0.3 + clock["f"] * 0.006
        r = 22.0
        engine.camera.set_pose((np.cos(a) * r, 9.0, np.sin(a) * r), (0.0, 0.0, 0.0))

    engine.cli()


if __name__ == "__main__":
    main()
