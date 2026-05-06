"""End-to-end integration test for Compute systems.

A trivial Compute that doubles the X/Y/Z components of every Transform —
proves the full pipeline (registration → compile → bind group → dispatch
→ readback) works correctly. Skips when no offscreen wgpu backend is
available (CI hosts without GPU).
"""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.compute import Compute, ReadsWrites
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import BasicMaterial, sphere


# WGSL kernel that multiplies the first three floats of each Transform
# (= position xyz) by 2. Transform is laid out as 10 floats per entity:
# pos.xyz (3) + rot.xyzw (4) + scale.xyz (3).
_DOUBLE_X_WGSL = """
@group(0) @binding(0) var<storage, read_write> transforms: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let stride = 10u;
    let base = i * stride;
    transforms[base + 0u] *= 2.0;
    transforms[base + 1u] *= 2.0;
    transforms[base + 2u] *= 2.0;
}
"""


class DoublePositions(Compute):
    """Doubles xyz of every Transform entity."""

    transforms: ReadsWrites[Transform]

    workgroup_size = 64
    dispatch = "entity_count"

    def compile(self) -> str:
        return _DOUBLE_X_WGSL


def _make_offscreen_engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("test", width=64, height=64)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_compute_doubles_transform_positions_after_one_frame():
    engine = _make_offscreen_engine()

    # Spawn 4 entities at varied positions.
    n = 4
    positions = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]],
        dtype=np.float32,
    )
    engine.spawn(
        Mesh(sphere(0.1, 8)),
        Material(BasicMaterial("#ffffff")),
        Transform(pos=positions),
        n=n,
    )
    engine.compute(DoublePositions)

    engine._draw_frame()

    transforms = engine.store._components["Transform"]
    np.testing.assert_allclose(transforms[:n, 0], positions[:, 0] * 2.0, rtol=1e-5)
    np.testing.assert_allclose(transforms[:n, 1], positions[:, 1] * 2.0, rtol=1e-5)
    np.testing.assert_allclose(transforms[:n, 2], positions[:, 2] * 2.0, rtol=1e-5)


def test_compute_runs_every_frame():
    """Multiple frames stack the doubling: 1 → 2 → 4 → 8."""
    engine = _make_offscreen_engine()
    engine.spawn(
        Mesh(sphere(0.1, 8)),
        Material(BasicMaterial("#ffffff")),
        Transform(pos=(1.0, 1.0, 1.0)),
        n=1,
    )
    engine.compute(DoublePositions)

    expected = 1.0
    for _ in range(3):
        engine._draw_frame()
        expected *= 2.0

    transforms = engine.store._components["Transform"]
    np.testing.assert_allclose(transforms[0, :3], [expected, expected, expected], rtol=1e-5)


def test_compute_with_uniform_constant():
    """A Compute using a Uniform[float] constant works correctly."""
    from manifoldx.compute import Uniform

    class ScalePositions(Compute):
        transforms: ReadsWrites[Transform]
        factor: Uniform[float] = 3.0
        workgroup_size = 64
        dispatch = "entity_count"

        def compile(self) -> str:
            return """
            struct Uniforms { factor: f32, };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read_write> transforms: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                let base = i * 10u;
                transforms[base + 0u] *= uniforms.factor;
                transforms[base + 1u] *= uniforms.factor;
                transforms[base + 2u] *= uniforms.factor;
            }
            """

    engine = _make_offscreen_engine()
    engine.spawn(
        Mesh(sphere(0.1, 8)),
        Material(BasicMaterial("#ffffff")),
        Transform(pos=(2.0, 2.0, 2.0)),
        n=1,
    )
    engine.compute(ScalePositions)
    engine._draw_frame()

    transforms = engine.store._components["Transform"]
    np.testing.assert_allclose(transforms[0, :3], [6.0, 6.0, 6.0], rtol=1e-5)
