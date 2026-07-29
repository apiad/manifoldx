import numpy as np
import pytest
import wgpu

from manifoldx.modeling import fields, field_to_wgsl, FieldNotTranspilable


def _compile(src: str):
    """Create a WGSL shader module (raises on invalid WGSL)."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    device.create_shader_module(code=src)


def test_ast_is_recorded_and_cpu_path_unchanged():
    f = fields.fbm(seed=8, freq=0.8) * 0.5 + 0.2
    assert f._ast is not None
    # CPU evaluation still works and is finite.
    out = f(np.random.default_rng(0).random((16, 3)).astype(np.float32))
    assert out.shape == (16,) and np.all(np.isfinite(out))


def test_transpile_composed_field_compiles():
    # A field like the planet terrain: ridged + fbm, domain-warped, remapped.
    terrain = (
        fields.ridged(seed=4, freq=0.24, octaves=5) * 0.72
        + fields.fbm(seed=8, freq=0.8, octaves=5) * 0.20
    ).warp(0.7, fx=fields.fbm(2, 0.38), fz=fields.fbm(9, 0.38)).remap(0.0, 1.0, -0.6, 0.8)
    src = field_to_wgsl(terrain, name="terrain_field")
    assert "fn terrain_field(P: vec3<f32>) -> f32" in src
    assert "_foctaves" in src and "_fvnoise" in src   # noise prelude included
    _compile(src)                                       # must be valid WGSL


def test_transpile_covers_combinators():
    f = (fields.coord("y").abs().clamp(0.0, 1.0)
         .mix(fields.distance(), 0.5)
         .maximum(fields.constant(0.1)))
    src = field_to_wgsl(f)
    _compile(src)


def test_hand_written_field_is_not_transpilable():
    from manifoldx.modeling.fields import Field
    raw = Field(lambda p: p[:, 0])          # no AST
    with pytest.raises(FieldNotTranspilable):
        field_to_wgsl(raw)


def test_worley_source_is_not_transpilable():
    with pytest.raises(FieldNotTranspilable):
        field_to_wgsl(fields.worley(seed=1, freq=1.0))
