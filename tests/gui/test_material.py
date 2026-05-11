"""Tests for manifoldx.gui.material — RectMaterial WGSL + uniform layout.

No GPU device is required: we introspect _compile() and uniform_type()
directly. End-to-end pipeline-cache differentiation is exercised in
tests/gui/test_render_gpu.py (Task 6).
"""

from manifoldx.gui.material import RectMaterial


def test_rect_material_compile_returns_wgsl_with_sdf_helper():
    src = RectMaterial._compile()
    assert isinstance(src, str)
    # The signed-distance rounded-rect expression is the load-bearing geom.
    # We check for the canonical SDF helper name we'll define.
    assert "rounded_rect_sdf" in src
    # And for the per-instance struct.
    assert "RectInstance" in src
    # Vertex + fragment entry points.
    assert "@vertex" in src
    assert "@fragment" in src


def test_rect_material_uniform_type_describes_globals_and_instances():
    types = RectMaterial.uniform_type()
    # Per-instance fields, in the order they're packed into a row-of-floats.
    expected = {
        "xy": "vec2<f32>",
        "size": "vec2<f32>",
        "radius": "f32",
        "border": "f32",
        "bg": "vec4<f32>",
        "border_color": "vec4<f32>",
    }
    for k, v in expected.items():
        assert types.get(k) == v, f"missing/mismatched uniform field {k}"


def test_rect_material_pipeline_subtype_is_gui():
    # The design says: pipeline-cache 5th element is "gui" so this pass
    # doesn't share pipelines with sci-viz label.
    assert RectMaterial().pipeline_subtype == "gui"


def test_rect_material_pack_instances_layout():
    import numpy as np
    from manifoldx.gui.layout import LayoutBox
    from manifoldx.gui.painter import RectOp

    ops = [
        RectOp(
            box=LayoutBox(10, 20, 100, 50),
            fill=(0.5, 0.6, 0.7, 0.8),
            border_color=(0.1, 0.2, 0.3, 0.4),
            border=2.0,
            radius=4.0,
        )
    ]
    arr = RectMaterial.pack_instances(ops)
    assert arr.dtype == np.float32
    assert arr.shape == (1, 14)
    np.testing.assert_allclose(
        arr[0],
        [10, 20, 100, 50, 4.0, 2.0, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4],
    )


def test_rect_material_pack_empty_returns_zero_rows():
    import numpy as np
    arr = RectMaterial.pack_instances([])
    assert arr.shape == (0, 14)
    assert arr.dtype == np.float32
