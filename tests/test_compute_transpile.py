"""Unit tests for the Phase-2 Python→WGSL transpiler."""
import pytest


def test_component_layout_table_derived_from_field_specs():
    """Component subclasses get a `_layout` dict mapping field → (offset, length)."""
    from manifoldx.components import Component
    from manifoldx.types import Float, Vector3

    class MyVel(Component):
        vector: Vector3
        spin: Float

    assert MyVel._layout == {"vector": (0, 3), "spin": (3, 1)}


def test_component_layout_handles_marker_only_components():
    """A marker component (no fields) gets an empty `_layout`."""
    from manifoldx.components import Component

    class Marker(Component):
        pass

    assert Marker._layout == {}


def test_builtin_transform_layout():
    """Transform is pre-`Component` base — explicit `_layout` matches its 10-float layout."""
    from manifoldx.components import Transform

    assert Transform._layout == {"pos": (0, 3), "rot": (3, 4), "scale": (7, 3)}


def test_builtin_mesh_and_material_layouts():
    """Mesh and Material are scalar u32 references — `_layout` reflects that."""
    from manifoldx.components import Material, Mesh

    assert Mesh._layout == {"geometry_id": (0, 1)}
    assert Material._layout == {"material_id": (0, 1)}


def test_shader_type_tags_usable_as_annotations():
    """vec3 and vec4 work as PEP-526 annotation tags inside kernel bodies."""
    from manifoldx.compute.shader import vec3, vec4

    # The transpiler will look up each annotation as a name reference; the
    # tag's identity is what matters.
    def example(x: vec3, y: vec4) -> vec3:
        return x

    assert example.__annotations__ == {"x": vec3, "y": vec4, "return": vec3}


def test_shader_type_tags_still_raise_when_called_at_runtime():
    """The tags remain shader-only — calling outside a kernel still raises."""
    from manifoldx.compute.shader import vec3, vec4

    with pytest.raises(NotImplementedError, match="shader primitive"):
        vec3(1.0, 2.0, 3.0)
    with pytest.raises(NotImplementedError, match="shader primitive"):
        vec4(1.0, 2.0, 3.0, 4.0)


def test_compute_shader_compile_error_formats_with_source_line():
    """ComputeShaderCompileError carries file/line/col/category and renders cleanly."""
    from manifoldx.compute.transpile import ComputeShaderCompileError

    err = ComputeShaderCompileError(
        category="missing-annotation",
        message="local 'x' used without annotation",
        filename="<kernel>",
        line=12,
        col=8,
        source_line="    x = 1.0",
    )
    text = str(err)
    assert "<kernel>:12:8" in text
    assert "missing-annotation" in text
    assert "local 'x' used without annotation" in text
    assert "    x = 1.0" in text
    # Caret line indicates the column.
    assert "^" in text
    assert err.category == "missing-annotation"


def test_compute_shader_compile_error_without_source_line():
    """ComputeShaderCompileError tolerates missing source_line (e.g. wgpu-validation)."""
    from manifoldx.compute.transpile import ComputeShaderCompileError

    err = ComputeShaderCompileError(
        category="wgpu-validation",
        message="invalid storage binding",
        filename="kernel.py",
        line=0,
        col=0,
        source_line=None,
    )
    text = str(err)
    assert "kernel.py" in text
    assert "wgpu-validation" in text
    assert "invalid storage binding" in text
