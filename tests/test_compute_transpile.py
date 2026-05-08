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
