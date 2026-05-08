"""Sci-viz ECS components: PointCloud (marker), ScalarValue, Radius, TextLabel.

All four inherit from `manifoldx.components.Component`, which means dtype +
shape are derived from the class-level annotations and the Engine
auto-registers them with the store on first spawn — no need for manual
`engine.store.register_component(...)` boilerplate in user code.
"""

from manifoldx.components import Component
from manifoldx.types import Float


class ScalarValue(Component):
    """Per-entity scalar attribute, mapped through ColormapMaterial's LUT.

    Storage layout: 1 float per entity (column 0).

    Usage:
        ScalarValue()                      # default 0.0 for all
        ScalarValue(value=1.5)             # broadcast scalar
        ScalarValue(value=array_shape_N)   # explicit per-entity
    """

    value: Float = 0.0


class Radius(Component):
    """Per-entity world-space radius for sprite scaling.

    Storage layout: 1 float per entity (column 0). Default: 1.0.

    Usage:
        Radius()                       # all 1.0
        Radius(radius=0.05)            # broadcast scalar
        Radius(radius=array_shape_N)   # explicit per-entity
    """

    radius: Float = 1.0


class PointCloud(Component):
    """Marker component — tags entities for the sprite render path.

    Carries no per-entity data. The renderer detects this component
    and substitutes the SPRITE_QUAD geometry, ignoring any Mesh component.
    """


class TextLabel(Component):
    """Per-entity atlas slice index for a rasterized label.

    Storage layout: 1 float per entity (column 0). The float holds an integer
    slice index in [0, MAX_LABELS); the shader casts it to u32. Float storage
    keeps `TextLabel` symmetric with `ScalarValue` and `Radius`, all of which
    flow through the same _FieldView path.

    Usage:
        TextLabel()                              # all 0
        TextLabel(index=7)                       # broadcast scalar
        TextLabel(index=array_shape_N_int)       # explicit per-entity
    """

    index: Float = 0.0


class AxisFrame(Component):
    """Tag entity for axis line rendering with `AxisMaterial`.

    Storage layout: 2 floats per entity (extent, thickness).

    `extent` is the half-length of the axis line in world units. The line
    geometry is a unit line; the entity's `Transform` provides position +
    rotation + scale (axis direction is encoded by which AXIS_* geometry
    the entity carries).

    `thickness` is reserved in v1 — the line render path uses native wgpu
    LineList topology, which on Vulkan/Metal/D3D12 always renders 1px wide.
    Kept in the schema for future quad-extrusion support.
    """

    extent: Float = 1.0
    thickness: Float = 1.0


class Volume(Component):
    """Per-entity reference to a registered 3D scalar field.

    The voxel data itself lives in `Engine._volume_registry`, keyed by
    the integer handle returned from `engine.register_volume(array)`.
    Mirrors the resource-pointer pattern used by `Mesh.geometry_id` and
    `Material.material_id`. Stored as Float (matching `TextLabel.index`)
    so the field flows through the existing _FieldView path; the renderer
    casts to u32 in the WGSL shader.
    """

    volume_id: Float = 0.0
