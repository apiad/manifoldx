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
