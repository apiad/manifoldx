"""manifoldx.viz — Scientific visualization primitives.

Public surface for Plan 1:
    PointCloud, ScalarValue, Radius, ColormapMaterial

Future plans add:
    TextLabel, AxisFrame, ScaleBar, LabelMaterial, AxisMaterial,
    point_cloud(), axes(), scale_bar(), colormap_legend()
"""

from manifoldx.viz.components import PointCloud, Radius, ScalarValue
from manifoldx.viz.materials import ColormapMaterial

__all__ = ["PointCloud", "ScalarValue", "Radius", "ColormapMaterial"]
