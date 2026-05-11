"""ManifoldX in-engine GUI layer.

Public API:
- `Panel`, `Text`, `ValueDisplay`, `Button`, `Toggle` — widgets.
- `style` — theme + named classes + per-widget overrides.

Plan 2 (in progress) will add `Slider`.
"""

from manifoldx.gui import style  # noqa: F401
from manifoldx.gui.button import Button  # noqa: F401
from manifoldx.gui.toggle import Toggle  # noqa: F401
from manifoldx.gui.value_display import ValueDisplay  # noqa: F401
from manifoldx.gui.widgets import Panel, Text  # noqa: F401

__all__ = ["Button", "Panel", "Text", "Toggle", "ValueDisplay", "style"]
