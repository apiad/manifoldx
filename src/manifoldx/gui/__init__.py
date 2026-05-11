"""ManifoldX in-engine GUI layer.

Public API (Plan 1):
- `Panel`, `Text`, `ValueDisplay` — non-interactive widgets.
- `style` — theme + named classes + per-widget overrides.

Plan 2 will add `Button`, `Slider`, `Toggle`.
"""

from manifoldx.gui import style  # noqa: F401
from manifoldx.gui.value_display import ValueDisplay  # noqa: F401
from manifoldx.gui.widgets import Panel, Text  # noqa: F401

__all__ = ["Panel", "Text", "ValueDisplay", "style"]
