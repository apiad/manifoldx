"""ManifoldX in-engine GUI layer.

Public API:

- `Panel`, `Text`, `ValueDisplay` — non-interactive widgets (Plan 1).
- `Button`, `Slider`, `Toggle` — interactive widgets (Plan 2, not yet exposed).
- `style` — theme + named classes + per-widget overrides.
"""

from manifoldx.gui import style  # noqa: F401
from manifoldx.gui.widgets import Panel, Text  # noqa: F401

__all__ = ["Panel", "Text", "style"]
