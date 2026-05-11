"""Dynamic text widget — renders a getter()'s return value each frame.

Caching discipline:
- `refresh()` is called once per frame by the gui render pass.
- If the new string equals the cached one, intrinsic_size is reused (no atlas churn).
- If the new string differs, intrinsic_size is recomputed and the widget marks
  itself layout-dirty (Plan 2 will hook layout-dirty propagation into the bridge;
  Plan 1 ignores it since layout currently recomputes every frame).

For values that fluctuate frame-to-frame, callers SHOULD pass `min_width` so
the intrinsic width doesn't oscillate.
"""

from __future__ import annotations

from typing import Any, Callable

from manifoldx.gui.widgets import Widget, _measure_text


class ValueDisplay(Widget):
    def __init__(
        self,
        getter: Callable[[], str],
        *,
        min_width: float | None = None,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        self._getter = getter
        self._min_width = min_width
        self._text: str = ""
        self._cached_intrinsic: tuple[float, float] | None = None

    @property
    def text(self) -> str:
        return self._text

    def refresh(self) -> None:
        """Call the getter, update cached text/intrinsic if changed."""
        new = self._getter()
        if not isinstance(new, str):
            new = str(new)
        if new != self._text:
            self._text = new
            self._cached_intrinsic = None

    def intrinsic_size(self) -> tuple[float, float]:
        if self._cached_intrinsic is None:
            font_size = int(self.effective_style().get("font_size", 12))
            w, h = _measure_text(self._text, font_size)
            if self._min_width is not None:
                w = max(w, float(self._min_width))
            self._cached_intrinsic = (w, h)
        return self._cached_intrinsic
