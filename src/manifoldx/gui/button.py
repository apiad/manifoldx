"""Button widget — emits `gui:button:<name>:click` on pointer_down."""

from __future__ import annotations

from typing import Any

from manifoldx.gui.widgets import Widget, _measure_text


class Button(Widget):
    _is_gui_interactive = True
    _gui_captures_pointer = False

    def __init__(
        self,
        *,
        name: str,
        label: str,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if not isinstance(name, str) or not name:
            raise ValueError("Button.name is required and must be a non-empty string")
        self.name = name
        self.label = label
        self._click_latch: bool = False

    def intrinsic_size(self) -> tuple[float, float]:
        font_size = int(self.effective_style().get("font_size", 12))
        w, h = _measure_text(self.label, font_size)
        # Add 12px horizontal padding around the label, 4px vertical.
        return (w + 24.0, h + 8.0)

    def was_clicked(self) -> bool:
        """Return True if the button has been clicked since the previous read.
        Latch consumed on read."""
        v = self._click_latch
        self._click_latch = False
        return v

    def _on_pointer_down(self, ev: Any, engine: Any) -> None:
        self._click_latch = True
        engine.emit(f"gui:button:{self.name}:click", {})
