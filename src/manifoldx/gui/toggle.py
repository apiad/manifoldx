"""Toggle widget — click flips boolean, emits `gui:toggle:<name>:change`."""

from __future__ import annotations

from typing import Any

from manifoldx.gui.widgets import Widget, _measure_text


class Toggle(Widget):
    _is_gui_interactive = True
    _gui_captures_pointer = False

    def __init__(
        self,
        *,
        name: str,
        value: bool,
        label: str,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if not isinstance(name, str) or not name:
            raise ValueError("Toggle.name is required and must be a non-empty string")
        self.name = name
        self.value = bool(value)
        self.label = label

    def intrinsic_size(self) -> tuple[float, float]:
        font_size = int(self.effective_style().get("font_size", 12))
        lw, lh = _measure_text(self.label, font_size)
        return (14.0 + 6.0 + lw, max(14.0, lh))

    def _on_pointer_down(self, ev: Any, engine: Any) -> None:
        self.value = not self.value
        engine.emit(f"gui:toggle:{self.name}:change", {"value": self.value})
