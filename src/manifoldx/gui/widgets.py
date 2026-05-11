"""Widget classes for the GUI layer.

This module defines the non-interactive widgets — interactive ones
(Button/Slider/Toggle) land in Plan 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from manifoldx.gui import style
from manifoldx.gui.style import parse_padding

# Anchors are corners on the viewport; offset is added in pixels.
_VALID_ANCHORS = frozenset({
    "top-left", "top-right", "top-center",
    "bottom-left", "bottom-right", "bottom-center",
    "center-left", "center-right", "center",
})


class Widget(ABC):
    """Base class for all GUI widgets.

    A widget owns its style references (`style` is a class name, `style_overrides`
    is a per-instance dict) and exposes the surface the layout, painter, and
    bridge need.
    """

    def __init__(
        self,
        *,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.style = style
        self.style_overrides = dict(style_overrides) if style_overrides else {}

    def effective_style(self) -> dict[str, Any]:
        return _style_resolve(self.style, self.style_overrides)

    @abstractmethod
    def intrinsic_size(self) -> tuple[float, float]:
        """(width, height) in pixels — used when no explicit size or flex."""


def _style_resolve(class_name: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    # Indirection so tests can monkeypatch resolution if needed.
    return style.resolve(class_name, overrides)


class Panel(Widget):
    """A container of widgets. Lays out children in a stack (vertical or
    horizontal) per its style `direction`, with `padding` around and `gap`
    between children.

    `anchor` and `offset` apply only to root panels (those added directly to
    `engine.gui`). Nested panels are positioned by the parent layout and
    ignore these fields.
    """

    def __init__(
        self,
        children: list[Widget] | None = None,
        *,
        anchor: str = "top-left",
        offset: tuple[int, int] = (0, 0),
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if anchor not in _VALID_ANCHORS:
            raise ValueError(
                f"invalid anchor {anchor!r}; expected one of {sorted(_VALID_ANCHORS)}"
            )
        self.children: list[Widget] = list(children or [])
        self.anchor = anchor
        self.offset = tuple(offset)

    def intrinsic_size(self) -> tuple[float, float]:
        # Containers without explicit size grow to fit their viewport slot;
        # the intrinsic value is a fallback for nested-without-flex cases.
        return (0.0, 0.0)

    def build_layout_spec(self) -> dict[str, Any]:
        s = self.effective_style()
        return {
            "direction": s["direction"],
            "padding": parse_padding(s["padding"]),
            "gap": int(s["gap"]),
            "width": s.get("width"),
            "height": s.get("height"),
            "flex": s.get("flex"),
            "intrinsic": self.intrinsic_size(),
            "children": [_child_spec(c) for c in self.children],
        }


def _child_spec(widget: Widget) -> dict[str, Any]:
    """Convert a widget into a layout spec dict. Defers to Panel for nested
    containers; leaves use the widget's intrinsic_size + style."""
    if isinstance(widget, Panel):
        return widget.build_layout_spec()
    s = widget.effective_style()
    return {
        "direction": "v",  # ignored for leaves
        "padding": (0, 0, 0, 0),
        "gap": 0,
        "width": s.get("width"),
        "height": s.get("height"),
        "flex": s.get("flex"),
        "intrinsic": widget.intrinsic_size(),
        "children": [],
    }


class Text(Widget):
    """A static text label. Intrinsic size is derived from rasterized glyph
    extents at the effective font size; cached on construction (re-measured
    if you mutate `.text`)."""

    def __init__(
        self,
        text: str,
        *,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        self._text = text
        self._cached_intrinsic: tuple[float, float] | None = None

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if value != self._text:
            self._cached_intrinsic = None
        self._text = value

    def intrinsic_size(self) -> tuple[float, float]:
        if self._cached_intrinsic is None:
            font_size = int(self.effective_style().get("font_size", 12))
            self._cached_intrinsic = _measure_text(self._text, font_size)
        return self._cached_intrinsic


def _measure_text(text: str, font_size: int) -> tuple[float, float]:
    """Approximate the rasterized extents without going through wgpu.

    We deliberately use a coarse linear model here (no PIL dependency in
    Plan 1) — the actual atlas-side measurement lands in Task 7, and any
    discrepancy at that point triggers a re-layout via the dirty bit.
    """
    char_w = font_size * 0.55
    return (char_w * max(1, len(text)), float(font_size) * 1.25)


class _GuiRoot:
    """Container exposed as `engine.gui`. List-like over root panels, plus
    a `pointer_over_gui` flag the bridge will toggle (Plan 2)."""

    def __init__(self) -> None:
        self._panels: list[Panel] = []
        self.pointer_over_gui: bool = False

    def append(self, panel: Panel) -> None:
        self._panels.append(panel)

    def remove(self, panel: Panel) -> None:
        self._panels.remove(panel)

    def clear(self) -> None:
        self._panels.clear()

    def __iter__(self):
        return iter(self._panels)

    def __len__(self) -> int:
        return len(self._panels)

    def __getitem__(self, i: int) -> Panel:
        return self._panels[i]
