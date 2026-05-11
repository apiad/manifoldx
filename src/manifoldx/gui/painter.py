"""Painter accumulator for the GUI layer.

The painter is the seam between the widget tree (pure Python, layout-aware)
and the render pass (GPU-aware, batches into two instanced draws).

Painting walks the tree top-down, emitting:

- One `RectOp` per Panel (and, in Plan 2, per Button/Slider/Toggle).
- One `TextOp` per Text / ValueDisplay glyph batch.

Ops carry resolved pixel boxes and resolved colors — the painter does no
style lookup itself; widgets pass already-resolved styles through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.style import parse_color
from manifoldx.gui.value_display import ValueDisplay
from manifoldx.gui.widgets import Panel, Text, Widget


@dataclass(frozen=True, slots=True)
class RectOp:
    box: LayoutBox
    fill: tuple[float, float, float, float]
    border_color: tuple[float, float, float, float]
    border: float
    radius: float


@dataclass(frozen=True, slots=True)
class TextOp:
    box: LayoutBox
    text: str
    font_size: int
    fg: tuple[float, float, float, float]


@dataclass
class Painter:
    rect_ops: list[RectOp] = field(default_factory=list)
    text_ops: list[TextOp] = field(default_factory=list)

    def draw_rect(
        self,
        *,
        box: LayoutBox,
        fill: tuple[float, float, float, float],
        border_color: tuple[float, float, float, float] = (0, 0, 0, 0),
        border: float = 0.0,
        radius: float = 0.0,
    ) -> None:
        self.rect_ops.append(
            RectOp(
                box=box,
                fill=fill,
                border_color=border_color,
                border=border,
                radius=radius,
            )
        )

    def draw_text(
        self,
        *,
        box: LayoutBox,
        text: str,
        font_size: int,
        fg: tuple[float, float, float, float],
    ) -> None:
        self.text_ops.append(TextOp(box=box, text=text, font_size=font_size, fg=fg))

    def clear(self) -> None:
        self.rect_ops.clear()
        self.text_ops.clear()


def paint(
    widget: Widget,
    spec: dict[str, Any],
    boxes: dict[int, LayoutBox],
    painter: Painter,
) -> None:
    """Walk the widget tree top-down, emitting ops into `painter`.

    `spec` is the layout spec for `widget` (as produced by Panel.build_layout_spec
    or the equivalent for a leaf); `boxes` is the result of compute_layout.
    """
    box = boxes[id(spec)]
    if isinstance(widget, Panel):
        s = widget.effective_style()
        painter.draw_rect(
            box=box,
            fill=parse_color(s["bg"]),
            border_color=parse_color(s["border_color"]),
            border=float(s["border"]),
            radius=float(s["radius"]),
        )
        for child, child_spec in zip(widget.children, spec["children"]):
            paint(child, child_spec, boxes, painter)
    elif isinstance(widget, Text):
        s = widget.effective_style()
        painter.draw_text(
            box=box,
            text=widget.text,
            font_size=int(s["font_size"]),
            fg=parse_color(s["fg"]),
        )
    elif isinstance(widget, ValueDisplay):
        s = widget.effective_style()
        painter.draw_text(
            box=box,
            text=widget.text,
            font_size=int(s["font_size"]),
            fg=parse_color(s["fg"]),
        )
    # Plan 2 will add Button / Slider / Toggle branches.
