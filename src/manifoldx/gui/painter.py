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

from manifoldx.gui.button import Button
from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.slider import Slider
from manifoldx.gui.style import parse_color
from manifoldx.gui.toggle import Toggle
from manifoldx.gui.value_display import ValueDisplay
from manifoldx.gui.widgets import Panel, Text, Widget, _measure_text


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
    elif isinstance(widget, Button):
        s = widget.effective_style()
        painter.draw_rect(
            box=box,
            fill=parse_color(s["bg"]),
            border_color=parse_color(s["border_color"]),
            border=float(s["border"]),
            radius=float(s["radius"]),
        )
        # Center the label inside the button box.
        font_size = int(s["font_size"])
        lw, lh = _measure_text(widget.label, font_size)
        text_box = LayoutBox(
            box.x + (box.w - lw) * 0.5,
            box.y + (box.h - lh) * 0.5,
            lw,
            lh,
        )
        painter.draw_text(
            box=text_box,
            text=widget.label,
            font_size=font_size,
            fg=parse_color(s["fg"]),
        )
    elif isinstance(widget, Toggle):
        s = widget.effective_style()
        cb_size = 14.0
        cb_box = LayoutBox(box.x, box.y + (box.h - cb_size) * 0.5, cb_size, cb_size)
        painter.draw_rect(
            box=cb_box,
            fill=parse_color(s["bg"]),
            border_color=parse_color(s["fg"]),
            border=1.0,
            radius=2.0,
        )
        if widget.value:
            inset = 3.0
            inner = LayoutBox(
                cb_box.x + inset, cb_box.y + inset,
                cb_box.w - 2 * inset, cb_box.h - 2 * inset,
            )
            painter.draw_rect(
                box=inner,
                fill=parse_color(s["fg"]),
                radius=1.0,
            )
        font_size = int(s["font_size"])
        _lw, lh = _measure_text(widget.label, font_size)
        text_box = LayoutBox(
            box.x + cb_size + 6.0,
            box.y + (box.h - lh) * 0.5,
            box.w - cb_size - 6.0,
            lh,
        )
        painter.draw_text(
            box=text_box, text=widget.label,
            font_size=font_size, fg=parse_color(s["fg"]),
        )
    elif isinstance(widget, Slider):
        s = widget.effective_style()
        painter.draw_rect(
            box=box,
            fill=parse_color(s["bg"]),
            border_color=parse_color(s["border_color"]),
            border=float(s["border"]),
            radius=float(s["radius"]),
        )
        if widget.max > widget.min:
            t = (widget.value - widget.min) / (widget.max - widget.min)
        else:
            t = 0.0
        t = max(0.0, min(1.0, t))
        if t > 0:
            fill_box = LayoutBox(box.x, box.y, box.w * t, box.h)
            painter.draw_rect(
                box=fill_box,
                fill=parse_color(s["fg"]),
                radius=float(s["radius"]),
            )
