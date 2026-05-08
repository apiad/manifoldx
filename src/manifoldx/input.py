"""Input layer: event dataclasses, polling state, rendercanvas bridge.

See .knowledge/analysis/2026-05-08-input-events-design.md for the design.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KeyEvent:
    """A keyboard event delivered on `key_down` or `key_up`."""

    key: str
    modifiers: tuple[str, ...]
    is_down: bool


@dataclass(frozen=True, slots=True)
class PointerEvent:
    """A pointer event delivered on `pointer_down` / `pointer_up` / `pointer_move`.

    `dx` / `dy` is the delta from the previous pointer position the bridge
    saw. On `pointer_down` and `pointer_up` it is always 0.0 and the bridge
    resets its "last position" tracker so a fresh drag starts clean.
    """

    x: float
    y: float
    dx: float
    dy: float
    button: int
    buttons: tuple[int, ...]
    modifiers: tuple[str, ...]
    phase: str  # "down" | "up" | "move"


@dataclass(frozen=True, slots=True)
class WheelEvent:
    """A scroll-wheel event. `dy` is rendercanvas-scaled (~100 per notch on glfw)."""

    dx: float
    dy: float
    x: float
    y: float
    buttons: tuple[int, ...]
    modifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResizeEvent:
    """A canvas-resize event. Sizes are in logical pixels."""

    width: int
    height: int
    pixel_ratio: float
