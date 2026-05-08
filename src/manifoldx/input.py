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


class InputState:
    """Per-engine input polling state. Read-only public surface; only the
    bridge mutates via `_record_*` and `_begin_frame`.

    Three flavors of state:

    - "Live" (e.g. `pressed_keys`, `mouse_pos`): mutated immediately on
      every bridge callback. Reads at any point in the frame see the
      freshest value.
    - "Frame-bound" (`just_pressed_keys`, `just_released_keys`,
      `just_pressed_buttons`, `just_released_buttons`): visible only
      between calls to `_begin_frame`. The pending buffer accumulates
      between swaps; `_begin_frame` moves pending → current and clears
      pending.
    - "Accumulators" (`mouse_delta`, `wheel_delta`): summed across all
      bridge callbacks since the last `_begin_frame`; finalized into the
      public `_mouse_delta` / `_wheel_delta` snapshots on swap.
    """

    def __init__(self) -> None:
        # Keyboard
        self._pressed_keys: set[str] = set()
        self._modifiers: tuple[str, ...] = ()
        self._pending_just_pressed_keys: set[str] = set()
        self._pending_just_released_keys: set[str] = set()
        self._just_pressed_keys: frozenset[str] = frozenset()
        self._just_released_keys: frozenset[str] = frozenset()

    # ---- Keyboard public API ----

    def is_pressed(self, key: str) -> bool:
        return key in self._pressed_keys

    def just_pressed(self, key: str) -> bool:
        return key in self._just_pressed_keys

    def just_released(self, key: str) -> bool:
        return key in self._just_released_keys

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._pressed_keys)

    @property
    def modifiers(self) -> tuple[str, ...]:
        return self._modifiers

    # ---- Bridge-only mutators ----

    def _record_key_down(self, key: str, modifiers: tuple[str, ...]) -> None:
        if key not in self._pressed_keys:
            self._pending_just_pressed_keys.add(key)
        self._pressed_keys.add(key)
        self._modifiers = modifiers

    def _record_key_up(self, key: str, modifiers: tuple[str, ...]) -> None:
        if key in self._pressed_keys:
            self._pending_just_released_keys.add(key)
        self._pressed_keys.discard(key)
        self._modifiers = modifiers

    def _begin_frame(self) -> None:
        # Swap keyboard pending → current.
        self._just_pressed_keys = frozenset(self._pending_just_pressed_keys)
        self._just_released_keys = frozenset(self._pending_just_released_keys)
        self._pending_just_pressed_keys.clear()
        self._pending_just_released_keys.clear()
