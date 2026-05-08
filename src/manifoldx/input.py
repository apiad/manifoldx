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

        # Mouse
        self._mouse_pos: tuple[float, float] = (0.0, 0.0)
        self._accum_mouse_dx: float = 0.0
        self._accum_mouse_dy: float = 0.0
        self._mouse_delta: tuple[float, float] = (0.0, 0.0)
        self._accum_wheel_dx: float = 0.0
        self._accum_wheel_dy: float = 0.0
        self._wheel_delta: tuple[float, float] = (0.0, 0.0)
        self._pressed_buttons: set[int] = set()
        self._pending_just_pressed_buttons: set[int] = set()
        self._pending_just_released_buttons: set[int] = set()
        self._just_pressed_buttons: frozenset[int] = frozenset()
        self._just_released_buttons: frozenset[int] = frozenset()

        # Window
        self._viewport_size: tuple[int, int] = (0, 0)

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

    # ---- Mouse public API ----

    @property
    def mouse_pos(self) -> tuple[float, float]:
        return self._mouse_pos

    @property
    def mouse_delta(self) -> tuple[float, float]:
        return self._mouse_delta

    @property
    def wheel_delta(self) -> tuple[float, float]:
        return self._wheel_delta

    def is_mouse_pressed(self, button: int) -> bool:
        return button in self._pressed_buttons

    def just_mouse_pressed(self, button: int) -> bool:
        return button in self._just_pressed_buttons

    def just_mouse_released(self, button: int) -> bool:
        return button in self._just_released_buttons

    @property
    def pressed_buttons(self) -> tuple[int, ...]:
        return tuple(sorted(self._pressed_buttons))

    # ---- Window public API ----

    @property
    def viewport_size(self) -> tuple[int, int]:
        return self._viewport_size

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

    def _record_pointer_down(self, ev: PointerEvent) -> None:
        if ev.button not in self._pressed_buttons:
            self._pending_just_pressed_buttons.add(ev.button)
        self._pressed_buttons.add(ev.button)
        self._mouse_pos = (ev.x, ev.y)
        self._modifiers = ev.modifiers

    def _record_pointer_up(self, ev: PointerEvent) -> None:
        if ev.button in self._pressed_buttons:
            self._pending_just_released_buttons.add(ev.button)
        self._pressed_buttons.discard(ev.button)
        self._mouse_pos = (ev.x, ev.y)
        self._modifiers = ev.modifiers

    def _record_pointer_move(self, ev: PointerEvent) -> None:
        self._accum_mouse_dx += ev.dx
        self._accum_mouse_dy += ev.dy
        self._mouse_pos = (ev.x, ev.y)
        self._modifiers = ev.modifiers

    def _record_wheel(self, ev: WheelEvent) -> None:
        self._accum_wheel_dx += ev.dx
        self._accum_wheel_dy += ev.dy
        self._modifiers = ev.modifiers

    def _record_resize(self, ev: ResizeEvent) -> None:
        self._viewport_size = (ev.width, ev.height)

    def _begin_frame(self) -> None:
        # Swap keyboard pending → current.
        self._just_pressed_keys = frozenset(self._pending_just_pressed_keys)
        self._just_released_keys = frozenset(self._pending_just_released_keys)
        self._pending_just_pressed_keys.clear()
        self._pending_just_released_keys.clear()

        # Swap mouse-button pending → current.
        self._just_pressed_buttons = frozenset(self._pending_just_pressed_buttons)
        self._just_released_buttons = frozenset(self._pending_just_released_buttons)
        self._pending_just_pressed_buttons.clear()
        self._pending_just_released_buttons.clear()

        # Finalize delta accumulators.
        self._mouse_delta = (self._accum_mouse_dx, self._accum_mouse_dy)
        self._wheel_delta = (self._accum_wheel_dx, self._accum_wheel_dy)
        self._accum_mouse_dx = 0.0
        self._accum_mouse_dy = 0.0
        self._accum_wheel_dx = 0.0
        self._accum_wheel_dy = 0.0
