"""Input layer: event dataclasses, polling state, rendercanvas bridge.

See .knowledge/analysis/2026-05-08-input-events-design.md for the design.
"""

from __future__ import annotations

import os
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
    saw on a `pointer_move`; on `pointer_down` and `pointer_up` it is always
    0.0. The bridge re-anchors its "last position" tracker to `(x, y)` on
    every pointer event, so the first `pointer_move` after a click produces
    a delta measured from the click point — not from wherever the cursor
    last reported during a previous drag.
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


# Gamepad button names, in GLFW's (SDL-mapped) button order.
GAMEPAD_BUTTONS: tuple[str, ...] = (
    "a", "b", "x", "y", "lb", "rb", "back", "start", "guide",
    "lstick", "rstick", "dpad_up", "dpad_right", "dpad_down", "dpad_left",
)
# Gamepad axis names, in GLFW's axis order. Sticks are in [-1, 1] with
# y positive = down (as GLFW reports it); triggers are in [0, 1].
GAMEPAD_AXES: tuple[str, ...] = ("left_x", "left_y", "right_x", "right_y", "lt", "rt")
# Virtual buttons: true when the d-pad OR the left stick points that way.
GAMEPAD_DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")


@dataclass(frozen=True, slots=True)
class GamepadInfo:
    """A connected gamepad, delivered on `gamepad_connected` / `gamepad_disconnected`.

    `index` is the GLFW joystick id (0..15). `mapped` is False for a joystick
    with no SDL gamepad mapping, whose buttons come from a guessed raw layout
    (see `_GlfwGamepadPoller`).
    """

    index: int
    name: str
    guid: str
    mapped: bool


@dataclass(frozen=True, slots=True)
class GamepadButtonEvent:
    """A gamepad button event delivered on `gamepad_down` or `gamepad_up`.

    `button` is a name from `GAMEPAD_BUTTONS`, a virtual direction from
    `GAMEPAD_DIRECTIONS`, or `"buttonN"` for an unmapped pad's raw button N.
    """

    pad: int
    button: str
    is_down: bool


@dataclass(frozen=True, slots=True)
class GamepadSnapshot:
    """One pad's state from one poll: the pressed button names and the axes.

    Plain data, so tests can build it without a device. Virtual directions
    are not included; `InputState` derives them.
    """

    info: GamepadInfo
    buttons: frozenset[str]
    axes: dict[str, float]


def _button_order(name: str) -> tuple[int, int, str]:
    if name in GAMEPAD_BUTTONS:
        return (0, GAMEPAD_BUTTONS.index(name), name)
    if name in GAMEPAD_DIRECTIONS:
        return (1, GAMEPAD_DIRECTIONS.index(name), name)
    return (2, len(name), name)  # "button2" before "button10"


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

    Gamepads are polled, not event-driven: the bridge calls
    `_record_gamepads` once per frame, right after `_begin_frame`, and the
    gamepad edges (`gamepad_just_pressed` / `gamepad_just_released`) are the
    difference between that poll and the previous one.
    """

    # Left-stick magnitude past which the stick counts as a virtual
    # direction ("up"/"down"/"left"/"right"). Set per instance to change it.
    gamepad_deadzone: float = 0.5

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

        # Gamepads, keyed by pad index. Buttons include virtual directions.
        self._gamepads: dict[int, GamepadInfo] = {}
        self._pad_buttons: dict[int, frozenset[str]] = {}
        self._pad_axes: dict[int, dict[str, float]] = {}
        self._pad_just_pressed: dict[int, frozenset[str]] = {}
        self._pad_just_released: dict[int, frozenset[str]] = {}

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

    # ---- Gamepad public API ----
    #
    # `pad=None` means any connected pad: buttons OR over all pads, axes read
    # the first pad (lowest index). An unknown pad reads False / 0.0.

    def gamepad_pressed(self, button: str, pad: int | None = None) -> bool:
        return self._any_pad(self._pad_buttons, button, pad)

    def gamepad_just_pressed(self, button: str, pad: int | None = None) -> bool:
        return self._any_pad(self._pad_just_pressed, button, pad)

    def gamepad_just_released(self, button: str, pad: int | None = None) -> bool:
        return self._any_pad(self._pad_just_released, button, pad)

    def gamepad_axis(self, name: str, pad: int | None = None) -> float:
        if pad is None:
            if not self._pad_axes:
                return 0.0
            pad = min(self._pad_axes)
        return self._pad_axes.get(pad, {}).get(name, 0.0)

    @property
    def gamepads(self) -> tuple[GamepadInfo, ...]:
        return tuple(self._gamepads[i] for i in sorted(self._gamepads))

    @staticmethod
    def _any_pad(per_pad: dict[int, frozenset[str]], button: str, pad: int | None) -> bool:
        if pad is None:
            return any(button in b for b in per_pad.values())
        return button in per_pad.get(pad, ())

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

    def _record_gamepads(self, snapshots) -> list[tuple[str, object]]:
        """Replace the gamepad state with this frame's poll.

        Returns the bus events the change implies, in order: per pad (by
        index) `gamepad_connected`, then `gamepad_up`s, then `gamepad_down`s,
        then `gamepad_disconnected`. A pad that disconnects releases every
        button it held, so `gamepad_up` handlers never see a stuck button.
        """
        new_infos = {s.info.index: s.info for s in snapshots}
        new_buttons = {s.info.index: self._with_directions(s) for s in snapshots}
        events: list[tuple[str, object]] = []
        just_pressed: dict[int, frozenset[str]] = {}
        just_released: dict[int, frozenset[str]] = {}
        for i in sorted(set(self._gamepads) | set(new_infos)):
            old = self._pad_buttons.get(i, frozenset())
            new = new_buttons.get(i, frozenset())
            pressed, released = new - old, old - new
            just_pressed[i], just_released[i] = pressed, released
            if i not in self._gamepads:
                events.append(("gamepad_connected", new_infos[i]))
            for b in sorted(released, key=_button_order):
                events.append(("gamepad_up", GamepadButtonEvent(pad=i, button=b, is_down=False)))
            for b in sorted(pressed, key=_button_order):
                events.append(("gamepad_down", GamepadButtonEvent(pad=i, button=b, is_down=True)))
            if i not in new_infos:
                events.append(("gamepad_disconnected", self._gamepads[i]))
        self._gamepads = new_infos
        self._pad_buttons = new_buttons
        self._pad_axes = {s.info.index: dict(s.axes) for s in snapshots}
        self._pad_just_pressed = just_pressed
        self._pad_just_released = just_released
        return events

    def _with_directions(self, snap: GamepadSnapshot) -> frozenset[str]:
        dz = self.gamepad_deadzone
        x = snap.axes.get("left_x", 0.0)
        y = snap.axes.get("left_y", 0.0)
        b = snap.buttons
        dirs = {
            "up": "dpad_up" in b or y < -dz,
            "down": "dpad_down" in b or y > dz,
            "left": "dpad_left" in b or x < -dz,
            "right": "dpad_right" in b or x > dz,
        }
        return b | {d for d, on in dirs.items() if on}

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

        # Gamepad edges live for one poll; the bridge's poll right after
        # this call sets the new ones (none, if there is no poller).
        self._pad_just_pressed = {}
        self._pad_just_released = {}


# Raw button index -> name for joysticks GLFW has no gamepad mapping for.
# This is the layout of the cheap generic USB pads (DragonRise 0079:0006,
# ShanWan, the 0810 "USB Gamepad"/Twin USB adapters): the SDL mappings that
# GLFW embeds for them agree on y:b0, b:b1, a:b2, x:b3, back:b8, start:b9,
# lstick:b10, rstick:b11, and most put the bumpers on b4/b5. It is a guess
# for an unknown pad; set MANIFOLDX_GAMEPAD_MAPPINGS for a real mapping.
_RAW_BUTTONS: dict[int, str] = {
    0: "y", 1: "b", 2: "a", 3: "x", 4: "lb", 5: "rb",
    8: "back", 9: "start", 10: "lstick", 11: "rstick",
}


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


class _GlfwGamepadPoller:
    """Read every connected joystick from GLFW into `GamepadSnapshot`s.

    Needs GLFW initialised, which `GlfwRenderCanvas` does. Pads with an SDL
    gamepad mapping (`joystick_is_gamepad`) read the 15 named buttons and 6
    axes; triggers are remapped from GLFW's [-1, 1] to [0, 1] so every axis
    rests at 0. Other joysticks fall back to raw access: hat 0 drives the
    d-pad, axes 0/1 the left stick, raw buttons get names from
    `_RAW_BUTTONS`, and every raw button and axis is also exposed as
    `"buttonN"` / `"axisN"`.

    If `MANIFOLDX_GAMEPAD_MAPPINGS` is set (an SDL mapping string, or the
    path to a gamecontrollerdb-style file), it is passed to
    `glfw.update_gamepad_mappings` once, here.
    """

    def __init__(self, glfw_module=None) -> None:
        if glfw_module is None:
            import glfw as glfw_module
        self._glfw = glfw_module
        mappings = os.environ.get("MANIFOLDX_GAMEPAD_MAPPINGS")
        if mappings:
            if os.path.isfile(mappings):
                with open(mappings, encoding="utf-8") as f:
                    mappings = f.read()
            self._glfw.update_gamepad_mappings(mappings)

    def poll(self) -> tuple[GamepadSnapshot, ...]:
        g = self._glfw
        return tuple(
            self._snapshot(jid)
            for jid in range(g.JOYSTICK_LAST + 1)
            if g.joystick_present(jid)
        )

    def _snapshot(self, jid: int) -> GamepadSnapshot:
        g = self._glfw
        guid = _text(g.get_joystick_guid(jid))
        state = g.get_gamepad_state(jid) if g.joystick_is_gamepad(jid) else None
        if state is not None:
            name = _text(g.get_gamepad_name(jid) or g.get_joystick_name(jid))
            buttons = frozenset(n for n, v in zip(GAMEPAD_BUTTONS, state.buttons) if v)
            axes = dict(zip(GAMEPAD_AXES, (float(v) for v in state.axes)))
            for t in ("lt", "rt"):
                axes[t] = (axes[t] + 1.0) * 0.5
            return GamepadSnapshot(GamepadInfo(jid, name, guid, True), buttons, axes)

        info = GamepadInfo(jid, _text(g.get_joystick_name(jid)), guid, False)
        raw_buttons, nb = g.get_joystick_buttons(jid)
        raw_axes, na = g.get_joystick_axes(jid)
        raw_hats, nh = g.get_joystick_hats(jid)
        pressed: set[str] = set()
        for i in range(nb):
            if raw_buttons[i]:
                pressed.add(f"button{i}")
                if i in _RAW_BUTTONS:
                    pressed.add(_RAW_BUTTONS[i])
        if nh:
            hat = raw_hats[0]
            for bit, name in ((g.HAT_UP, "dpad_up"), (g.HAT_RIGHT, "dpad_right"),
                              (g.HAT_DOWN, "dpad_down"), (g.HAT_LEFT, "dpad_left")):
                if hat & bit:
                    pressed.add(name)
        axes = {f"axis{i}": float(raw_axes[i]) for i in range(na)}
        if na >= 2:
            axes["left_x"], axes["left_y"] = axes["axis0"], axes["axis1"]
        return GamepadSnapshot(info, frozenset(pressed), axes)


class _InputBridge:
    """Forward rendercanvas events to the engine event bus and the InputState.

    Attaches as a single `add_event_handler` callback for all input event
    types we care about. Translates the rendercanvas dict into the matching
    typed event dataclass, mutates `InputState` immediately (so live reads
    are fresh), then `engine.emit(event_name, typed_event)` so the typed
    event rides the standard one-frame-delayed bus queue.

    Threading: rendercanvas marshals callbacks to the main thread before
    invoking us, so `_on_event` always runs on the same thread that pumps
    `_draw_frame`. No locking.
    """

    def __init__(self, engine, state: InputState) -> None:
        self._engine = engine
        self._state = state
        self._last_pointer_pos: tuple[float, float] | None = None

    def attach(self, canvas) -> None:
        canvas.add_event_handler(
            self._on_event,
            "key_down",
            "key_up",
            "pointer_down",
            "pointer_up",
            "pointer_move",
            "wheel",
            "resize",
        )

    def begin_frame(self) -> None:
        self._state._begin_frame()

    def _on_event(self, rc: dict) -> None:
        et = rc.get("event_type")
        if et == "key_down":
            ev = KeyEvent(key=rc["key"], modifiers=rc["modifiers"], is_down=True)
            self._state._record_key_down(ev.key, ev.modifiers)
            self._engine.emit("key_down", ev)
        elif et == "key_up":
            ev = KeyEvent(key=rc["key"], modifiers=rc["modifiers"], is_down=False)
            self._state._record_key_up(ev.key, ev.modifiers)
            self._engine.emit("key_up", ev)
        elif et in ("pointer_down", "pointer_up", "pointer_move"):
            ev = self._build_pointer_event(rc, et)
            if et == "pointer_down":
                self._state._record_pointer_down(ev)
                self._last_pointer_pos = (ev.x, ev.y)
            elif et == "pointer_up":
                self._state._record_pointer_up(ev)
                self._last_pointer_pos = (ev.x, ev.y)
            else:  # pointer_move
                self._state._record_pointer_move(ev)
                self._last_pointer_pos = (ev.x, ev.y)
            self._engine.emit(et, ev)
        elif et == "wheel":
            ev = WheelEvent(
                dx=rc["dx"], dy=rc["dy"],
                x=rc["x"], y=rc["y"],
                buttons=rc["buttons"], modifiers=rc["modifiers"],
            )
            self._state._record_wheel(ev)
            self._engine.emit("wheel", ev)
        elif et == "resize":
            ev = ResizeEvent(
                width=rc["width"], height=rc["height"],
                pixel_ratio=rc["pixel_ratio"],
            )
            self._state._record_resize(ev)
            self._engine.emit("resize", ev)
        # Other rendercanvas events (pointer_enter, pointer_leave,
        # double_click, char, before_draw, animate) are ignored in v1.

    def _build_pointer_event(self, rc: dict, et: str) -> PointerEvent:
        x, y = rc["x"], rc["y"]
        if et == "pointer_move" and self._last_pointer_pos is not None:
            dx = x - self._last_pointer_pos[0]
            dy = y - self._last_pointer_pos[1]
        else:
            dx = 0.0
            dy = 0.0
        phase = {
            "pointer_down": "down",
            "pointer_up": "up",
            "pointer_move": "move",
        }[et]
        return PointerEvent(
            x=x, y=y, dx=dx, dy=dy,
            button=rc.get("button", 0),
            buttons=rc["buttons"],
            modifiers=rc["modifiers"],
            phase=phase,
        )
