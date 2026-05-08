# Input Layer v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the keyboard + mouse + window-resize input layer specced in `.knowledge/analysis/2026-05-08-input-events-design.md` — typed event dataclasses on the existing event bus, a Bevy-style `engine.input` polling state, the `_InputBridge` that translates rendercanvas events into both, and two demonstration examples.

**Architecture:** A new `manifoldx.input` module holds the `KeyEvent` / `PointerEvent` / `WheelEvent` / `ResizeEvent` dataclasses, the `InputState` polling object, and the `_InputBridge`. `Engine` gains `self.input` and `self._input_bridge`; the bridge attaches to the canvas in `_init_canvas` and gets a per-frame `begin_frame()` call inserted between waiter resolution and pending dispatch (new step 2.5). Bus emits ride the existing one-frame-delayed `_event_bus._pending` queue; polling-state reads are live (mutated immediately on the bridge callback).

**Tech Stack:** Python 3.13+, `dataclasses` (frozen + slotted), `rendercanvas` (offscreen + glfw backends), `pytest`, `uv`. Event-bus + frame-loop infrastructure from `2026-05-08-event-driven-system-design.md` is already in place.

**Branch:** All work happens on `input-v1` (matching the `events-v1` predecessor that just shipped). Branch off of `main`. Final merge to `main` happens outside the plan, the same way `events-v1` did.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/manifoldx/input.py` | **CREATE** | Event dataclasses (`KeyEvent`, `PointerEvent`, `WheelEvent`, `ResizeEvent`), `InputState`, `_InputBridge`. ~270 lines. |
| `src/manifoldx/engine.py` | **MODIFY** | `__init__` constructs `self.input` + `self._input_bridge`; `_init_canvas` calls `self._input_bridge.attach(canvas)`; `_draw_frame` inserts `self._input_bridge.begin_frame()` between waiter resolution and pending dispatch. |
| `src/manifoldx/__init__.py` | **MODIFY** | Re-export `KeyEvent`, `PointerEvent`, `WheelEvent`, `ResizeEvent` from `manifoldx.input` so users can write `from manifoldx import KeyEvent`. |
| `tests/test_input_state.py` | **CREATE** | Pure-Python unit tests on `InputState` (no engine, no canvas). |
| `tests/test_input_bridge.py` | **CREATE** | Bridge translation tests — feed raw rendercanvas dicts to `_InputBridge._on_event`, assert on bus emissions and state mutations. |
| `tests/test_input_e2e.py` | **CREATE** | End-to-end via `engine._render_canvas.submit_event(...)`, exercising the real rendercanvas registration path. |
| `examples/input_orbit.py` | **CREATE** | Drag-to-orbit + wheel-zoom camera, demonstrates event-driven `PointerEvent.dx/dy` and `WheelEvent.dy`. Carries a small local helper that recomputes `camera.position` from `(azimuth, elevation, distance, target)`. |
| `examples/input_fly.py` | **CREATE** | WASD fly cam in a system, demonstrates `engine.input.is_pressed(...)` and `engine.input.mouse_delta`. |
| `CHANGELOG.md` | **MODIFY** | New entry under `[Unreleased] / Features`. |

---

### Task 1: Event dataclasses (`KeyEvent`, `PointerEvent`, `WheelEvent`, `ResizeEvent`)

**Files:**
- Create: `src/manifoldx/input.py`
- Test: `tests/test_input_state.py` (this file accumulates across Tasks 1-3; we start it here with the dataclass tests)

- [ ] **Step 1: Write the failing test**

Create `tests/test_input_state.py`:

```python
"""InputState + event dataclass unit tests (no engine, no canvas)."""
import pytest
from dataclasses import FrozenInstanceError

from manifoldx.input import (
    KeyEvent,
    PointerEvent,
    WheelEvent,
    ResizeEvent,
)


def test_key_event_is_frozen():
    ev = KeyEvent(key="a", modifiers=("Shift",), is_down=True)
    assert ev.key == "a"
    assert ev.modifiers == ("Shift",)
    assert ev.is_down is True
    with pytest.raises(FrozenInstanceError):
        ev.key = "b"


def test_pointer_event_fields():
    ev = PointerEvent(
        x=10.0, y=20.0, dx=1.0, dy=2.0,
        button=1, buttons=(1,), modifiers=(), phase="down",
    )
    assert ev.x == 10.0
    assert ev.dx == 1.0
    assert ev.phase == "down"
    with pytest.raises(FrozenInstanceError):
        ev.x = 99.0


def test_wheel_event_fields():
    ev = WheelEvent(dx=0.0, dy=100.0, x=5.0, y=5.0, buttons=(), modifiers=())
    assert ev.dy == 100.0
    with pytest.raises(FrozenInstanceError):
        ev.dy = 0.0


def test_resize_event_fields():
    ev = ResizeEvent(width=800, height=600, pixel_ratio=2.0)
    assert ev.width == 800
    assert ev.pixel_ratio == 2.0
    with pytest.raises(FrozenInstanceError):
        ev.width = 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_input_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manifoldx.input'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/manifoldx/input.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_input_state.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/input.py tests/test_input_state.py
git commit -m "feat(input): KeyEvent / PointerEvent / WheelEvent / ResizeEvent dataclasses"
```

---

### Task 2: `InputState` — keyboard surface

**Files:**
- Modify: `src/manifoldx/input.py` (append `InputState` keyboard portion)
- Test: `tests/test_input_state.py` (append keyboard cases)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_input_state.py`:

```python
from manifoldx.input import InputState


def test_is_pressed_reflects_live_state_without_frame_swap():
    state = InputState()
    assert not state.is_pressed("w")
    state._record_key_down("w", modifiers=())
    # Live: no _begin_frame call yet, but is_pressed still answers True.
    assert state.is_pressed("w")
    state._record_key_up("w", modifiers=())
    assert not state.is_pressed("w")


def test_just_pressed_visible_only_after_frame_swap():
    state = InputState()
    state._record_key_down("w", modifiers=())
    # Before the swap, just_pressed is empty.
    assert not state.just_pressed("w")
    state._begin_frame()
    assert state.just_pressed("w")
    assert state.is_pressed("w")


def test_just_pressed_clears_on_subsequent_frame():
    state = InputState()
    state._record_key_down("w", modifiers=())
    state._begin_frame()
    assert state.just_pressed("w")
    state._begin_frame()
    assert not state.just_pressed("w")
    assert state.is_pressed("w")  # still held


def test_just_released_visible_only_after_frame_swap():
    state = InputState()
    state._record_key_down("w", modifiers=())
    state._begin_frame()
    state._record_key_up("w", modifiers=())
    assert not state.just_released("w")
    state._begin_frame()
    assert state.just_released("w")
    assert not state.is_pressed("w")
    state._begin_frame()
    assert not state.just_released("w")


def test_pressed_keys_returns_immutable_frozenset():
    state = InputState()
    state._record_key_down("a", modifiers=())
    state._record_key_down("b", modifiers=())
    keys = state.pressed_keys
    assert isinstance(keys, frozenset)
    assert keys == frozenset({"a", "b"})


def test_modifiers_property_reflects_latest_event():
    state = InputState()
    state._record_key_down("a", modifiers=("Shift",))
    assert state.modifiers == ("Shift",)
    state._record_key_up("a", modifiers=())
    assert state.modifiers == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_input_state.py -v`
Expected: FAIL with `ImportError: cannot import name 'InputState'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/manifoldx/input.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_input_state.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/input.py tests/test_input_state.py
git commit -m "feat(input): InputState keyboard surface (is/just_pressed/released)"
```

---

### Task 3: `InputState` — mouse + window surface

**Files:**
- Modify: `src/manifoldx/input.py` (extend `InputState`)
- Test: `tests/test_input_state.py` (append mouse + resize cases)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_input_state.py`:

```python
from manifoldx.input import PointerEvent, WheelEvent, ResizeEvent


def _pmove(x: float, y: float, dx: float, dy: float) -> PointerEvent:
    return PointerEvent(
        x=x, y=y, dx=dx, dy=dy,
        button=0, buttons=(), modifiers=(), phase="move",
    )


def _pdown(x: float, y: float, button: int) -> PointerEvent:
    return PointerEvent(
        x=x, y=y, dx=0.0, dy=0.0,
        button=button, buttons=(button,), modifiers=(), phase="down",
    )


def _pup(x: float, y: float, button: int) -> PointerEvent:
    return PointerEvent(
        x=x, y=y, dx=0.0, dy=0.0,
        button=button, buttons=(), modifiers=(), phase="up",
    )


def test_mouse_pos_reflects_latest_move_without_swap():
    state = InputState()
    assert state.mouse_pos == (0.0, 0.0)
    state._record_pointer_move(_pmove(10.0, 20.0, 0.0, 0.0))
    assert state.mouse_pos == (10.0, 20.0)
    state._record_pointer_move(_pmove(15.0, 25.0, 5.0, 5.0))
    assert state.mouse_pos == (15.0, 25.0)


def test_mouse_delta_accumulates_across_events_then_resets():
    state = InputState()
    state._record_pointer_move(_pmove(10.0, 10.0, 2.0, 0.0))
    state._record_pointer_move(_pmove(15.0, 10.0, 5.0, 0.0))
    # Before swap: accumulator is internal; mouse_delta property reads the
    # last finalized value (still (0,0) since no _begin_frame has happened).
    assert state.mouse_delta == (0.0, 0.0)
    state._begin_frame()
    assert state.mouse_delta == (7.0, 0.0)
    # Next frame with no events: delta resets to zero.
    state._begin_frame()
    assert state.mouse_delta == (0.0, 0.0)


def test_wheel_delta_accumulates_then_resets():
    state = InputState()
    state._record_wheel(WheelEvent(dx=0.0, dy=100.0, x=0.0, y=0.0,
                                   buttons=(), modifiers=()))
    state._record_wheel(WheelEvent(dx=0.0, dy=-50.0, x=0.0, y=0.0,
                                   buttons=(), modifiers=()))
    assert state.wheel_delta == (0.0, 0.0)
    state._begin_frame()
    assert state.wheel_delta == (0.0, 50.0)
    state._begin_frame()
    assert state.wheel_delta == (0.0, 0.0)


def test_mouse_button_pressed_lifecycle():
    state = InputState()
    state._record_pointer_down(_pdown(0.0, 0.0, button=1))
    assert state.is_mouse_pressed(1)
    state._begin_frame()
    assert state.just_mouse_pressed(1)
    assert state.is_mouse_pressed(1)
    state._begin_frame()
    assert not state.just_mouse_pressed(1)
    assert state.is_mouse_pressed(1)
    state._record_pointer_up(_pup(0.0, 0.0, button=1))
    state._begin_frame()
    assert state.just_mouse_released(1)
    assert not state.is_mouse_pressed(1)


def test_pressed_buttons_is_sorted_tuple():
    state = InputState()
    state._record_pointer_down(_pdown(0.0, 0.0, button=2))
    state._record_pointer_down(_pdown(0.0, 0.0, button=1))
    assert state.pressed_buttons == (1, 2)


def test_resize_updates_viewport_size_immediately():
    state = InputState()
    assert state.viewport_size == (0, 0)
    state._record_resize(ResizeEvent(width=800, height=600, pixel_ratio=1.0))
    assert state.viewport_size == (800, 600)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_input_state.py -v`
Expected: FAIL — `mouse_pos`, `mouse_delta`, `wheel_delta`, `is_mouse_pressed`, `pressed_buttons`, `viewport_size` not yet on `InputState`.

- [ ] **Step 3: Extend `InputState`**

In `src/manifoldx/input.py`, extend `InputState.__init__` and add the new methods. Find the existing `__init__`:

```python
    def __init__(self) -> None:
        # Keyboard
        self._pressed_keys: set[str] = set()
        self._modifiers: tuple[str, ...] = ()
        self._pending_just_pressed_keys: set[str] = set()
        self._pending_just_released_keys: set[str] = set()
        self._just_pressed_keys: frozenset[str] = frozenset()
        self._just_released_keys: frozenset[str] = frozenset()
```

Replace with:

```python
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
```

Then, after the keyboard public API block but before the bridge mutators, add the mouse + window public API:

```python
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
```

Add the new bridge mutators next to the existing `_record_key_*`:

```python
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
```

Finally, extend `_begin_frame` to swap mouse-button pending sets and finalize the accumulators. Replace the existing `_begin_frame` body:

```python
    def _begin_frame(self) -> None:
        # Swap keyboard pending → current.
        self._just_pressed_keys = frozenset(self._pending_just_pressed_keys)
        self._just_released_keys = frozenset(self._pending_just_released_keys)
        self._pending_just_pressed_keys.clear()
        self._pending_just_released_keys.clear()
```

with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_input_state.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/input.py tests/test_input_state.py
git commit -m "feat(input): InputState mouse + window surface, frame-swap accumulators"
```

---

### Task 4: `_InputBridge` — translation + bus emission

**Files:**
- Modify: `src/manifoldx/input.py` (append `_InputBridge`)
- Test: `tests/test_input_bridge.py`

This task covers all 7 rendercanvas event types (`key_down`, `key_up`, `pointer_down`, `pointer_up`, `pointer_move`, `wheel`, `resize`) in one go because the dispatch is a single switch and each branch is small.

- [ ] **Step 1: Write the failing test**

Create `tests/test_input_bridge.py`:

```python
"""Bridge translation tests: feed rendercanvas dicts, observe typed bus
emissions and InputState mutations. Uses a stub engine to avoid GPU."""
import pytest

from manifoldx.input import (
    InputState,
    KeyEvent,
    PointerEvent,
    ResizeEvent,
    WheelEvent,
    _InputBridge,
)


class _StubEngine:
    """Mimics the surface of `Engine` that `_InputBridge` touches."""

    def __init__(self) -> None:
        self.emitted: list[tuple[str, object]] = []

    def emit(self, event: str, payload) -> None:
        self.emitted.append((event, payload))


@pytest.fixture
def setup():
    state = InputState()
    engine = _StubEngine()
    bridge = _InputBridge(engine, state)
    return engine, state, bridge


def test_key_down_translates_and_records(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "key_down",
        "key": "w",
        "modifiers": (),
    })
    assert len(engine.emitted) == 1
    name, ev = engine.emitted[0]
    assert name == "key_down"
    assert isinstance(ev, KeyEvent)
    assert ev.key == "w"
    assert ev.is_down is True
    assert state.is_pressed("w")


def test_key_up_translates_and_records(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "key_down", "key": "w", "modifiers": ()})
    bridge._on_event({"event_type": "key_up", "key": "w", "modifiers": ()})
    name, ev = engine.emitted[-1]
    assert name == "key_up"
    assert isinstance(ev, KeyEvent)
    assert ev.is_down is False
    assert not state.is_pressed("w")


def test_pointer_down_carries_zero_delta(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "pointer_down",
        "x": 100.0, "y": 200.0,
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    name, ev = engine.emitted[0]
    assert name == "pointer_down"
    assert isinstance(ev, PointerEvent)
    assert ev.dx == 0.0 and ev.dy == 0.0
    assert ev.phase == "down"
    assert state.is_mouse_pressed(1)


def test_pointer_move_computes_dx_dy_from_last_pointer_pos(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "pointer_move",
        "x": 100.0, "y": 100.0,
        "button": 0, "buttons": (), "modifiers": (),
        "ntouches": 0, "touches": {},
    })
    # First move: no prior position → dx/dy are 0.
    _, ev1 = engine.emitted[0]
    assert ev1.dx == 0.0 and ev1.dy == 0.0
    bridge._on_event({
        "event_type": "pointer_move",
        "x": 110.0, "y": 105.0,
        "button": 0, "buttons": (), "modifiers": (),
        "ntouches": 0, "touches": {},
    })
    _, ev2 = engine.emitted[1]
    assert ev2.dx == 10.0 and ev2.dy == 5.0


def test_pointer_down_resets_last_pointer_pos(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "pointer_move", "x": 100.0, "y": 100.0,
                      "button": 0, "buttons": (), "modifiers": (),
                      "ntouches": 0, "touches": {}})
    bridge._on_event({"event_type": "pointer_down", "x": 200.0, "y": 200.0,
                      "button": 1, "buttons": (1,), "modifiers": (),
                      "ntouches": 0, "touches": {}})
    bridge._on_event({"event_type": "pointer_move", "x": 205.0, "y": 200.0,
                      "button": 0, "buttons": (1,), "modifiers": (),
                      "ntouches": 0, "touches": {}})
    # After the down event, the bridge anchors at (200, 200), so the next
    # move's delta is (5, 0), NOT (105, 100).
    _, ev = engine.emitted[-1]
    assert ev.dx == 5.0 and ev.dy == 0.0


def test_wheel_event_translates(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "wheel",
        "dx": 0.0, "dy": 100.0, "x": 50.0, "y": 50.0,
        "buttons": (), "modifiers": (),
    })
    name, ev = engine.emitted[0]
    assert name == "wheel"
    assert isinstance(ev, WheelEvent)
    assert ev.dy == 100.0


def test_resize_event_translates_and_updates_viewport(setup):
    engine, state, bridge = setup
    bridge._on_event({
        "event_type": "resize",
        "width": 1024, "height": 768, "pixel_ratio": 2.0,
    })
    name, ev = engine.emitted[0]
    assert name == "resize"
    assert isinstance(ev, ResizeEvent)
    assert ev.width == 1024
    assert state.viewport_size == (1024, 768)


def test_unknown_event_type_is_ignored(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "pointer_enter"})
    bridge._on_event({"event_type": "double_click", "x": 0, "y": 0})
    bridge._on_event({"event_type": "char", "char_str": "a"})
    assert engine.emitted == []


def test_begin_frame_triggers_state_swap(setup):
    engine, state, bridge = setup
    bridge._on_event({"event_type": "key_down", "key": "w", "modifiers": ()})
    assert not state.just_pressed("w")
    bridge.begin_frame()
    assert state.just_pressed("w")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_input_bridge.py -v`
Expected: FAIL with `ImportError: cannot import name '_InputBridge'`.

- [ ] **Step 3: Append `_InputBridge` to `src/manifoldx/input.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_input_bridge.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/input.py tests/test_input_bridge.py
git commit -m "feat(input): _InputBridge translates rendercanvas dicts to typed events"
```

---

### Task 5: Wire `engine.input` + bridge into `Engine`

**Files:**
- Modify: `src/manifoldx/engine.py`
- Modify: `src/manifoldx/__init__.py` (re-exports)

- [ ] **Step 1: Modify `src/manifoldx/engine.py` — construct input on `__init__`**

Find the import block at the top of `src/manifoldx/engine.py`:

```python
from manifoldx.camera import Camera
```

Append the input import on the next line (or wherever the project-internal imports group ends):

```python
from manifoldx.camera import Camera
from manifoldx.input import InputState, _InputBridge
```

Find the `__init__` block where `self._event_bus`, `self._aio_loop`, `self._frame_waiters`, and `self._task_errors` are constructed (currently around line 50-60 of `engine.py`):

```python
        self._event_bus = EventBus()
        self._aio_loop = asyncio.new_event_loop()
        self._frame_waiters = FrameWaiters(self._aio_loop)
        # Task error spool — populated by add_done_callback when an async
        # handler raises (other than CancelledError). Drained by
        # _pump_aio_loop, which re-raises the first error per the v1
        # "errors crash the engine" policy.
        self._task_errors: list[BaseException] = []
```

After that block, before the `# === ECS Infrastructure ===` comment, add:

```python
        # Input layer — keyboard + mouse + resize. The bridge attaches to
        # the canvas in _init_canvas; until then it is dormant.
        self.input = InputState()
        self._input_bridge = _InputBridge(self, self.input)
```

- [ ] **Step 2: Modify `_init_canvas` — attach the bridge to the canvas**

Find `_init_canvas` in `src/manifoldx/engine.py` (currently at line 432):

```python
    def _init_canvas(self, canvas):
        """Initialize WebGPU context from a canvas (shared by run() and render())."""
        self._render_canvas = canvas

        # Get the wgpu context from the canvas
        self._wgpu_context = canvas.get_wgpu_context()
```

Replace with:

```python
    def _init_canvas(self, canvas):
        """Initialize WebGPU context from a canvas (shared by run() and render())."""
        self._render_canvas = canvas

        # Wire input event flow: rendercanvas → _InputBridge → bus + state.
        self._input_bridge.attach(canvas)

        # Get the wgpu context from the canvas
        self._wgpu_context = canvas.get_wgpu_context()
```

- [ ] **Step 3: Modify `_draw_frame` — insert begin_frame at step 2.5**

Find the existing `_draw_frame` body where waiter resolution and dispatch_pending happen (currently around line 484-500):

```python
        # Step 2: resolve frame waiters (tick / delay / elapsed_at)
        self._frame_waiters.resolve(self.elapsed)

        # Clear command buffer ONCE at the head of the frame so events,
        # async handlers, and systems all contribute to the same buffer
        # that gets flushed at step 6.
        self.commands.clear()

        # Step 3: drain pending events (frame N-1's emits + this frame's 'frame')
```

Replace with:

```python
        # Step 2: resolve frame waiters (tick / delay / elapsed_at)
        self._frame_waiters.resolve(self.elapsed)

        # Step 2.5: input bridge frame swap — finalize this-frame
        # just_pressed/just_released sets and delta accumulators before
        # any handler or system runs.
        self._input_bridge.begin_frame()

        # Clear command buffer ONCE at the head of the frame so events,
        # async handlers, and systems all contribute to the same buffer
        # that gets flushed at step 6.
        self.commands.clear()

        # Step 3: drain pending events (frame N-1's emits + this frame's 'frame')
```

- [ ] **Step 4: Re-export the event dataclasses from the package root**

Find the existing exports in `src/manifoldx/__init__.py` and append (placement: alongside existing `from manifoldx.events import ...` if present, or at the end of the file):

```python
from manifoldx.input import KeyEvent, PointerEvent, WheelEvent, ResizeEvent
```

If `__init__.py` has an `__all__` list, append the four names to it:

```python
__all__ = [
    # ... existing entries ...
    "KeyEvent",
    "PointerEvent",
    "WheelEvent",
    "ResizeEvent",
]
```

If no `__all__` exists, the bare imports above are sufficient — `from manifoldx import KeyEvent` will work.

- [ ] **Step 5: Verify nothing broke**

Run: `uv run pytest tests/ -x 2>&1 | tail -30`
Expected: all existing tests still pass; the new `tests/test_input_state.py` and `tests/test_input_bridge.py` continue to pass.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/engine.py src/manifoldx/__init__.py
git commit -m "feat(input): wire engine.input + _InputBridge into Engine and _draw_frame"
```

---

### Task 6: End-to-end test via `canvas.submit_event`

**Files:**
- Test: `tests/test_input_e2e.py`

Tests through the real public canvas surface — `submit_event` works on the offscreen backend and exercises the same `add_event_handler` path real input devices use.

- [ ] **Step 1: Write the failing test**

Create `tests/test_input_e2e.py`:

```python
"""End-to-end input tests: inject events via canvas.submit_event, drive
frames through engine._draw_frame, observe both bus dispatch and
engine.input state."""
import pytest


def _make_offscreen_engine(width: int = 64, height: int = 64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_synthetic_key_down_drives_polling_state():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "key_down",
        "key": "w",
        "modifiers": (),
    })
    engine._draw_frame()
    assert engine.input.is_pressed("w")
    assert engine.input.just_pressed("w")
    engine._draw_frame()
    assert engine.input.is_pressed("w")
    assert not engine.input.just_pressed("w")


def test_synthetic_key_up_clears_polling_state():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "key_down", "key": "w", "modifiers": (),
    })
    engine._render_canvas.submit_event({
        "event_type": "key_up", "key": "w", "modifiers": (),
    })
    engine._draw_frame()
    assert not engine.input.is_pressed("w")
    assert engine.input.just_released("w")


def test_synthetic_pointer_event_dispatches_on_bus():
    engine = _make_offscreen_engine()
    from manifoldx.input import PointerEvent
    received: list[PointerEvent] = []

    @engine.on("pointer_down")
    def on_down(ev):
        received.append(ev)

    engine._render_canvas.submit_event({
        "event_type": "pointer_down",
        "x": 50.0, "y": 75.0,
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._draw_frame()
    assert len(received) == 1
    assert received[0].x == 50.0
    assert received[0].phase == "down"
    assert engine.input.is_mouse_pressed(1)


def test_synthetic_pointer_move_accumulates_delta():
    engine = _make_offscreen_engine()
    for x in (10.0, 15.0, 25.0):
        engine._render_canvas.submit_event({
            "event_type": "pointer_move",
            "x": x, "y": 0.0,
            "button": 0, "buttons": (), "modifiers": (),
            "ntouches": 0, "touches": {},
        })
    engine._draw_frame()
    # First move yields dx=0 (no prior anchor); subsequent moves yield 5 + 10.
    assert engine.input.mouse_delta == (15.0, 0.0)
    # Second frame with no events resets the delta.
    engine._draw_frame()
    assert engine.input.mouse_delta == (0.0, 0.0)


def test_synthetic_wheel_event_accumulates_delta():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "wheel",
        "dx": 0.0, "dy": 100.0, "x": 0.0, "y": 0.0,
        "buttons": (), "modifiers": (),
    })
    engine._render_canvas.submit_event({
        "event_type": "wheel",
        "dx": 0.0, "dy": -50.0, "x": 0.0, "y": 0.0,
        "buttons": (), "modifiers": (),
    })
    engine._draw_frame()
    assert engine.input.wheel_delta == (0.0, 50.0)


def test_synthetic_resize_updates_viewport_size():
    engine = _make_offscreen_engine()
    engine._render_canvas.submit_event({
        "event_type": "resize",
        "width": 1024, "height": 768, "pixel_ratio": 1.0,
    })
    engine._draw_frame()
    assert engine.input.viewport_size == (1024, 768)
```

- [ ] **Step 2: Run test to verify it fails (or passes by accident)**

Run: `uv run pytest tests/test_input_e2e.py -v`
Expected: PASS — Task 5 already wired the bridge into `_init_canvas`. If anything fails here, the failure pinpoints a wiring gap from Task 5.

If a test fails because `submit_event` queues asynchronously and the event hasn't arrived by the time `_draw_frame` runs, add a one-line guard before each `_draw_frame` call: `engine._render_canvas._process_events()` (offscreen-canvas private API that flushes the queue synchronously). Inspect the failure first; do not pre-emptively add the guard.

- [ ] **Step 3: Commit**

```bash
git add tests/test_input_e2e.py
git commit -m "test(input): end-to-end via canvas.submit_event"
```

---

### Task 7: `examples/input_orbit.py` — drag-to-orbit + wheel-zoom

**Files:**
- Create: `examples/input_orbit.py`

The current `Camera` exposes `position` directly but no `(azimuth, elevation, distance, target)` setters. The example carries a tiny local helper that recomputes `camera.position` from spherical coords; this is intentional — we don't add a `Camera.set_orbit` API until a second example wants it.

- [ ] **Step 1: Write the example**

Create `examples/input_orbit.py`:

```python
"""Event-driven orbit camera: drag with left button to rotate, wheel to zoom.

Demonstrates `PointerEvent.dx/dy` and `WheelEvent.dy` directly off the bus.
The orbit math (azimuth/elevation/distance → camera.position) is local to
this example; it does not (yet) live on the Camera class.
"""

import math
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.input import PointerEvent, WheelEvent


engine = mx.Engine("Input Orbit")

cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.RED)


# Orbit state — module-level so the handlers and recomputation share it.
_orbit = {
    "azimuth": 45.0,    # degrees
    "elevation": 25.0,  # degrees
    "distance": 5.0,
    "target": np.array([0.0, 0.0, 0.0], dtype=np.float32),
}


def _recompute_camera_position() -> None:
    """Map (azimuth, elevation, distance, target) → camera.position."""
    a = math.radians(_orbit["azimuth"])
    e = math.radians(_orbit["elevation"])
    r = _orbit["distance"]
    target = _orbit["target"]
    pos = target + np.array([
        r * math.cos(e) * math.cos(a),
        r * math.sin(e),
        r * math.cos(e) * math.sin(a),
    ], dtype=np.float32)
    engine.camera.position = pos
    engine.camera.target = target


@engine.on("startup")
def setup(_payload):
    engine.spawn(
        Mesh(cube_mesh),
        Material(cube_material),
        Transform(pos=(0, 0, 0)),
        n=1,
    )
    _recompute_camera_position()


@engine.on("pointer_move")
def orbit(ev: PointerEvent):
    # Rotate when left button (1) is held during the move.
    if 1 in ev.buttons:
        _orbit["azimuth"] += ev.dx * 0.5
        _orbit["elevation"] = max(-89.0, min(89.0, _orbit["elevation"] + ev.dy * 0.5))
        _recompute_camera_position()


@engine.on("wheel")
def zoom(ev: WheelEvent):
    # rendercanvas wheel dy is ~100 per notch; scale to a 5 % step per notch.
    _orbit["distance"] = max(0.5, _orbit["distance"] * (1.0 - ev.dy * 0.0005))
    _recompute_camera_position()


if __name__ == "__main__":
    engine.cli()
```

- [ ] **Step 2: Smoke render with synthetic input**

The `--render` path runs headless without a real mouse, so we cannot exercise the orbit handlers from the CLI. Instead, smoke-test that the example imports and runs one frame without raising:

```bash
uv run python -c "
import importlib, sys
sys.argv = ['input_orbit.py', '--render', '--duration', '1', '--fps', '10', '--output', '/tmp/input_orbit_smoke.mp4']
spec = importlib.util.spec_from_file_location('input_orbit', 'examples/input_orbit.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
"
```

Expected: produces `/tmp/input_orbit_smoke.mp4` without exceptions. The video shows a static red cube (no input → no orbit), which is the correct behavior for the smoke test.

- [ ] **Step 3: Commit**

```bash
git add examples/input_orbit.py
git commit -m "feat(examples): input_orbit.py — drag-to-orbit + wheel-zoom via input events"
```

---

### Task 8: `examples/input_fly.py` — WASD fly cam (polling)

**Files:**
- Create: `examples/input_fly.py`

- [ ] **Step 1: Write the example**

Create `examples/input_fly.py`:

```python
"""WASD fly-cam driven by polling state on `engine.input`.

Demonstrates `engine.input.is_pressed(...)` and `engine.input.mouse_delta`
read from inside a system. W/A/S/D translate along the camera basis;
Space/Shift go up/down; right-mouse-drag rotates the look direction.
"""

import math
import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.systems import Query


engine = mx.Engine("Input Fly")

cube_mesh = mx.geometry.cube(1, 1, 1)
cube_material = mx.material.phong(mx.colors.GREEN)


# Fly state mirrors the orbit example pattern.
_fly = {
    "azimuth": 45.0,
    "elevation": 0.0,
    "speed": 5.0,
}


def _direction_from(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    return np.array([
        math.cos(e) * math.cos(a),
        math.sin(e),
        math.cos(e) * math.sin(a),
    ], dtype=np.float32)


@engine.on("startup")
def setup(_payload):
    # A 5x5 grid of cubes so motion is visible.
    for ix in range(-2, 3):
        for iz in range(-2, 3):
            engine.spawn(
                Mesh(cube_mesh),
                Material(cube_material),
                Transform(pos=(ix * 2.0, 0.0, iz * 2.0)),
                n=1,
            )
    engine.camera.position = np.array([0.0, 1.0, 5.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 1.0, 0.0], dtype=np.float32)


@engine.system
def fly(query: Query[Transform], dt: float):
    # Rotate when right button (2) is held; mouse_delta is this-frame only.
    if engine.input.is_mouse_pressed(2):
        dx, dy = engine.input.mouse_delta
        _fly["azimuth"] += dx * 0.2
        _fly["elevation"] = max(-89.0, min(89.0, _fly["elevation"] - dy * 0.2))

    forward = _direction_from(_fly["azimuth"], _fly["elevation"])
    # Right vector = world-up × forward, then normalize.
    right = np.cross(np.array([0, 1, 0], np.float32), forward)
    n = np.linalg.norm(right)
    if n > 0:
        right = right / n

    move = np.zeros(3, np.float32)
    if engine.input.is_pressed("w"): move += forward
    if engine.input.is_pressed("s"): move -= forward
    if engine.input.is_pressed("d"): move += right
    if engine.input.is_pressed("a"): move -= right
    if engine.input.is_pressed("Space"): move[1] += 1
    if engine.input.is_pressed("Shift"): move[1] -= 1

    nm = np.linalg.norm(move)
    if nm > 0:
        engine.camera.position = engine.camera.position + (move / nm) * _fly["speed"] * dt

    # Always look in the current forward direction.
    engine.camera.target = engine.camera.position + forward


if __name__ == "__main__":
    engine.cli()
```

- [ ] **Step 2: Smoke render**

```bash
uv run python -c "
import importlib, sys
sys.argv = ['input_fly.py', '--render', '--duration', '1', '--fps', '10', '--output', '/tmp/input_fly_smoke.mp4']
spec = importlib.util.spec_from_file_location('input_fly', 'examples/input_fly.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
"
```

Expected: produces `/tmp/input_fly_smoke.mp4` showing a static 5×5 grid of green cubes (no input → no motion, but the system still runs).

- [ ] **Step 3: Commit**

```bash
git add examples/input_fly.py
git commit -m "feat(examples): input_fly.py — WASD fly cam via engine.input polling"
```

---

### Task 9: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the entry**

Open `CHANGELOG.md`. Under the existing `## [Unreleased]` heading, in the `### Features` block, prepend the following bullet:

```markdown
- **Input layer v1** — keyboard, mouse, and resize events ride the event bus from event-driven-system v1. New `manifoldx.input` module exposes `KeyEvent`, `PointerEvent`, `WheelEvent`, and `ResizeEvent` dataclasses (re-exported from the package root). Handlers register the same way as any other event: `@engine.on("key_down")`, `@engine.on("pointer_move")`, etc. Alongside discrete events, a polling state at `engine.input` exposes Bevy-style `is_pressed(key)` / `just_pressed(key)` / `just_released(key)` for keys and mouse buttons, plus `mouse_pos`, `mouse_delta`, `wheel_delta`, and `viewport_size`. Polling state is updated immediately on every event (so `is_pressed` reads are live), while `just_*` sets and `*_delta` accumulators are this-frame-bound and reset at the new step 2.5 of `_draw_frame`. New `examples/input_orbit.py` (drag-to-orbit + wheel-zoom via events) and `examples/input_fly.py` (WASD fly cam via polling). Design: `.knowledge/analysis/2026-05-08-input-events-design.md`.
```

- [ ] **Step 2: Verify the file parses**

```bash
head -40 CHANGELOG.md
```

Expected: the new bullet shows under `[Unreleased] / Features`, before earlier entries.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(input): CHANGELOG entry for input layer v1"
```

---

## Final verification

- [ ] **Step 1: Full test suite**

Run: `uv run pytest tests/ -v 2>&1 | tail -40`
Expected: all tests pass. No `RuntimeWarning` about the asyncio loop, no leaked rendercanvas event handlers.

- [ ] **Step 2: Lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 3: Smoke-render the new examples + a sample of existing examples**

Confirms the input wiring didn't break anything else:

```bash
uv run python examples/cube.py --render --duration 1 --fps 10 --output /tmp/cube_post.mp4
uv run python examples/event_dolly.py --render --duration 2 --fps 30 --output /tmp/dolly_post.mp4
uv run python -c "
import importlib, sys
for name in ('input_orbit', 'input_fly'):
    sys.argv = [f'{name}.py', '--render', '--duration', '1', '--fps', '10', '--output', f'/tmp/{name}_post.mp4']
    spec = importlib.util.spec_from_file_location(name, f'examples/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
"
```

Expected: all four MP4s produced cleanly.

- [ ] **Step 4: Confirm public API surface**

```bash
uv run python -c "
import manifoldx as mx
assert hasattr(mx, 'KeyEvent')
assert hasattr(mx, 'PointerEvent')
assert hasattr(mx, 'WheelEvent')
assert hasattr(mx, 'ResizeEvent')
e = mx.Engine('check', width=32, height=32)
assert hasattr(e, 'input')
assert callable(e.input.is_pressed)
print('OK')
"
```

Expected: prints `OK`.

- [ ] **Step 5: Confirm no leftover scaffolding**

```bash
grep -rn "TODO\|FIXME\|XXX" src/manifoldx/input.py examples/input_orbit.py examples/input_fly.py
```

Expected: no matches (or only pre-existing comments unrelated to this work).
