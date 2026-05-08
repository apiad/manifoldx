# Input Layer for ManifoldX — Design

**Date:** 2026-05-08
**Status:** Approved, ready for implementation planning.
**Scope:** Keyboard + mouse + window-resize input riding on the event bus from `2026-05-08-event-driven-system-design.md`. Bus events for discrete reactions, an `engine.input` polling state for held-key / continuous-motion reads. No camera-control helpers, no GPU picking — those are separate designs.

## Goal

Feed every keyboard and mouse signal from the canvas into both:

1. The existing `EventBus` as typed events (`@engine.on("key_down")` etc.), with dataclass payloads.
2. A queryable `engine.input` state object exposing Bevy-style `is_pressed` / `just_pressed` / `just_released` plus mouse position and per-frame deltas.

The motivating use cases — both first-class:

- **Camera control.** Drag-to-orbit, wheel-to-zoom, WASD fly. Mix of held-key polling (continuous motion) and discrete events (mouse delta).
- **Discrete interactions.** Hotkeys (`Space` to pause, `R` to reset), one-shot button presses, resize-driven layout updates.

Both `engine.input` reads and bus event handlers see consistent state every frame.

## Non-goals (v1)

- **`pointer_enter` / `pointer_leave` / `double_click` / `char`** — rendercanvas surfaces them but they aren't load-bearing for the camera/hotkey path. Trivial to add later.
- **Key auto-repeat.** rendercanvas's glfw backend explicitly drops `glfw.REPEAT`; held-key polling makes it unnecessary.
- **Pointer locking / cursor capture.** Needed for first-person games, not for orbit/fly. Deferred.
- **Touch / multi-touch.** `touches` is on the dict, glfw doesn't fill it.
- **Gamepad / joystick.** Not on rendercanvas's surface yet.
- **Higher-level camera controllers** (OrbitControls / FlyControls / TrackballControls). The two new examples are the prototypes; we extract a helper once two more demos want the same code.
- **GPU pick buffer** (click → entity_id). Its own design — separate render pass, async readback.
- **Input remapping / action sets** (Bevy-style `Input<Action>` with config bindings). Way out of scope.

## Architecture

A new `manifoldx.input` module, plus surgical additions to `Engine`:

```
src/manifoldx/
├── input.py     # NEW: KeyEvent / PointerEvent / WheelEvent / ResizeEvent
│                #      InputState (engine.input)
│                #      _InputBridge (canvas → bus + state)
└── engine.py    # adds: self.input, self._input_bridge
                 #       _input_bridge.attach(canvas) in _init_canvas
                 #       _input_bridge.begin_frame() at new step 2.5
```

### Data flow

```
rendercanvas callback (main thread, post-marshal)
        │
        ▼
canvas.add_event_handler(self._on_canvas_event, "key_down", ...)
        │
        ▼
_InputBridge._on_event(rc_dict)
        │
        ├── translate dict → typed event dataclass
        ├── update InputState pending buffers + live "currently pressed" sets
        ├── accumulate mouse_delta / wheel_delta / mouse_pos
        └── engine.emit(event_name, typed_event)   ← rides existing bus

frame N  ─ step 2: _frame_waiters.resolve(elapsed)
         ─ step 2.5: _input_bridge.begin_frame()
                     (swap pending → current for just_*; finalize deltas)
         ─ step 3: _event_bus.dispatch_pending(self)   ← input events drain here
         ─ step 4: pump asyncio
         ─ step 5: systems run        ← read engine.input.is_pressed("w")
         ─ step 6: commands flush
         ─ step 7: compute
         ─ step 8: render
```

**Why the split between "live" state and "frame-bound" state.** Sync handlers and systems both want fresh data when they read `engine.input.is_pressed("w")` — no frame lag. So `_pressed_keys` is mutated *immediately* on the bridge callback. But `just_pressed` / `just_released` and the delta accumulators are by definition this-frame-bound; they reset every frame. The new step 2.5 (`begin_frame`) is where that reset happens, before any handler or system runs.

**Why input emits go through `_event_bus._pending` like any other emit.** Consistency with the rest of the bus, and zero special-casing. Input events arrive between frames (during canvas-callback time, never during `_draw_frame`), so step 3 always sees them in their natural slot. The one-frame contract — "events emitted in frame N drain at step 3 of frame N+1" — holds without modification, *except* that `'frame'` already cuts the queue (it's prepended at step 3 with current-frame data); input events don't need that cut because handler-side `engine.input.*` reads already see live state.

## Module: `manifoldx.input`

### Event dataclasses

Frozen, slotted, dataclass-only. No methods beyond `__init__`. Handler ergonomics: attribute access, `dataclasses.replace` for tests.

```python
@dataclass(frozen=True, slots=True)
class KeyEvent:
    key: str                      # rendercanvas's name: "a", "Space", "ArrowUp", "Shift", ...
    modifiers: tuple[str, ...]    # e.g. ("Control", "Shift") — pre-aggregated, sorted
    is_down: bool                 # True for key_down, False for key_up

@dataclass(frozen=True, slots=True)
class PointerEvent:
    x: float                      # logical pixels, origin top-left
    y: float
    dx: float                     # delta from the previous pointer pos the bridge saw
    dy: float
    button: int                   # 1=left, 2=right, 3=middle on down/up; 0 on move
    buttons: tuple[int, ...]      # currently-held buttons, sorted
    modifiers: tuple[str, ...]
    phase: str                    # "down" | "up" | "move"

@dataclass(frozen=True, slots=True)
class WheelEvent:
    dx: float                     # rendercanvas-scaled (~100 per notch on glfw)
    dy: float
    x: float                      # pointer pos at scroll time
    y: float
    buttons: tuple[int, ...]
    modifiers: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ResizeEvent:
    width: int                    # logical pixels
    height: int
    pixel_ratio: float
```

**`PointerEvent.dx/dy` semantics.** Per-event delta from the previous pointer position the bridge observed, *not* this-frame accumulated movement. Rationale: a handler integrating mouse motion into camera rotation wants per-event deltas (smoother sub-frame integration when multiple `pointer_move` events fire same frame); a system polling cumulative movement reads `engine.input.mouse_delta` instead. `pointer_down` and `pointer_up` carry `dx=dy=0.0` and reset the bridge's "last pos" so a fresh drag starts clean (no jump from where the last drag ended).

### `InputState` (mounted as `engine.input`)

```python
class InputState:
    # ---- Keyboard ----
    def is_pressed(self, key: str) -> bool: ...
    def just_pressed(self, key: str) -> bool: ...
    def just_released(self, key: str) -> bool: ...
    @property
    def pressed_keys(self) -> frozenset[str]: ...
    @property
    def modifiers(self) -> tuple[str, ...]: ...

    # ---- Mouse ----
    @property
    def mouse_pos(self) -> tuple[float, float]: ...
    @property
    def mouse_delta(self) -> tuple[float, float]: ...   # this-frame accumulated
    @property
    def wheel_delta(self) -> tuple[float, float]: ...   # this-frame accumulated
    def is_mouse_pressed(self, button: int) -> bool: ...
    def just_mouse_pressed(self, button: int) -> bool: ...
    def just_mouse_released(self, button: int) -> bool: ...
    @property
    def pressed_buttons(self) -> tuple[int, ...]: ...

    # ---- Window ----
    @property
    def viewport_size(self) -> tuple[int, int]: ...     # last value seen on resize

    # ---- Internal mutators (bridge-only; underscore-prefixed) ----
    def _record_key_down(self, key: str, modifiers: tuple) -> None: ...
    def _record_key_up(self, key: str, modifiers: tuple) -> None: ...
    def _record_pointer_down(self, ev: PointerEvent) -> None: ...
    def _record_pointer_up(self, ev: PointerEvent) -> None: ...
    def _record_pointer_move(self, ev: PointerEvent) -> None: ...
    def _record_wheel(self, ev: WheelEvent) -> None: ...
    def _record_resize(self, ev: ResizeEvent) -> None: ...
    def _begin_frame(self) -> None: ...
```

`_record_*` methods mutate the live "currently pressed" sets and append to pending buffers (`_pending_just_pressed_keys` etc.) and accumulators (`_accum_mouse_dx`, `_accum_wheel_dy`, ...). `_begin_frame()` swaps pending → current for `just_*`, freezes the accumulators into `_mouse_delta` / `_wheel_delta`, then clears pending.

`InputState` exposes only properties and methods — no public attributes. Returns immutable types (`frozenset`, `tuple`) to discourage accidental mutation.

### `_InputBridge`

Owned by `Engine`, construction:

```python
class _InputBridge:
    def __init__(self, engine: Engine, state: InputState) -> None:
        self._engine = engine
        self._state = state
        self._last_pointer_pos: tuple[float, float] | None = None  # for dx/dy

    def attach(self, canvas) -> None:
        canvas.add_event_handler(self._on_event,
            "key_down", "key_up",
            "pointer_down", "pointer_up", "pointer_move",
            "wheel", "resize")

    def _on_event(self, rc: dict) -> None:
        et = rc["event_type"]
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
            elif et == "pointer_up":
                self._state._record_pointer_up(ev)
            else:
                self._state._record_pointer_move(ev)
            self._engine.emit(et, ev)
        elif et == "wheel":
            ev = WheelEvent(dx=rc["dx"], dy=rc["dy"], x=rc["x"], y=rc["y"],
                            buttons=rc["buttons"], modifiers=rc["modifiers"])
            self._state._record_wheel(ev)
            self._engine.emit("wheel", ev)
        elif et == "resize":
            ev = ResizeEvent(width=rc["width"], height=rc["height"],
                             pixel_ratio=rc["pixel_ratio"])
            self._state._record_resize(ev)
            self._engine.emit("resize", ev)
        # All other rendercanvas events (pointer_enter, double_click, char,
        # before_draw, animate) are ignored in v1.

    def _build_pointer_event(self, rc: dict, et: str) -> PointerEvent:
        x, y = rc["x"], rc["y"]
        if et == "pointer_move" and self._last_pointer_pos is not None:
            dx = x - self._last_pointer_pos[0]
            dy = y - self._last_pointer_pos[1]
        else:
            dx = dy = 0.0
        self._last_pointer_pos = (x, y)
        phase = {"pointer_down": "down", "pointer_up": "up", "pointer_move": "move"}[et]
        return PointerEvent(x=x, y=y, dx=dx, dy=dy,
                            button=rc.get("button", 0),
                            buttons=rc["buttons"],
                            modifiers=rc["modifiers"],
                            phase=phase)

    def begin_frame(self) -> None:
        self._state._begin_frame()
```

## Engine changes

`src/manifoldx/engine.py`:

- **`__init__`**: after the existing event-bus construction:
  ```python
  from manifoldx.input import InputState, _InputBridge
  self.input = InputState()
  self._input_bridge = _InputBridge(self, self.input)
  ```
- **`_init_canvas(canvas)`**: after the existing canvas wiring, before the first frame:
  ```python
  self._input_bridge.attach(canvas)
  ```
- **`_draw_frame`**: insert one line between waiter resolution and pending dispatch:
  ```python
  self._frame_waiters.resolve(self.elapsed)
  self._input_bridge.begin_frame()              # NEW: step 2.5
  ...
  self._event_bus.dispatch_pending(self)
  ```

No other engine code changes. No changes to `EventBus` itself.

## Threading

rendercanvas's glfw backend marshals events from its callback worker onto the main thread before invoking `add_event_handler` callbacks (verified by inspection of `rendercanvas/glfw.py:_on_*` paths — every backend callback ends in `submit_event` which queues onto the main loop). So `_on_event` always runs on the same thread that pumps `_draw_frame`. No locking needed.

This assumption is documented in `_InputBridge`'s docstring; if a future backend violates it (e.g., a hypothetical raw-pyqt path), that backend gains a queue. v1 doesn't pre-build it.

## Testing

Three layers, all running under the offscreen canvas (no GPU, runnable on CI machines without a display).

### Layer 1 — `InputState` unit tests (`tests/test_input_state.py`)

Pure-Python; no engine, no canvas. Direct calls to `_record_*` and `_begin_frame`. ~10 tests covering:

- `just_pressed` / `just_released` clear after `_begin_frame`.
- `is_pressed` stays true while held across frames.
- `mouse_delta` accumulates across multiple `pointer_move` events in one frame, then resets after `_begin_frame`.
- `wheel_delta` accumulates the same way.
- `mouse_pos` reflects the latest move event, *without* a frame swap.
- `pressed_keys` and `pressed_buttons` are sorted, immutable views.

### Layer 2 — `_InputBridge` translation tests (`tests/test_input_bridge.py`)

Feed raw rendercanvas dicts to `_InputBridge._on_event`, assert on:

- The typed event delivered through the bus.
- `InputState` mutations.
- `pointer_move` `dx/dy` is computed from the previous pointer position; resets on `pointer_down`/`pointer_up`.

### Layer 3 — End-to-end via `canvas.submit_event(...)` (`tests/test_input_e2e.py`)

The offscreen rendercanvas backend supports `submit_event`. Tests inject events through the public canvas surface, exercising the same path real input takes — `add_event_handler` registration, callback dispatch, bridge translation, bus emission, frame-loop integration:

```python
def test_synthetic_key_press_drives_engine_input():
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
```

Coverage targets: keyboard down → polling state, pointer drag → `mouse_delta`, wheel → `wheel_delta`, resize → `viewport_size`, bus delivery for each event name with the expected dataclass type.

## Examples

Two new files. Existing examples are untouched (none of them currently use input).

### `examples/input_orbit.py` — orbit camera (event-driven)

Drag with left button to rotate; wheel to zoom. Demonstrates `PointerEvent.dx/dy` and `WheelEvent.dy` directly off the bus.

The current `Camera` class exposes `position` directly (`camera.py:9-27`) but no `azimuth`/`elevation`/`distance` setters that recompute `position`. Two options:

1. **Helper inside the example.** Local function that takes (azimuth, elevation, distance, target) → position; example holds those values in module-level state. Smallest diff, no engine API change.
2. **Add `Camera.set_orbit(azimuth, elevation, distance, target)`.** Cleaner, reusable across future examples.

The implementation plan picks option 1 for v1 (smallest carve-off; no premature API commitment). If `input_fly.py` and a future demo both want orbit semantics, option 2 happens then.

### `examples/input_fly.py` — WASD fly cam (polling)

Demonstrates `engine.input.is_pressed(...)` and `engine.input.mouse_delta` from inside a system. W/A/S/D translate along camera basis; Space/Shift go up/down; right-mouse-drag rotates.

## Smoke renders

Both new examples need a `--render` smoke path that exercises the input system without a real device:

- The example accepts an optional `--inject-events` flag (or a per-example list of synthetic events) and feeds them via `canvas.submit_event` between frames during `engine.render(...)`.
- Detailed in the implementation plan (one task per example).

## Migration

Zero. None of the existing 16 examples or any test code uses input; the design adds capability without rewriting.

## Open caveats

- **`PointerEvent.dx/dy` semantics on `pointer_down`.** The bridge resets `_last_pointer_pos` on every down/up, so the *first* `pointer_move` after a `pointer_down` carries `dx=dy=0` (same anchor point). Documented; matches typical drag-handler expectations.
- **`InputState.mouse_delta` is one-frame-bound.** A system that doesn't run every frame will miss accumulated motion. v1 accepts this; if a use case surfaces, we add `engine.input.frame_history(n)` later.
- **`KeyEvent.key` strings come from rendercanvas's keymap** (`a`, `Space`, `ArrowUp`, `F1`, ...). We don't normalize. If we later need a manifoldx-native enum, it goes on top of the existing strings, not replacing them.
- **No queue overflow.** rendercanvas's `submit_event` is unbounded; if a frame stalls badly, all queued events deliver in one burst on the next `dispatch_pending`. Same behavior as the rest of the bus.
