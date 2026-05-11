# GUI v1 — Plan 2: Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the interactive surface of GUI v1 — `_GuiBridge` (pointer-event router with hit-testing and drag capture), three interactive widgets (`Button`, `Slider`, `Toggle`), a demo example, and the final CHANGELOG entry that closes the GUI v1 sub-project.

**Architecture:** `_GuiBridge` is a small router constructed on `Engine.__init__` (right after `engine.gui = _GuiRoot()`). It subscribes to `pointer_down` / `pointer_up` / `pointer_move` via `@engine.on(...)` (no special wiring — the event bus already exists). On every pointer event it walks the widget tree top-down, finds the topmost interactive widget under the cursor, and routes the event accordingly. It tracks one piece of state — the currently captured widget (slider drag in progress). Each interactive widget owns its own state mutation (click latch, toggle value, slider value) and emits `gui:<kind>:<name>:<event>` on the bus. `engine.gui.pointer_over_gui` is set true on any pointer event that hits a widget; the bridge resets it during `engine._draw_frame()` right after `_input_bridge.begin_frame()`.

**Tech Stack:** Python 3.13+, wgpu, numpy, pytest. Builds on the GUI foundation from Plan 1 (`Panel`, `Text`, `ValueDisplay`, `Painter`, `RectMaterial`, gui render pass).

**Scope:** Interaction only — no new render-pass primitives, no atlas changes, no engine.py refactors beyond the bridge construction and the one `begin_frame()` hook. Widgets reuse the existing rect + glyph paint paths from Plan 1.

**Design reference:** `.knowledge/analysis/2026-05-10-gui-v1-design.md`. Especially: "Public API → Widgets", "Events", "Input flow → Hit testing / Drag capture / Cooperative consume".

**Prereqs:** Plan 1 landed (commit `39e5f63` ships the foundation; suite at 506 passing tests).

---

## File Structure

**New files (under `src/manifoldx/gui/`):**

- `bridge.py` — `_GuiBridge` class (pointer routing, hit-test, drag capture, pointer_over_gui flag management).
- `hit_test.py` — pure-Python `hit_test(panels, x, y) -> Widget | None` helper. Separated from bridge so it's CPU-testable without the engine.
- `button.py` — `Button` widget.
- `toggle.py` — `Toggle` widget.
- `slider.py` — `Slider` widget. Slider is the only widget that captures the pointer (drag), so it carries the most state.

**Modified files:**

- `src/manifoldx/gui/__init__.py` — export `Button`, `Slider`, `Toggle`.
- `src/manifoldx/gui/painter.py` — add paint branches for `Button` / `Slider` / `Toggle`.
- `src/manifoldx/engine.py` — construct `self._gui_bridge = _GuiBridge(self)` in `__init__` (right after `self.gui = _GuiRoot()`), and call `self._gui_bridge.begin_frame()` inside `_draw_frame()` right after `self._input_bridge.begin_frame()`.
- `CHANGELOG.md` — replace the "GUI v1 — foundation (in progress)" bullet with the final shipped entry.
- `AGENTS.md` — update "Sub-projects in flight" to mark GUI v1 fully landed.

**New tests:**

- `tests/gui/test_bridge.py` — pure-CPU bridge tests (synthetic pointer event dispatch).
- `tests/gui/test_hit_test.py` — pure-CPU hit-test tests.
- `tests/gui/test_button.py` — Button construction + click handling + was_clicked latch.
- `tests/gui/test_toggle.py` — Toggle construction + click flips value + change event.
- `tests/gui/test_slider.py` — Slider construction + drag updates value + change/commit events.
- `tests/gui/test_interaction_e2e.py` — e2e: spawn an engine + offscreen canvas, push synthetic pointer events via `canvas.submit_event`, assert handlers fired with the right payloads.

**New examples:**

- `examples/gui_demo.py` — N-body rerun with docked control panel.

---

## Conventions (from Plan 1, confirmed)

- **Plans dir:** `.knowledge/plans/<date>-<slug>.md`.
- **TDD:** failing test → confirm fail → minimal impl → confirm pass → commit. Don't skip the fail step.
- **Per-test invocation:** `uv run pytest tests/gui/<file>.py::<test> -v`
- **Full suite:** `make test`
- **Lint:** `make lint`. Stay clean (E402 in `compute/_core.py` is pre-existing — out of scope).
- **Commit per task:** conventional `feat(gui):` / `test(gui):` / `docs(gui):`.
- **No push** unless the controller explicitly authorizes. Plan 1 had one subagent push to origin/main without ask — don't repeat.
- **WGSL alignment vigilance:** Plan 1 caught a 14→16 float alignment bug in RectInstance. Stay alert for similar issues. (Plan 2 doesn't add new WGSL; reuses the existing rect + glyph pipelines.)

---

## Task 1: `_GuiBridge` skeleton + `pointer_over_gui` lifecycle

**Files:**
- Create: `src/manifoldx/gui/bridge.py`
- Modify: `src/manifoldx/engine.py`
- Create: `tests/gui/test_bridge.py`

### Step 1.1 — Failing test for bridge construction + flag reset

Create `tests/gui/test_bridge.py`:

```python
"""Tests for manifoldx.gui.bridge — pointer routing skeleton."""

import pytest

from manifoldx.gui import Panel, style
from manifoldx.gui.bridge import _GuiBridge


def setup_function(_):
    style.reset()


def test_bridge_constructs_with_engine_and_subscribes_to_pointer_events():
    # Synthetic engine stub: provides .emit(), .on(), and .gui attribute.
    received = []

    class _Stub:
        def __init__(self):
            self.gui = _GuiRootStub()
            self._handlers = {}
        def on(self, event_name):
            def decorator(fn):
                self._handlers.setdefault(event_name, []).append(fn)
                return fn
            return decorator
        def emit(self, name, payload):
            received.append((name, payload))

    class _GuiRootStub:
        def __init__(self):
            self.pointer_over_gui = False
            self._panels = []
        def __iter__(self):
            return iter(self._panels)
        def __len__(self):
            return len(self._panels)

    eng = _Stub()
    bridge = _GuiBridge(eng)
    # Bridge subscribed to all three pointer events.
    assert "pointer_down" in eng._handlers
    assert "pointer_move" in eng._handlers
    assert "pointer_up" in eng._handlers


def test_bridge_begin_frame_clears_pointer_over_gui_flag():
    class _Stub:
        def __init__(self):
            self.gui = type("_G", (), {"pointer_over_gui": True, "_panels": []})()
            self.gui.__iter__ = lambda self: iter([])
            self.gui.__len__ = lambda self: 0
            self._handlers = {}
        def on(self, event_name):
            def deco(fn):
                self._handlers.setdefault(event_name, []).append(fn)
                return fn
            return deco
        def emit(self, *a, **k):
            pass
    eng = _Stub()
    bridge = _GuiBridge(eng)
    eng.gui.pointer_over_gui = True
    bridge.begin_frame()
    assert eng.gui.pointer_over_gui is False
```

Run: `uv run pytest tests/gui/test_bridge.py -v` — expect 2 FAIL (no module).

### Step 1.2 — Implement `bridge.py` (skeleton only)

Create `src/manifoldx/gui/bridge.py`:

```python
"""GUI input bridge — routes pointer events from the bus to widget state.

Subscribes to `pointer_down` / `pointer_move` / `pointer_up` via `engine.on(...)`.
Each event walks `engine.gui` top-down (last appended panel = topmost), uses
`hit_test` (Task 2) to find the topmost interactive widget under the cursor,
and routes the event accordingly.

State:
- `_captured`: widget currently in a drag (Slider only in v1). While captured,
  every `pointer_move` goes to that widget regardless of where the cursor is.
- `engine.gui.pointer_over_gui`: set True on any pointer event that hits a
  widget; reset to False at the head of every frame (in `begin_frame`).

Cooperative consume: the event bus does NOT have stop-propagation. User systems
must check `engine.gui.pointer_over_gui` themselves to skip game-world clicks.
"""

from __future__ import annotations

from typing import Any


class _GuiBridge:
    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._captured: Any | None = None

        @engine.on("pointer_down")
        def _on_down(ev):  # noqa: F841
            self._on_pointer(ev, "down")

        @engine.on("pointer_move")
        def _on_move(ev):  # noqa: F841
            self._on_pointer(ev, "move")

        @engine.on("pointer_up")
        def _on_up(ev):  # noqa: F841
            self._on_pointer(ev, "up")

    def begin_frame(self) -> None:
        """Reset per-frame state. Called by Engine._draw_frame after the
        input bridge's begin_frame."""
        self._engine.gui.pointer_over_gui = False

    def _on_pointer(self, ev: Any, phase: str) -> None:
        """Stub — routing logic lands in Task 2 (hit-test) and Tasks 3–5
        (per-widget handling). For now the bridge silently swallows events."""
        return
```

Run tests — 2 PASS.

### Step 1.3 — Wire bridge into engine + `begin_frame` hook

In `src/manifoldx/engine.py`, find the GUI wiring block from Plan 1 Task 6:

```python
        # GUI layer — list-like container of root Panels plus a
        # pointer_over_gui flag. The bridge (Plan 2) will toggle the flag.
        from manifoldx.gui.widgets import _GuiRoot
        self.gui = _GuiRoot()
```

Append immediately after:

```python

        # GUI input bridge — routes pointer events through hit-test
        # and per-widget state. Subscribes to pointer_down/move/up at
        # construction; needs `self.gui` to already exist.
        from manifoldx.gui.bridge import _GuiBridge
        self._gui_bridge = _GuiBridge(self)
```

In `_draw_frame`, find the input-bridge `begin_frame` call (around line 498):

```python
        # Step 2.5: input bridge frame swap — finalize this-frame
        # just_pressed/just_released sets and delta accumulators before
        # any handler or system runs.
        self._input_bridge.begin_frame()
```

Append immediately after:

```python
        # GUI bridge frame reset — pointer_over_gui rolls back to False
        # at the head of each frame; the per-event hit-test will set it
        # True again if the cursor is over a widget this frame.
        self._gui_bridge.begin_frame()
```

### Step 1.4 — Run gui suite + full suite

`uv run pytest tests/gui/ -v` — all PASS (including the new bridge tests).
`make test` — full suite still green. Expected: 508 passing (506 + 2 new bridge tests).

### Step 1.5 — Lint + commit

```bash
git add src/manifoldx/gui/bridge.py src/manifoldx/engine.py tests/gui/test_bridge.py
git commit -m "feat(gui): _GuiBridge skeleton + pointer_over_gui begin_frame reset"
```

---

## Task 2: Hit testing

**Files:**
- Create: `src/manifoldx/gui/hit_test.py`
- Modify: `src/manifoldx/gui/bridge.py` (wire hit-test into `_on_pointer`)
- Create: `tests/gui/test_hit_test.py`
- Modify: `tests/gui/test_bridge.py` (assert flag flips on hit)

### Step 2.1 — Failing test for `hit_test`

Create `tests/gui/test_hit_test.py`:

```python
"""Tests for manifoldx.gui.hit_test — topmost interactive widget under a point."""

from manifoldx.gui import Panel, Text, style
from manifoldx.gui.hit_test import hit_test
from manifoldx.gui.layout import LayoutBox, compute_layout


def setup_function(_):
    style.reset()


def test_hit_test_returns_none_when_no_panels():
    assert hit_test([], 50.0, 50.0, viewport=LayoutBox(0, 0, 100, 100)) is None


def test_hit_test_returns_none_when_point_outside_all_panels():
    p = Panel(children=[], anchor="top-left", offset=(0, 0),
              style_overrides={"width": 30, "height": 30})
    # Click far outside the panel's 30x30 bbox.
    assert hit_test([p], 200.0, 200.0, viewport=LayoutBox(0, 0, 256, 256)) is None


def test_hit_test_non_interactive_widgets_dont_consume_hits():
    # A panel containing only a Text widget — Text isn't interactive,
    # so the hit returns None (the panel itself isn't "interactive" either).
    p = Panel(children=[Text("hi")], anchor="top-left", offset=(0, 0),
              style_overrides={"width": 100, "height": 30, "padding": 0})
    assert hit_test([p], 10.0, 10.0, viewport=LayoutBox(0, 0, 256, 256)) is None


def test_hit_test_later_panel_topmost():
    """When two panels overlap, the panel appended LAST wins."""
    # We don't have interactive widgets yet in Plan 2 Task 2, so the
    # interactive-only contract gets exercised in Tasks 3–5. This test
    # locks in the "last panel walked first" iteration order via a
    # synthetic interactive marker.
    from manifoldx.gui.widgets import Widget

    class _Marker(Widget):
        def __init__(self, name):
            super().__init__()
            self._name = name
        def intrinsic_size(self):
            return (10.0, 10.0)
        @property
        def _is_gui_interactive(self):
            return True

    a, b = _Marker("a"), _Marker("b")
    p1 = Panel(children=[a], anchor="top-left", offset=(0, 0),
               style_overrides={"width": 50, "height": 50, "padding": 0})
    p2 = Panel(children=[b], anchor="top-left", offset=(0, 0),
               style_overrides={"width": 50, "height": 50, "padding": 0})
    # p2 appended later → checked first → b wins.
    hit = hit_test([p1, p2], 10.0, 10.0, viewport=LayoutBox(0, 0, 256, 256))
    assert hit is b
```

Run: `uv run pytest tests/gui/test_hit_test.py -v` — expect 4 FAIL.

### Step 2.2 — Implement `hit_test.py`

Create `src/manifoldx/gui/hit_test.py`:

```python
"""Topmost-interactive-widget lookup at a screen point.

A widget participates in hit testing only if it sets the attribute or
property `_is_gui_interactive = True`. `Button`, `Slider`, `Toggle` (Tasks 3–5)
will provide this; `Text`, `ValueDisplay`, `Panel` will not — they are
non-interactive and don't consume hits.

The lookup walks `panels` in REVERSE order (last appended = topmost), then
recurses into each panel's children depth-first, returning the deepest
interactive widget whose layout box contains `(x, y)`. Returns `None` if no
interactive widget matches.

`viewport` is the screen rect compute_layout will see; it's typically
`LayoutBox(0, 0, engine.w, engine.h)`. We need this because root panels'
slot positions come from their `anchor`+`offset`+(width|height), not from a
pre-computed layout — the bridge can't share the gui pass's layout map.
"""

from __future__ import annotations

from typing import Iterable

from manifoldx.gui.layout import LayoutBox, compute_layout
from manifoldx.gui.widgets import Panel, Widget


def hit_test(
    panels: Iterable[Panel],
    x: float,
    y: float,
    viewport: LayoutBox,
) -> Widget | None:
    """Return the topmost interactive widget under (x, y), or None."""
    panels = list(panels)
    for panel in reversed(panels):
        spec = panel.build_layout_spec()
        slot = _anchored_slot(panel, viewport)
        boxes = compute_layout(spec, viewport=slot)
        hit = _walk(panel, spec, boxes, x, y)
        if hit is not None:
            return hit
    return None


def _anchored_slot(panel: Panel, viewport: LayoutBox) -> LayoutBox:
    """Compute the panel's layout slot from anchor + offset + explicit size.

    Mirrors the same logic the gui render pass uses to keep hit-test and
    paint geometry in sync. Currently top-left anchor only; the gui pass
    will grow the rest as examples need it.
    """
    s = panel.effective_style()
    w = s.get("width") or viewport.w
    h = s.get("height") or viewport.h
    ox, oy = panel.offset
    return LayoutBox(viewport.x + ox, viewport.y + oy, float(w), float(h))


def _walk(
    widget: Widget,
    spec: dict,
    boxes: dict,
    x: float,
    y: float,
) -> Widget | None:
    box = boxes[id(spec)]
    if not _contains(box, x, y):
        return None
    if getattr(widget, "_is_gui_interactive", False):
        # Recurse first — a child interactive hit wins over the parent.
        if isinstance(widget, Panel):
            for child, child_spec in zip(widget.children, spec["children"]):
                hit = _walk(child, child_spec, boxes, x, y)
                if hit is not None:
                    return hit
        return widget
    if isinstance(widget, Panel):
        for child, child_spec in zip(widget.children, spec["children"]):
            hit = _walk(child, child_spec, boxes, x, y)
            if hit is not None:
                return hit
    return None


def _contains(box: LayoutBox, x: float, y: float) -> bool:
    return box.x <= x < box.x + box.w and box.y <= y < box.y + box.h
```

Run tests — 4 PASS.

### Step 2.3 — Wire hit-test into the bridge

Edit `src/manifoldx/gui/bridge.py` — replace the `_on_pointer` stub with:

```python
    def _on_pointer(self, ev: Any, phase: str) -> None:
        """Route a pointer event through hit-test to widget handlers."""
        gui = self._engine.gui
        viewport = self._viewport()

        # Drag capture takes precedence over hit-test.
        if self._captured is not None and phase != "down":
            self._dispatch(self._captured, ev, phase)
            gui.pointer_over_gui = True
            if phase == "up":
                self._captured = None
            return

        from manifoldx.gui.hit_test import hit_test
        widget = hit_test(list(gui), ev.x, ev.y, viewport=viewport)
        if widget is None:
            return
        gui.pointer_over_gui = True
        self._dispatch(widget, ev, phase)
        if phase == "down" and getattr(widget, "_gui_captures_pointer", False):
            self._captured = widget

    def _dispatch(self, widget: Any, ev: Any, phase: str) -> None:
        """Default dispatch — defer to widget if it implements the hook."""
        hook = getattr(widget, f"_on_pointer_{phase}", None)
        if hook is not None:
            hook(ev, self._engine)

    def _viewport(self):
        eng = self._engine
        from manifoldx.gui.layout import LayoutBox
        return LayoutBox(0.0, 0.0, float(eng.w), float(eng.h))
```

### Step 2.4 — Bridge test for pointer_over_gui

Append to `tests/gui/test_bridge.py`:

```python
def test_pointer_event_over_widget_sets_pointer_over_gui_flag():
    """Drive the bridge with a real engine + offscreen canvas."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    from manifoldx.gui.widgets import Widget

    class _Marker(Widget):
        _is_gui_interactive = True
        def intrinsic_size(self):
            return (10.0, 10.0)

    engine = mx.Engine("test", width=128, height=128)
    engine._init_canvas(canvas)
    engine._running = True

    panel = Panel(
        children=[_Marker()],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 50, "height": 50, "padding": 0},
    )
    engine.gui.append(panel)

    # Synthetic pointer_down inside the panel.
    engine._render_canvas.submit_event({
        "event_type": "pointer_down",
        "x": 10.0, "y": 10.0,
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    # Pointer was over the panel during dispatch.
    assert engine.gui.pointer_over_gui is True
```

Note: events fire AFTER `_input_bridge.attach`, dispatch happens during `_process_events` followed by `_draw_frame`. Because `_draw_frame` calls `_gui_bridge.begin_frame()` at the head — BEFORE event handlers (events ride at "Step 3: drain pending events"), the flag will be True after the frame ends only if the dispatch order puts hit-test AFTER begin_frame's reset. Verify the order experimentally; if the flag ends up False, the fix is to dispatch events before the reset (move the gui begin_frame after dispatch) — flag at `tests/test_bridge.py` and document.

### Step 2.5 — Full suite + lint + commit

```bash
make test     # 512 passing expected (508 + 4)
make lint
git add src/manifoldx/gui/hit_test.py src/manifoldx/gui/bridge.py \
        tests/gui/test_hit_test.py tests/gui/test_bridge.py
git commit -m "feat(gui): hit-testing + pointer-event routing through bridge"
```

---

## Task 3: Button widget

**Files:**
- Create: `src/manifoldx/gui/button.py`
- Modify: `src/manifoldx/gui/painter.py` (add Button branch)
- Modify: `src/manifoldx/gui/__init__.py` (export Button)
- Create: `tests/gui/test_button.py`

### Step 3.1 — Failing tests

Create `tests/gui/test_button.py`:

```python
"""Tests for manifoldx.gui.button — click handling, was_clicked latch, events."""

import pytest

from manifoldx.gui import Button, style


def setup_function(_):
    style.reset()


def test_button_construction_requires_name_and_label():
    b = Button(name="reset", label="Reset")
    assert b.name == "reset"
    assert b.label == "Reset"
    assert b._is_gui_interactive is True


def test_button_was_clicked_latch_consumed_on_read():
    b = Button(name="x", label="X")
    assert b.was_clicked() is False
    b._on_pointer_down(_fake_pointer(0, 0), _fake_engine())
    assert b.was_clicked() is True
    # Latch consumed — second read returns False.
    assert b.was_clicked() is False


def test_button_emits_click_event_on_pointer_down():
    received = []
    eng = _fake_engine(emit_sink=received)
    b = Button(name="reset", label="Reset")
    b._on_pointer_down(_fake_pointer(0, 0), eng)
    assert received == [("gui:button:reset:click", {})]


def test_button_does_not_capture_pointer():
    b = Button(name="x", label="X")
    assert getattr(b, "_gui_captures_pointer", False) is False


def test_button_intrinsic_size_grows_with_label_length():
    short = Button(name="x", label="X")
    long_ = Button(name="x", label="XXXXXXXXX")
    sw, _ = short.intrinsic_size()
    lw, _ = long_.intrinsic_size()
    assert lw > sw


def _fake_pointer(x, y):
    from manifoldx.input import PointerEvent
    return PointerEvent(x=x, y=y, dx=0, dy=0, button=1, buttons=(1,), modifiers=(), phase="down")


def _fake_engine(emit_sink=None):
    sink = emit_sink if emit_sink is not None else []
    class _E:
        def emit(self, name, payload):
            sink.append((name, payload))
    return _E()
```

Run: 5 FAIL — no module.

### Step 3.2 — Implement `button.py`

Create `src/manifoldx/gui/button.py`:

```python
"""Button widget — emits `gui:button:<name>:click` on pointer_down."""

from __future__ import annotations

from typing import Any

from manifoldx.gui.widgets import Widget, _measure_text


class Button(Widget):
    _is_gui_interactive = True
    _gui_captures_pointer = False

    def __init__(
        self,
        *,
        name: str,
        label: str,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if not isinstance(name, str) or not name:
            raise ValueError("Button.name is required and must be a non-empty string")
        self.name = name
        self.label = label
        self._click_latch: bool = False

    def intrinsic_size(self) -> tuple[float, float]:
        font_size = int(self.effective_style().get("font_size", 12))
        w, h = _measure_text(self.label, font_size)
        # Add 12px horizontal padding around the label, 4px vertical.
        return (w + 24.0, h + 8.0)

    def was_clicked(self) -> bool:
        """Return True if the button has been clicked since the previous read.
        Latch consumed on read."""
        v = self._click_latch
        self._click_latch = False
        return v

    def _on_pointer_down(self, ev: Any, engine: Any) -> None:
        self._click_latch = True
        engine.emit(f"gui:button:{self.name}:click", {})
```

Run tests — 5 PASS.

### Step 3.3 — Painter branch for Button

In `src/manifoldx/gui/painter.py`, update the imports:

```python
from manifoldx.gui.button import Button
```

In `paint()`, add a branch after `ValueDisplay`:

```python
    elif isinstance(widget, Button):
        s = widget.effective_style()
        # Rect background + 1px border + label centered (no per-glyph centering;
        # the painter just emits one text op for the full label).
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
```

Add the import for `_measure_text` at the top of painter.py (alongside other widget imports):

```python
from manifoldx.gui.widgets import Panel, Text, Widget, _measure_text
```

### Step 3.4 — Export Button

In `src/manifoldx/gui/__init__.py`:

```python
"""ManifoldX in-engine GUI layer.

Public API:
- `Panel`, `Text`, `ValueDisplay` — non-interactive widgets.
- `Button`, `Slider`, `Toggle` — interactive widgets.
- `style` — theme + named classes + per-widget overrides.
"""

from manifoldx.gui import style  # noqa: F401
from manifoldx.gui.button import Button  # noqa: F401
from manifoldx.gui.slider import Slider  # noqa: F401
from manifoldx.gui.toggle import Toggle  # noqa: F401
from manifoldx.gui.value_display import ValueDisplay  # noqa: F401
from manifoldx.gui.widgets import Panel, Text  # noqa: F401

__all__ = ["Button", "Panel", "Slider", "Text", "Toggle", "ValueDisplay", "style"]
```

> **Note for the implementer:** This export block depends on `Slider` and `Toggle` modules existing. If you're implementing in task order, this commit will fail import until Tasks 4 and 5 land. Workaround: in Task 3's commit, export ONLY `Button` for now (drop the Slider/Toggle imports + `__all__` entries); add them back in Tasks 4 and 5's commits.

### Step 3.5 — Run + lint + commit

```bash
uv run pytest tests/gui/test_button.py tests/gui/test_painter.py tests/gui/test_widgets.py -v
make test
make lint
git add src/manifoldx/gui/button.py src/manifoldx/gui/painter.py \
        src/manifoldx/gui/__init__.py tests/gui/test_button.py
git commit -m "feat(gui): Button widget — click event + was_clicked latch"
```

---

## Task 4: Toggle widget

**Files:**
- Create: `src/manifoldx/gui/toggle.py`
- Modify: `src/manifoldx/gui/painter.py`
- Modify: `src/manifoldx/gui/__init__.py` (now add Toggle export)
- Create: `tests/gui/test_toggle.py`

### Step 4.1 — Failing tests

Create `tests/gui/test_toggle.py`:

```python
"""Tests for manifoldx.gui.toggle — click flips value + emits change."""

from manifoldx.gui import Toggle, style


def setup_function(_):
    style.reset()


def test_toggle_construction_requires_name_value_label():
    t = Toggle(name="trails", value=False, label="Trails")
    assert t.name == "trails"
    assert t.value is False
    assert t.label == "Trails"
    assert t._is_gui_interactive is True


def test_toggle_does_not_capture_pointer():
    t = Toggle(name="x", value=False, label="X")
    assert getattr(t, "_gui_captures_pointer", False) is False


def test_toggle_pointer_down_flips_value_and_emits_change():
    received = []
    eng = _fake_engine(emit_sink=received)
    t = Toggle(name="trails", value=False, label="Trails")
    t._on_pointer_down(_fake_pointer(0, 0), eng)
    assert t.value is True
    assert received == [("gui:toggle:trails:change", {"value": True})]
    # Click again — flips back.
    t._on_pointer_down(_fake_pointer(0, 0), eng)
    assert t.value is False
    assert received[-1] == ("gui:toggle:trails:change", {"value": False})


def _fake_pointer(x, y):
    from manifoldx.input import PointerEvent
    return PointerEvent(x=x, y=y, dx=0, dy=0, button=1, buttons=(1,),
                        modifiers=(), phase="down")


def _fake_engine(emit_sink):
    class _E:
        def emit(self, name, payload):
            emit_sink.append((name, payload))
    return _E()
```

Run: 3 FAIL.

### Step 4.2 — Implement `toggle.py`

Create `src/manifoldx/gui/toggle.py`:

```python
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
        # ~14px checkbox + 6px gap + label width.
        font_size = int(self.effective_style().get("font_size", 12))
        lw, lh = _measure_text(self.label, font_size)
        return (14.0 + 6.0 + lw, max(14.0, lh))

    def _on_pointer_down(self, ev: Any, engine: Any) -> None:
        self.value = not self.value
        engine.emit(f"gui:toggle:{self.name}:change", {"value": self.value})
```

Run tests — 3 PASS.

### Step 4.3 — Painter branch for Toggle

Add to `painter.py`'s imports:

```python
from manifoldx.gui.toggle import Toggle
```

Add a branch in `paint()` after Button:

```python
    elif isinstance(widget, Toggle):
        s = widget.effective_style()
        # Checkbox square (14x14) on the left; label text to the right.
        cb_size = 14.0
        cb_box = LayoutBox(box.x, box.y + (box.h - cb_size) * 0.5, cb_size, cb_size)
        # Outer border.
        painter.draw_rect(
            box=cb_box,
            fill=parse_color(s["bg"]),
            border_color=parse_color(s["fg"]),
            border=1.0,
            radius=2.0,
        )
        # Inner fill when checked (smaller inset).
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
```

### Step 4.4 — Export + commit

Update `__init__.py` to include `Toggle` (per the note in Task 3.4 — Toggle is now real, so add the import back).

```bash
make test
make lint
git add src/manifoldx/gui/toggle.py src/manifoldx/gui/painter.py \
        src/manifoldx/gui/__init__.py tests/gui/test_toggle.py
git commit -m "feat(gui): Toggle widget — flips value on click + emits change"
```

---

## Task 5: Slider widget + drag capture

**Files:**
- Create: `src/manifoldx/gui/slider.py`
- Modify: `src/manifoldx/gui/painter.py`
- Modify: `src/manifoldx/gui/__init__.py` (add Slider export)
- Create: `tests/gui/test_slider.py`

### Step 5.1 — Failing tests

Create `tests/gui/test_slider.py`:

```python
"""Tests for manifoldx.gui.slider — drag capture + change/commit events."""

from manifoldx.gui import Slider, style
from manifoldx.gui.layout import LayoutBox


def setup_function(_):
    style.reset()


def test_slider_construction_validates_args():
    s = Slider(name="G", min=0.1, max=10.0, value=1.0, label="G")
    assert s.name == "G"
    assert s.min == 0.1
    assert s.max == 10.0
    assert s.value == 1.0
    assert s._is_gui_interactive is True
    assert s._gui_captures_pointer is True


def test_slider_pointer_down_sets_value_proportional_to_x_in_box():
    received = []
    eng = _fake_engine(received)
    s = Slider(name="G", min=0.0, max=100.0, value=0.0, label="G")
    # Simulate having a layout box assigned (set externally by bridge before dispatch).
    s._layout_box = LayoutBox(10.0, 20.0, 200.0, 16.0)
    # Click at the midpoint → value should be ~50.
    ev = _fake_pointer(110.0, 28.0)
    s._on_pointer_down(ev, eng)
    assert 49.0 < s.value < 51.0
    assert any(name == "gui:slider:G:change" for name, _ in received)


def test_slider_pointer_move_updates_value_and_emits_change():
    received = []
    eng = _fake_engine(received)
    s = Slider(name="G", min=0.0, max=10.0, value=0.0, label="G")
    s._layout_box = LayoutBox(0.0, 0.0, 100.0, 16.0)
    s._on_pointer_down(_fake_pointer(0.0, 8.0), eng)
    received.clear()
    s._on_pointer_move(_fake_pointer(50.0, 8.0), eng)
    assert s.value == 5.0
    assert received == [("gui:slider:G:change", {"value": 5.0})]


def test_slider_pointer_up_emits_commit():
    received = []
    eng = _fake_engine(received)
    s = Slider(name="G", min=0.0, max=10.0, value=0.0, label="G")
    s._layout_box = LayoutBox(0.0, 0.0, 100.0, 16.0)
    s._on_pointer_down(_fake_pointer(50.0, 8.0), eng)
    received.clear()
    s._on_pointer_up(_fake_pointer(50.0, 8.0), eng)
    assert ("gui:slider:G:commit", {"value": 5.0}) in received


def test_slider_value_clamped_outside_box():
    s = Slider(name="G", min=0.0, max=10.0, value=0.0, label="G")
    s._layout_box = LayoutBox(0.0, 0.0, 100.0, 16.0)
    # Drag far to the right.
    s._on_pointer_down(_fake_pointer(500.0, 8.0), _fake_engine([]))
    assert s.value == 10.0
    # And far to the left.
    s._on_pointer_move(_fake_pointer(-50.0, 8.0), _fake_engine([]))
    assert s.value == 0.0


def _fake_pointer(x, y):
    from manifoldx.input import PointerEvent
    return PointerEvent(x=x, y=y, dx=0, dy=0, button=1, buttons=(1,),
                        modifiers=(), phase="down")


def _fake_engine(sink):
    class _E:
        def emit(self, name, payload):
            sink.append((name, payload))
    return _E()
```

Run: 5 FAIL.

### Step 5.2 — Implement `slider.py`

Create `src/manifoldx/gui/slider.py`:

```python
"""Slider widget — horizontal drag updates value, emits change + commit.

Bridge contract: the bridge must set `slider._layout_box` to the slider's
current pixel-space LayoutBox BEFORE dispatching pointer events. This
indirection avoids forcing the slider to recompute its own layout from
scratch on every event. The painter does this for free at paint time, but
the bridge owns event dispatch, so we let it stash the box during hit-test.
"""

from __future__ import annotations

from typing import Any

from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.widgets import Widget, _measure_text


class Slider(Widget):
    _is_gui_interactive = True
    _gui_captures_pointer = True

    def __init__(
        self,
        *,
        name: str,
        min: float,
        max: float,
        value: float,
        label: str | None = None,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if not isinstance(name, str) or not name:
            raise ValueError("Slider.name is required and must be a non-empty string")
        if max <= min:
            raise ValueError(f"Slider max ({max}) must be > min ({min})")
        self.name = name
        self.min = float(min)
        self.max = float(max)
        self.value = float(value)
        self.label = label
        # Bridge stashes the current pixel-space box here before dispatching events.
        self._layout_box: LayoutBox | None = None

    def intrinsic_size(self) -> tuple[float, float]:
        # Default 160px wide × 16px tall.
        return (160.0, 16.0)

    def _update_from_x(self, x: float) -> None:
        if self._layout_box is None:
            return
        box = self._layout_box
        if box.w <= 0:
            return
        t = (x - box.x) / box.w
        t = max(0.0, min(1.0, t))
        self.value = self.min + t * (self.max - self.min)

    def _on_pointer_down(self, ev: Any, engine: Any) -> None:
        self._update_from_x(ev.x)
        engine.emit(f"gui:slider:{self.name}:change", {"value": self.value})

    def _on_pointer_move(self, ev: Any, engine: Any) -> None:
        self._update_from_x(ev.x)
        engine.emit(f"gui:slider:{self.name}:change", {"value": self.value})

    def _on_pointer_up(self, ev: Any, engine: Any) -> None:
        self._update_from_x(ev.x)
        engine.emit(f"gui:slider:{self.name}:commit", {"value": self.value})
```

Run tests — 5 PASS.

### Step 5.3 — Bridge must populate `slider._layout_box` before dispatch

This needs a small extension in `bridge.py`'s `_on_pointer`. After hit_test returns a widget, if it's a `Slider`, look up its layout box from the hit_test pass's box map. The cleanest way is to have `hit_test` return both the widget AND the box, or have the slider re-do layout — but per the design we want minimal recomputation.

Pragmatic approach: refactor `hit_test` to return `(widget, box)` tuple. Update `bridge.py` to unpack and set `widget._layout_box = box` before dispatch. Update `tests/gui/test_hit_test.py` accordingly.

Edit `hit_test.py` to return `tuple[Widget, LayoutBox] | None`:

```python
def hit_test(...) -> tuple[Widget, LayoutBox] | None:
    ...
    for panel in reversed(panels):
        ...
        hit = _walk(panel, spec, boxes, x, y)
        if hit is not None:
            return hit
    return None


def _walk(widget, spec, boxes, x, y) -> tuple[Widget, LayoutBox] | None:
    box = boxes[id(spec)]
    if not _contains(box, x, y):
        return None
    if getattr(widget, "_is_gui_interactive", False):
        if isinstance(widget, Panel):
            for child, child_spec in zip(widget.children, spec["children"]):
                hit = _walk(child, child_spec, boxes, x, y)
                if hit is not None:
                    return hit
        return (widget, box)
    if isinstance(widget, Panel):
        for child, child_spec in zip(widget.children, spec["children"]):
            hit = _walk(child, child_spec, boxes, x, y)
            if hit is not None:
                return hit
    return None
```

Update `test_hit_test.py` — the existing tests destructure `hit` as a tuple now. For example:

```python
def test_hit_test_later_panel_topmost():
    ...
    hit = hit_test([p1, p2], 10.0, 10.0, viewport=LayoutBox(0, 0, 256, 256))
    assert hit is not None
    widget, box = hit
    assert widget is b
```

And `test_hit_test_returns_none_when_*` tests stay as-is (they assert `is None`).

Update `bridge.py`'s `_on_pointer` to unpack and stash:

```python
        from manifoldx.gui.hit_test import hit_test
        result = hit_test(list(gui), ev.x, ev.y, viewport=viewport)
        if result is None:
            return
        widget, box = result
        gui.pointer_over_gui = True
        # Slider needs its layout box on hand before dispatch.
        widget._layout_box = box  # noqa: SLF001 — internal contract
        self._dispatch(widget, ev, phase)
        if phase == "down" and getattr(widget, "_gui_captures_pointer", False):
            self._captured = widget
```

Also adjust the captured-drag branch to ALSO refresh the captured widget's box on each subsequent event (in case the panel moved — usually it doesn't in v1, but the box must still be set since we don't re-hit-test). Simplest: keep the last-known box during capture:

```python
        if self._captured is not None and phase != "down":
            # Re-anchor the slider's box from the current panel layout.
            # For v1 the box doesn't change between events, but we still
            # need _layout_box set (it was set during the down event).
            self._dispatch(self._captured, ev, phase)
            gui.pointer_over_gui = True
            if phase == "up":
                self._captured = None
            return
```

### Step 5.4 — Painter branch for Slider

In `painter.py`, import:

```python
from manifoldx.gui.slider import Slider
```

Add a branch:

```python
    elif isinstance(widget, Slider):
        s = widget.effective_style()
        # Background track.
        painter.draw_rect(
            box=box,
            fill=parse_color(s["bg"]),
            border_color=parse_color(s["border_color"]),
            border=float(s["border"]),
            radius=float(s["radius"]),
        )
        # Fill portion proportional to value.
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
```

### Step 5.5 — Run + commit

```bash
make test
make lint
git add src/manifoldx/gui/slider.py src/manifoldx/gui/hit_test.py \
        src/manifoldx/gui/bridge.py src/manifoldx/gui/painter.py \
        src/manifoldx/gui/__init__.py tests/gui/test_slider.py \
        tests/gui/test_hit_test.py
git commit -m "feat(gui): Slider widget — drag capture, change + commit events"
```

---

## Task 6: End-to-end interaction test

**Files:**
- Create: `tests/gui/test_interaction_e2e.py`

The pure-CPU tests in Tasks 3–5 exercise widget state directly. This task adds an e2e GPU-gated test that pushes synthetic pointer events through `canvas.submit_event` (the existing test affordance) and asserts user-registered `@engine.on("gui:slider:G:change")` handlers fire with correct payloads.

### Step 6.1 — Write the test

Create `tests/gui/test_interaction_e2e.py`:

```python
"""End-to-end GUI interaction tests via synthetic pointer events."""

import pytest

from manifoldx.gui import Button, Panel, Slider, Toggle, style


def _make_offscreen_engine(width: int = 256, height: int = 128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine, canvas


def setup_function(_):
    style.reset()


def _click(engine, x, y):
    engine._render_canvas.submit_event({
        "event_type": "pointer_down",
        "x": float(x), "y": float(y),
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_up",
        "x": float(x), "y": float(y),
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()


def test_button_click_fires_handler():
    style.set_theme({"padding": 0, "gap": 0})
    engine, _ = _make_offscreen_engine()
    received = []

    @engine.on("gui:button:reset:click")
    def _(payload):
        received.append(payload)

    panel = Panel(
        children=[Button(name="reset", label="Reset")],
        anchor="top-left", offset=(0, 0),
        style_overrides={"width": 100, "height": 30, "padding": 0},
    )
    engine.gui.append(panel)
    _click(engine, 50, 15)
    assert received == [{}]


def test_toggle_click_flips_and_emits():
    style.set_theme({"padding": 0, "gap": 0})
    engine, _ = _make_offscreen_engine()
    received = []

    @engine.on("gui:toggle:trails:change")
    def _(payload):
        received.append(payload)

    tg = Toggle(name="trails", value=False, label="Trails")
    panel = Panel(
        children=[tg],
        anchor="top-left", offset=(0, 0),
        style_overrides={"width": 100, "height": 30, "padding": 0},
    )
    engine.gui.append(panel)
    _click(engine, 10, 15)
    assert received == [{"value": True}]
    assert tg.value is True


def test_slider_drag_emits_change_and_commit():
    style.set_theme({"padding": 0, "gap": 0})
    engine, _ = _make_offscreen_engine(width=200, height=64)
    changes, commits = [], []

    @engine.on("gui:slider:G:change")
    def _on_change(payload):
        changes.append(payload["value"])

    @engine.on("gui:slider:G:commit")
    def _on_commit(payload):
        commits.append(payload["value"])

    slider = Slider(name="G", min=0.0, max=100.0, value=0.0, label="G")
    panel = Panel(
        children=[slider],
        anchor="top-left", offset=(0, 0),
        style_overrides={"width": 200, "height": 30, "padding": 0},
    )
    engine.gui.append(panel)

    # pointer_down at x=20 (10% across the 200px slider) → value=10
    engine._render_canvas.submit_event({
        "event_type": "pointer_down", "x": 20.0, "y": 15.0,
        "button": 1, "buttons": (1,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_move", "x": 100.0, "y": 15.0,
        "button": 0, "buttons": (1,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas.submit_event({
        "event_type": "pointer_up", "x": 100.0, "y": 15.0,
        "button": 1, "buttons": (0,), "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()

    assert len(changes) >= 2  # at minimum down + move emitted change
    assert commits == [50.0]  # up emits commit at x=100 → 50% of 100
    assert slider.value == 50.0
```

Run: tests should all PASS (or surface a bug in routing — likely culprit: dispatch ordering of synthetic events relative to begin_frame).

### Step 6.2 — Commit

```bash
make test
make lint
git add tests/gui/test_interaction_e2e.py
git commit -m "test(gui): e2e interaction — Button/Toggle/Slider via synthetic pointers"
```

---

## Task 7: `examples/gui_demo.py`

**Files:**
- Create: `examples/gui_demo.py`

### Step 7.1 — Author the example

Create `examples/gui_demo.py`:

```python
"""N-body simulation with a docked GUI control panel.

Sliders: G (gravitational constant), dt (timestep).
Toggle: trails (no-op stub in this example).
Button: reset (re-randomizes positions).
ValueDisplay: live FPS readout.
"""

from __future__ import annotations

import numpy as np

import manifoldx as mx
from manifoldx import gui
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, sphere

N = 100
G = 1.0
DT = 0.01
SOFTENING = 0.5


def main() -> None:
    engine = mx.Engine("gui demo — n-body with controls", width=960, height=640)

    sphere_geo = sphere(0.2, 16)
    mat = StandardMaterial(color="#88ccff", roughness=0.4, metallic=0.1)

    rng = np.random.default_rng(42)
    positions = rng.uniform(-3, 3, size=(N, 3)).astype(np.float32)
    velocities = np.zeros((N, 3), dtype=np.float32)
    masses = rng.uniform(0.5, 2.0, size=N).astype(np.float32)

    entities = [
        engine.spawn(Mesh(sphere_geo), Material(mat), Transform(pos=tuple(p)))
        for p in positions
    ]

    state = {"G": G, "dt": DT}

    @engine.system
    def nbody(query: mx.Query[Transform], dt: float):
        nonlocal velocities, positions
        diff = positions[None, :] - positions[:, None]
        dist = np.linalg.norm(diff, axis=2)
        dist = np.maximum(dist, SOFTENING)
        force_mag = state["G"] * (masses[None, :] * masses[:, None]) / dist ** 2
        direction = diff / dist[:, :, None]
        net_force = (force_mag[:, :, None] * direction).sum(axis=1)
        velocities += (net_force / masses[:, None]) * state["dt"]
        positions += velocities * state["dt"]
        query[Transform].pos = positions

    # --- GUI panel ---
    panel = gui.Panel(
        anchor="top-right",
        offset=(20, 20),
        style_overrides={"width": 240, "padding": 8, "gap": 6,
                         "bg": "#1a1a1aE0", "radius": 4},
        children=[
            gui.Text("N-Body Controls", style_overrides={"font_size": 14}),
            gui.Slider(name="G", min=0.1, max=10.0, value=G, label="G"),
            gui.Slider(name="dt", min=0.001, max=0.1, value=DT, label="dt"),
            gui.Toggle(name="trails", value=False, label="Trails (stub)"),
            gui.Button(name="reset", label="Reset"),
            gui.ValueDisplay(getter=lambda: f"N={N}  fps={engine.fps:.0f}",
                             min_width=160),
        ],
    )
    engine.gui.append(panel)

    @engine.on("gui:slider:G:change")
    def _(p):
        state["G"] = p["value"]

    @engine.on("gui:slider:dt:change")
    def _(p):
        state["dt"] = p["value"]

    @engine.on("gui:button:reset:click")
    def _(p):
        nonlocal positions, velocities
        positions[:] = rng.uniform(-3, 3, size=(N, 3)).astype(np.float32)
        velocities[:] = 0

    engine.camera.fit(radius=6.0, azimuth=30, elevation=25)
    engine.cli()


if __name__ == "__main__":
    main()
```

### Step 7.2 — Smoke check (render 1 frame)

`uv run python examples/gui_demo.py --render --duration 1 --fps 10 --output /tmp/gui_demo.mp4`

Expect: 10 frames render, mp4 file written. The control panel should be visible in the top-right corner. If `--render` mode is fully wired in the engine's CLI, this will produce output.

Don't add a pytest for this — examples are integration smoke-tests, validated by hand.

### Step 7.3 — Commit

```bash
git add examples/gui_demo.py
git commit -m "feat(examples): gui_demo.py — N-body with docked GUI control panel"
```

---

## Task 8: CHANGELOG + AGENTS.md closeout (Plan 2 / GUI v1 final)

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`

### Step 8.1 — CHANGELOG

In `CHANGELOG.md`'s `[Unreleased]` → `### Features`, REPLACE the "GUI v1 — foundation (in progress)" bullet from Plan 1 Task 9 with this final entry:

```markdown
- **GUI v1** — in-engine retained-mode GUI for sim controls + HUD readouts. New `manifoldx.gui` package: `Panel`, `Text`, `ValueDisplay`, `Button`, `Slider`, `Toggle`; stack + flex layout; theme + named-class styling with per-widget overrides. Rendering via a new `gui` render pass at the end of the order, using `RectMaterial` (WGSL signed-distance rounded rect) for backgrounds and the existing `LabelTextureAtlas` for glyphs (no new font dependency). Interaction via `_GuiBridge` — subscribes to pointer events on the bus, hit-tests the widget tree top-down, dispatches per-widget state mutations, and exposes `engine.gui.pointer_over_gui` for cooperative consume by user systems. Slider supports drag capture and emits `:change` continuously plus `:commit` on pointer-up. Example at `examples/gui_demo.py`.
```

### Step 8.2 — AGENTS.md

Find the `## Sub-projects in flight` section. Replace the existing "GUI v1 — Plan 1 (foundation)" bullet with:

```markdown
- **GUI v1** — landed. `manifoldx.gui` with widgets (`Panel`/`Text`/`ValueDisplay`/`Button`/`Slider`/`Toggle`), stack + flex layout, theme + named-class styling, `RectMaterial` (WGSL signed-distance rounded rect), `LabelTextureAtlas`-backed glyph path, `_GuiBridge` (pointer routing + hit-test + drag capture), `engine.gui.pointer_over_gui` cooperative-consume flag, and a new `gui` render pass at the end of the render order. Demo at `examples/gui_demo.py`. Plans at `.knowledge/plans/2026-05-11-gui-v1-plan-*.md`. Design at `.knowledge/analysis/2026-05-10-gui-v1-design.md`.
```

### Step 8.3 — Final full suite + lint

`make test` — all green. Expected ~520+ passing.
`make lint` — clean for all gui/ files and tests/gui/.

### Step 8.4 — Commit

```bash
git add CHANGELOG.md AGENTS.md
git commit -m "docs(gui): GUI v1 fully landed — CHANGELOG + AGENTS.md closeout"
```

---

## Self-review checklist

- [ ] **Spec coverage:** Bridge ✓ (Task 1+2), hit-test ✓ (Task 2), Button ✓ (Task 3), Toggle ✓ (Task 4), Slider w/ drag ✓ (Task 5), pointer_over_gui flag ✓ (Tasks 1+2), `:change`/`:commit` events ✓ (Tasks 4+5), drag capture ✓ (Task 5), e2e w/ synthetic events ✓ (Task 6), example ✓ (Task 7), CHANGELOG ✓ (Task 8).
- [ ] **Placeholders:** none.
- [ ] **Type consistency:** `_is_gui_interactive` and `_gui_captures_pointer` attributes used consistently in widgets, hit_test, bridge. `hit_test` return type `tuple[Widget, LayoutBox] | None` consistent across callers.
- [ ] **Public API:** `gui.Panel`, `gui.Text`, `gui.ValueDisplay`, `gui.Button`, `gui.Slider`, `gui.Toggle`, `gui.style` — matches design's "Public API → Widgets".
- [ ] **Event names:** `gui:button:<name>:click`, `gui:toggle:<name>:change`, `gui:slider:<name>:change`, `gui:slider:<name>:commit` — match design verbatim.

---

## Execution Handoff

**Plan complete and saved to `.knowledge/plans/2026-05-11-gui-v1-plan-2-interaction.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — controller dispatches a fresh subagent per task, reviews between tasks. Best fit because Tasks 5/6 have real engineering judgment (hit-test refactor, e2e dispatch ordering) and benefit from clean per-task context.

**2. Inline Execution** — via `superpowers:executing-plans`. Lower latency but risks context drift across the 8-task arc.

**Which approach?**
