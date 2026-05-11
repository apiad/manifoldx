"""GUI input bridge — routes pointer events from the bus to widget state.

Subscribes to `pointer_down` / `pointer_move` / `pointer_up` via `engine.on(...)`.
The full routing (hit-test, drag capture, per-widget dispatch) lands in Plan 2
Task 2. Task 1 ships only the constructor + the begin_frame hook that resets
`engine.gui.pointer_over_gui` to False at the head of each frame.
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
        """Route a pointer event through hit-test to widget handlers."""
        gui = self._engine.gui

        # Drag capture takes precedence over hit-test.
        if self._captured is not None and phase != "down":
            self._dispatch(self._captured, ev, phase)
            gui.pointer_over_gui = True
            if phase == "up":
                self._captured = None
            return

        from manifoldx.gui.hit_test import hit_test
        from manifoldx.gui.layout import LayoutBox
        viewport = LayoutBox(0.0, 0.0, float(self._engine.w), float(self._engine.h))
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
