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
        """Stub — routing logic lands in Task 2 (hit-test) and Tasks 3–5
        (per-widget handling). For now the bridge silently swallows events."""
        return
