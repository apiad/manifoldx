"""Tests for manifoldx.gui.bridge — pointer routing skeleton."""

import pytest

from manifoldx.gui import Panel, style
from manifoldx.gui.bridge import _GuiBridge


def setup_function(_):
    style.reset()


def test_bridge_constructs_with_engine_and_subscribes_to_pointer_events():
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
    assert "pointer_down" in eng._handlers
    assert "pointer_move" in eng._handlers
    assert "pointer_up" in eng._handlers


def test_bridge_begin_frame_clears_pointer_over_gui_flag():
    class _G:
        pointer_over_gui = True
        def __iter__(self):
            return iter([])
        def __len__(self):
            return 0

    class _Stub:
        def __init__(self):
            self.gui = _G()
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
