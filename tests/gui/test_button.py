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
    return PointerEvent(x=x, y=y, dx=0, dy=0, button=1, buttons=(1,),
                        modifiers=(), phase="down")


def _fake_engine(emit_sink=None):
    sink = emit_sink if emit_sink is not None else []
    class _E:
        def emit(self, name, payload):
            sink.append((name, payload))
    return _E()
