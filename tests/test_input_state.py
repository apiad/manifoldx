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
