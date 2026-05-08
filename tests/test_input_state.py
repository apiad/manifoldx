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
