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
