"""_GlfwGamepadPoller tests against a fake `glfw` module (no device, no
window). The fake mirrors the real package's return shapes: names and
GUIDs are bytes, raw buttons/axes/hats come back as `(sequence, count)`,
and `get_gamepad_state` returns an object with `.buttons` / `.axes`."""
from types import SimpleNamespace

import pytest

from manifoldx.input import GamepadInfo, _GlfwGamepadPoller


class _FakeGlfw:
    JOYSTICK_LAST = 15
    HAT_UP, HAT_RIGHT, HAT_DOWN, HAT_LEFT = 1, 2, 4, 8

    def __init__(self) -> None:
        # jid -> dict(name, guid, gamepad_state | None, buttons, axes, hats)
        self.pads: dict[int, dict] = {}
        self.mappings: list[str] = []

    def add_mapped(self, jid, buttons=(0,) * 15, axes=(0.0, 0.0, 0.0, 0.0, -1.0, -1.0)):
        self.pads[jid] = dict(
            name=b"Raw Name", gamepad_name="Mapped Pad", guid=b"0300abcd",
            gamepad_state=SimpleNamespace(buttons=list(buttons), axes=list(axes)),
            buttons=[], axes=[], hats=[],
        )

    def add_raw(self, jid, buttons=(), axes=(), hats=()):
        self.pads[jid] = dict(
            name=b"Generic USB Joystick", gamepad_name=None, guid=b"03007900",
            gamepad_state=None,
            buttons=list(buttons), axes=list(axes), hats=list(hats),
        )

    def joystick_present(self, jid):
        return int(jid in self.pads)

    def joystick_is_gamepad(self, jid):
        return self.pads[jid]["gamepad_state"] is not None

    def get_gamepad_state(self, jid):
        return self.pads[jid]["gamepad_state"]

    def get_gamepad_name(self, jid):
        return self.pads[jid]["gamepad_name"]

    def get_joystick_name(self, jid):
        return self.pads[jid]["name"]

    def get_joystick_guid(self, jid):
        return self.pads[jid]["guid"]

    def get_joystick_buttons(self, jid):
        b = self.pads[jid]["buttons"]
        return b, len(b)

    def get_joystick_axes(self, jid):
        a = self.pads[jid]["axes"]
        return a, len(a)

    def get_joystick_hats(self, jid):
        h = self.pads[jid]["hats"]
        return h, len(h)

    def update_gamepad_mappings(self, string):
        self.mappings.append(string)
        return 1


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.delenv("MANIFOLDX_GAMEPAD_MAPPINGS", raising=False)
    return _FakeGlfw()


def test_no_joysticks_polls_empty(fake):
    assert _GlfwGamepadPoller(fake).poll() == ()


def test_mapped_pad_reads_buttons_and_axes_by_name(fake):
    buttons = [0] * 15
    buttons[0] = 1   # a
    buttons[12] = 1  # dpad_right (GLFW order: up, right, down, left)
    fake.add_mapped(3, buttons=buttons, axes=(0.5, -0.25, 0.0, 0.0, -1.0, 1.0))
    (snap,) = _GlfwGamepadPoller(fake).poll()
    assert snap.info == GamepadInfo(index=3, name="Mapped Pad", guid="0300abcd", mapped=True)
    assert snap.buttons == frozenset({"a", "dpad_right"})
    assert snap.axes["left_x"] == 0.5
    assert snap.axes["left_y"] == -0.25
    # Triggers are remapped from GLFW's [-1, 1] to [0, 1]: rest reads 0.
    assert snap.axes["lt"] == 0.0
    assert snap.axes["rt"] == 1.0


def test_unmapped_pad_falls_back_to_raw(fake):
    # Generic layout: 0=y, 1=b, 2=a, 3=x. Hat 0 down+left.
    fake.add_raw(0, buttons=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 axes=[0.0, 0.75, 0.3], hats=[4 | 8])
    (snap,) = _GlfwGamepadPoller(fake).poll()
    assert snap.info == GamepadInfo(index=0, name="Generic USB Joystick",
                                    guid="03007900", mapped=False)
    assert snap.buttons == frozenset(
        {"a", "button2", "button12", "dpad_down", "dpad_left"}
    )
    assert snap.axes["left_x"] == 0.0
    assert snap.axes["left_y"] == 0.75
    assert snap.axes["axis2"] == 0.3


def test_unmapped_pad_without_hat_has_no_dpad(fake):
    fake.add_raw(1, buttons=[1, 0, 0, 1], axes=[])
    (snap,) = _GlfwGamepadPoller(fake).poll()
    assert snap.buttons == frozenset({"y", "x", "button0", "button3"})
    assert snap.axes == {}


def test_env_var_mapping_string_is_passed_to_glfw(fake, monkeypatch):
    mapping = "03000000790000000600000010010000,Pad,a:b2,platform:Linux,"
    monkeypatch.setenv("MANIFOLDX_GAMEPAD_MAPPINGS", mapping)
    _GlfwGamepadPoller(fake)
    assert fake.mappings == [mapping]


def test_env_var_mapping_file_is_read_and_passed(fake, monkeypatch, tmp_path):
    db = tmp_path / "gamecontrollerdb.txt"
    db.write_text("# comment\n0300,Pad,a:b2,platform:Linux,\n")
    monkeypatch.setenv("MANIFOLDX_GAMEPAD_MAPPINGS", str(db))
    _GlfwGamepadPoller(fake)
    assert fake.mappings == [db.read_text()]


def test_no_env_var_means_no_mapping_update(fake):
    _GlfwGamepadPoller(fake)
    assert fake.mappings == []
