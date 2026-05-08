"""EventBus: registration, queued dispatch, queue isolation between frames."""
from manifoldx.events import EventBus, Handler


class _DummyEngine:
    """Minimal stand-in for tests that don't need a real engine."""

    def __init__(self):
        self.store = None
        self.commands = None


def test_register_and_dispatch_sync_handler():
    bus = EventBus()
    seen = []

    @bus.on("ping")
    def handler(payload):
        seen.append(payload)

    bus.emit("ping", {"v": 1})
    assert seen == []  # not delivered yet
    bus.dispatch_pending(_DummyEngine())
    assert seen == [{"v": 1}]


def test_dispatch_order_is_registration_order():
    bus = EventBus()
    seen = []

    @bus.on("e")
    def first(payload):
        seen.append("first")

    @bus.on("e")
    def second(payload):
        seen.append("second")

    bus.emit("e", None)
    bus.dispatch_pending(_DummyEngine())
    assert seen == ["first", "second"]


def test_emits_during_dispatch_defer_to_next_frame():
    bus = EventBus()
    seen = []

    @bus.on("a")
    def a(payload):
        seen.append("a")
        bus.emit("b", None)

    @bus.on("b")
    def b(payload):
        seen.append("b")

    bus.emit("a", None)
    bus.dispatch_pending(_DummyEngine())
    assert seen == ["a"]  # b was queued, not delivered

    bus.dispatch_pending(_DummyEngine())
    assert seen == ["a", "b"]


def test_dispatch_immediate_runs_inline():
    bus = EventBus()
    seen = []

    @bus.on("phase")
    def h(payload):
        seen.append(payload)

    bus.dispatch_immediate(_DummyEngine(), "phase", {"k": 1})
    assert seen == [{"k": 1}]


def test_unregistered_event_dispatch_is_noop():
    bus = EventBus()
    bus.emit("nobody-listens", None)
    bus.dispatch_pending(_DummyEngine())  # must not raise
