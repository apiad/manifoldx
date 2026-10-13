"""Teardown for one-engine, many-screens apps: remove_system, off, clear_scene."""
import numpy as np
import pytest

import manifoldx as mx


def _engine():
    return mx.Engine("teardown")


# --- remove_system --------------------------------------------------------------


def test_remove_system_stops_it_running():
    engine = _engine()
    calls = []

    @engine.system
    def counter(query: mx.Query[mx.Transform], dt: float):
        calls.append(dt)

    engine.systems.run_all(engine, 0.1)
    engine.remove_system(counter)
    engine.systems.run_all(engine, 0.1)
    assert len(calls) == 1


def test_remove_system_from_inside_itself_keeps_the_rest_running():
    engine = _engine()
    seen = []

    @engine.system
    def once(query: mx.Query[mx.Transform], dt: float):
        seen.append("once")
        engine.remove_system(once)

    @engine.system
    def always(query: mx.Query[mx.Transform], dt: float):
        seen.append("always")

    engine.systems.run_all(engine, 0.1)
    engine.systems.run_all(engine, 0.1)
    assert seen == ["once", "always", "always"]


def test_removing_a_later_system_mid_frame_skips_it_this_frame():
    engine = _engine()
    seen = []

    @engine.system
    def first(query: mx.Query[mx.Transform], dt: float):
        seen.append("first")
        engine.remove_system(second)

    @engine.system
    def second(query: mx.Query[mx.Transform], dt: float):
        seen.append("second")

    engine.systems.run_all(engine, 0.1)
    assert seen == ["first"]


def test_remove_unknown_system_raises():
    engine = _engine()
    with pytest.raises(ValueError):
        engine.remove_system(lambda q, dt: None)


# --- off ------------------------------------------------------------------------


def test_off_stops_a_handler():
    engine = _engine()
    seen = []

    @engine.on("ping")
    def handler(payload):
        seen.append(payload)

    engine.emit("ping", 1)
    engine._event_bus.dispatch_pending(engine)
    engine.off("ping", handler)
    engine.emit("ping", 2)
    engine._event_bus.dispatch_pending(engine)
    assert seen == [1]


def test_off_during_dispatch_skips_later_handlers_and_keeps_others():
    engine = _engine()
    seen = []

    @engine.on("ping")
    def first(payload):
        seen.append("first")
        engine.off("ping", first)
        engine.off("ping", second)

    @engine.on("ping")
    def second(payload):
        seen.append("second")

    @engine.on("ping")
    def third(payload):
        seen.append("third")

    engine.emit("ping")
    engine.emit("ping")
    engine._event_bus.dispatch_pending(engine)
    assert seen == ["first", "third", "third"]


def test_off_unknown_handler_raises():
    engine = _engine()
    with pytest.raises(ValueError):
        engine.off("ping", lambda payload: None)
