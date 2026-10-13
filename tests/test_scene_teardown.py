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
