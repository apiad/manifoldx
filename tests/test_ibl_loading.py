"""The environment's precompute runs off the render loop: frames keep coming, with a
loading spinner, until it is ready; offline renders wait for it (manifoldx#25)."""

import threading

import pytest


def _engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube

    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    # A lit mesh: the environment is only uploaded when something lit is drawn.
    engine.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial(color="#ffffff")), Transform(pos=(0, 0, 0)))
    return engine


def _gated_env(rgb=(0.5, 0.5, 0.5)):
    """An environment whose precompute waits until the returned event is set, then fills
    maps of the right shapes: these tests are about the loading, not the IBL arithmetic."""
    import numpy as np

    from manifoldx.ibl import EnvironmentMap

    env = EnvironmentMap.from_color(rgb)
    gate = threading.Event()

    def gated():
        assert gate.wait(30), "the test never released the precompute"
        env._irradiance = np.ones((6, 64, 64, 4), np.float16)
        env._prefiltered = [np.ones((6, max(1, 128 >> m), max(1, 128 >> m), 4), np.float16) for m in range(8)]
        env._computed = True

    env._precompute = gated
    return env, gate


def test_set_environment_alone_starts_no_work():
    from manifoldx.engine import Engine
    from manifoldx.ibl import EnvironmentMap

    eng = Engine("test", max_entities=16)
    eng.set_environment(EnvironmentMap.from_color((0.5, 0.5, 0.5)))
    assert eng._env_task is None  # nothing runs until a frame needs it


def test_frames_keep_coming_while_the_environment_computes():
    engine = _engine()
    env, gate = _gated_env()
    engine.set_environment(env)
    engine._draw_frame()  # returns although the precompute is blocked
    engine._draw_frame()
    assert not env._computed
    assert engine._render_pipeline._ibl_env_id != id(env)  # drawn without it
    assert engine._loading_panel in list(engine.gui)  # and with the spinner up

    gate.set()
    engine._env_task.wait()
    engine._draw_frame()
    assert engine._loading_panel is None and len(engine.gui) == 0  # spinner gone
    assert engine._render_pipeline._ibl_env_id == id(env)  # and the environment in use


def test_the_spinner_says_what_is_loading():
    engine = _engine()
    env, gate = _gated_env()
    engine.set_environment(env)
    engine._draw_frame()
    first = engine._loading_line()
    assert "lighting" in first
    gate.set()
    engine._env_task.wait()


def test_render_frame_waits_for_the_environment():
    engine = _engine()
    env, gate = _gated_env()
    engine.set_environment(env)
    threading.Timer(0.2, gate.set).start()
    engine.render_frame(supersample=1)
    assert env._computed
    assert engine._loading_panel is None  # an offline frame never shows the spinner


def test_a_new_environment_replaces_the_pending_one():
    engine = _engine()
    env, gate = _gated_env()
    engine.set_environment(env)
    engine._draw_frame()
    other, other_gate = _gated_env((0.2, 0.2, 0.2))
    engine.set_environment(other)
    gate.set()
    other_gate.set()
    engine._settle_environment()
    engine._draw_frame()
    assert other._computed
    assert engine._render_pipeline._ibl_env_id == id(other)
