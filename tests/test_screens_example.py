"""examples/screens.py, driven headless with synthetic Space presses."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "screens.py"


def _load_example():
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    spec = importlib.util.spec_from_file_location("screens_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = module.engine
    engine._init_canvas(canvas)
    engine._running = True
    engine._fire_startup()
    return module, engine


def _press_space(engine):
    for kind in ("key_down", "key_up"):
        engine._render_canvas.submit_event({"event_type": kind, "key": " ", "modifiers": ()})
    engine._render_canvas._process_events()


def _scene(module, engine):
    alive = np.where(engine.store._alive)[0]
    geometry = engine.store._components["Mesh"][alive, 0]
    cube = engine._geometry_registry.register(module.CUBE)
    sphere = engine._geometry_registry.register(module.SPHERE)
    texts = [w.text for panel in engine.gui for w in panel.children]
    return {
        "cubes": int(np.sum(geometry == cube)),
        "spheres": int(np.sum(geometry == sphere)),
        "texts": texts,
        "systems": len(engine.systems._systems),
    }


def test_space_swaps_screens_and_back_without_leftovers():
    module, engine = _load_example()
    try:
        for _ in range(3):
            engine._draw_frame()
        a = _scene(module, engine)
        assert a == {"cubes": 1, "spheres": 0, "texts": ["Screen A: press Space"], "systems": 1}

        _press_space(engine)
        engine._draw_frame()
        engine._draw_frame()
        b = _scene(module, engine)
        assert b["cubes"] == 0 and b["spheres"] == 1 and b["systems"] == 0
        assert b["texts"] == ["Screen B: press Space to go back"]

        _press_space(engine)
        engine._draw_frame()
        engine._draw_frame()
        assert _scene(module, engine) == a
    finally:
        engine.shutdown_events()
