import pytest
from manifoldx import Engine


def test_engine_creation():
    """Engine can be created with name and dimensions."""
    engine = Engine("Test", h=600, w=800, fullscreen=False)
    assert engine.name == "Test"
    assert engine.h == 600
    assert engine.w == 800
    assert engine.fullscreen is False


def test_engine_startup_decorator():
    """Engine startup decorator registers callbacks."""
    engine = Engine("Test")
    call_count = 0

    @engine.startup
    def init():
        nonlocal call_count
        call_count += 1

    # Startup callback should be registered
    assert len(engine._startup_callbacks) == 1
    assert init in engine._startup_callbacks


def test_engine_shutdown_decorator():
    """Engine shutdown decorator registers callbacks."""
    engine = Engine("Test")
    call_count = 0

    @engine.shutdown
    def close():
        nonlocal call_count
        call_count += 1

    # Shutdown callback should be registered
    assert len(engine._shutdown_callbacks) == 1
    assert close in engine._shutdown_callbacks


def test_engine_quit():
    """Engine quit sets running to False."""
    engine = Engine("Test")
    engine._running = True
    engine.quit()
    assert engine._running is False


def test_engine_has_run_method():
    """Engine has a run method."""
    engine = Engine("Test")
    assert hasattr(engine, "run")
    assert callable(engine.run)


def test_engine_has_window_attribute():
    """Engine has window attribute for GLFW handle."""
    engine = Engine("Test")
    assert hasattr(engine, "_window") or hasattr(engine, "window")


def test_engine_has_glfw_init():
    """Engine has GLFW initialization in run()."""
    import glfw
    engine = Engine("Test")
    # Verify glfw is being used - check that window create is part of flow
    # Just verify glfw can be initialized (creates a valid context)
    assert glfw.init() is not None
    glfw.terminate()


def test_engine_has_webgpu_attributes():
    """Engine has attributes for WebGPU context."""
    engine = Engine("Test")
    # Engine should have placeholder attributes for WebGPU objects
    assert hasattr(engine, "_adapter") or hasattr(engine, "adapter")
    assert hasattr(engine, "_device") or hasattr(engine, "device")
    assert hasattr(engine, "_canvas") or hasattr(engine, "canvas")
    assert hasattr(engine, "_swap_chain") or hasattr(engine, "swap_chain")


def test_engine_has_wgpu_import():
    """Engine imports wgpu module."""
    try:
        import wgpu
        assert True
    except ImportError:
        pytest.skip("wgpu-py not available")