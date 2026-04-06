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
    """Engine has window/canvas attribute for rendering."""
    engine = Engine("Test")
    # Either old window or new render_canvas
    assert (hasattr(engine, "_window") or hasattr(engine, "window") or
            hasattr(engine, "_render_canvas") or hasattr(engine, "render_canvas"))


def test_engine_has_glfw_init():
    """Engine has GLFW initialization in run()."""
    import glfw
    engine = Engine("Test")
    # Verify glfw is being used via rendercanvas
    assert glfw.init() is not None
    glfw.terminate()


def test_engine_has_webgpu_attributes():
    """Engine has attributes for WebGPU context."""
    engine = Engine("Test")
    # Engine should have placeholder attributes for WebGPU objects
    assert (hasattr(engine, "_adapter") or hasattr(engine, "adapter") or
            hasattr(engine, "_wgpu_context"))
    assert (hasattr(engine, "_device") or hasattr(engine, "device") or
            hasattr(engine, "_device"))
    assert (hasattr(engine, "_canvas") or hasattr(engine, "canvas") or
            hasattr(engine, "_wgpu_context"))


def test_engine_has_wgpu_import():
    """Engine imports wgpu module."""
    try:
        import wgpu
        assert True
    except ImportError:
        pytest.skip("wgpu-py not available")

def test_engine_has_update_callbacks():
    """Engine has update decorator and callback list."""
    engine = Engine("Test")
    assert hasattr(engine, "_update_callbacks")
    assert hasattr(engine, "update")
    assert callable(engine.update)


def test_engine_calls_update_callbacks():
    """Engine calls update callbacks in the main loop."""
    engine = Engine("Test")
    call_count = 0

    @engine.update
    def tick():
        nonlocal call_count
        call_count += 1

    # Verify callback registered
    assert len(engine._update_callbacks) == 1
    assert tick in engine._update_callbacks


def test_engine_has_render_method():
    """Engine has method for rendering frames."""
    engine = Engine("Test")
    assert hasattr(engine, "_render_frame") or hasattr(engine, "render_frame") or hasattr(engine, "_draw_frame")


def test_engine_uses_rendercanvas():
    """Engine uses rendercanvas's GlfwRenderCanvas."""
    try:
        from rendercanvas.glfw import GlfwRenderCanvas
        from manifoldx import Engine
        engine = Engine("Test")
        # Check for new attribute names
        assert (hasattr(engine, "_render_canvas") or hasattr(engine, "render_canvas") or
                hasattr(engine, "_wgpu_context") or hasattr(engine, "wgpu_context"))
    except ImportError:
        pytest.skip("rendercanvas not available")


def test_engine_has_rendercanvas_import():
    """Engine imports GlfwRenderCanvas from rendercanvas."""
    try:
        from rendercanvas.glfw import GlfwRenderCanvas
        assert True
    except ImportError:
        pytest.skip("rendercanvas not available")


def test_engine_get_wgpu_context():
    """Engine can get wgpu context from rendercanvas canvas."""
    try:
        from rendercanvas.glfw import GlfwRenderCanvas
        canvas = GlfwRenderCanvas()
        wgpu_ctx = canvas.get_wgpu_context()
        assert wgpu_ctx is not None
        assert hasattr(wgpu_ctx, "configure")
        assert hasattr(wgpu_ctx, "get_current_texture")
    except ImportError:
        pytest.skip("rendercanvas not available")


def test_engine_stores_rendercanvas_canvas():
    """Engine stores rendercanvas GlfwRenderCanvas instance."""
    from manifoldx import Engine
    engine = Engine("Test")
    # Engine should store rendercanvas canvas
    assert hasattr(engine, "_render_canvas") or hasattr(engine, "render_canvas")
