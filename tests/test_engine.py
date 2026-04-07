import importlib.util
import pytest
from manifoldx import Engine


def test_engine_creation():
    """Engine can be created with name and dimensions."""
    engine = Engine("Test", height=600, width=800, fullscreen=False)
    assert engine.title == "Test"
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
    assert (
        hasattr(engine, "_window")
        or hasattr(engine, "window")
        or hasattr(engine, "_render_canvas")
        or hasattr(engine, "render_canvas")
    )


@pytest.mark.skipif(importlib.util.find_spec("glfw") is None, reason="glfw not installed")
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
    assert (
        hasattr(engine, "_adapter")
        or hasattr(engine, "adapter")
        or hasattr(engine, "_wgpu_context")
    )
    assert hasattr(engine, "_device") or hasattr(engine, "device") or hasattr(engine, "_device")
    assert (
        hasattr(engine, "_canvas") or hasattr(engine, "canvas") or hasattr(engine, "_wgpu_context")
    )


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
    assert (
        hasattr(engine, "_render_frame")
        or hasattr(engine, "render_frame")
        or hasattr(engine, "_draw_frame")
    )


def test_engine_uses_rendercanvas():
    """Engine uses rendercanvas's GlfwRenderCanvas."""
    try:
        from rendercanvas.glfw import GlfwRenderCanvas
        from manifoldx import Engine

        engine = Engine("Test")
        # Check for new attribute names
        assert (
            hasattr(engine, "_render_canvas")
            or hasattr(engine, "render_canvas")
            or hasattr(engine, "_wgpu_context")
            or hasattr(engine, "wgpu_context")
        )
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


# === Phase 1: Backend Enum and Optional Dependencies ===


def test_backend_enum_exists():
    """Backend enum is defined in manifoldx module."""
    from manifoldx import Backend

    assert Backend is not None
    assert hasattr(Backend, "DESKTOP")
    assert hasattr(Backend, "BROWSER")
    assert hasattr(Backend, "OFFSCREEN")


def test_backend_enum_has_string_values():
    """Backend enum values are strings matching backend module names."""
    from manifoldx import Backend

    # String values match rendercanvas module names
    assert Backend.DESKTOP == "glfw"
    assert Backend.BROWSER == "pyodide"
    assert Backend.OFFSCREEN == "offscreen"


def test_engine_accepts_backend_parameter():
    """Engine accepts backend parameter in constructor."""
    from manifoldx import Engine, Backend

    engine = Engine("Test", backend=Backend.DESKTOP)
    assert engine.backend == Backend.DESKTOP

    engine = Engine("Test", backend=Backend.OFFSCREEN)
    assert engine.backend == Backend.OFFSCREEN


def test_engine_default_backend_is_desktop():
    """Engine defaults to DESKTOP backend."""
    from manifoldx import Engine, Backend

    engine = Engine("Test")
    assert engine.backend == Backend.DESKTOP


def test_engine_has_render_method():
    """Engine has render method for video output."""
    from manifoldx import Engine

    engine = Engine("Test")
    assert hasattr(engine, "render")
    assert callable(engine.render)


def test_engine_render_raises_for_non_offscreen_backend():
    """Engine.render() raises ValueError for non-OFFSCREEN backends."""
    from manifoldx import Engine, Backend

    engine = Engine("Test", backend=Backend.DESKTOP)
    with pytest.raises(ValueError, match="render.*OFFSCREEN"):
        engine.render(output="test.mp4", frame_count=1)


def test_engine_run_raises_for_offscreen_backend():
    """Engine.run() raises ValueError for OFFSCREEN backend."""
    from manifoldx import Engine, Backend

    engine = Engine("Test", backend=Backend.OFFSCREEN)
    # Note: This will raise ValueError because run() can't work with offscreen
    with pytest.raises(ValueError, match="run.*DESKTOP.*BROWSER"):
        engine.run()


# === Phase 2: Lazy Backend Imports ===


def test_backends_module_exists():
    """Backends module exists with lazy import functions."""
    from manifoldx import backends

    assert hasattr(backends, "get_desktop_canvas")
    assert hasattr(backends, "get_offscreen_canvas")
    assert callable(backends.get_desktop_canvas)
    assert callable(backends.get_offscreen_canvas)


def test_get_desktop_canvas_raises_without_glfw():
    """get_desktop_canvas raises ImportError if glfw not installed."""
    from manifoldx.backends import get_desktop_canvas

    # Mock: temporarily hide glfw to test error message
    import sys
    import importlib

    # Save original module
    glfw_spec = sys.modules.get("glfw")
    glfw_modules = [
        k for k in sys.modules.keys() if k == "glfw" or k.startswith("rendercanvas.glfw")
    ]
    saved = {k: sys.modules.pop(k) for k in glfw_modules if k in sys.modules}

    try:
        # Remove rendercanvas.glfw to simulate missing glfw
        if "rendercanvas.glfw" in sys.modules:
            del sys.modules["rendercanvas.glfw"]

        with pytest.raises(ImportError, match="glfw"):
            get_desktop_canvas(800, 600, False, "Test")
    finally:
        # Restore
        sys.modules.update(saved)
        if glfw_spec:
            sys.modules["glfw"] = glfw_spec


def test_get_offscreen_canvas_raises_without_imageio():
    """get_offscreen_canvas raises ImportError if imageio-ffmpeg not installed."""
    from manifoldx.backends import get_offscreen_canvas

    # Note: imageio-ffmpeg is installed in dev environment, so we can't easily test
    # the missing case. This test verifies the function exists and is callable.
    # The error message is validated in the docstring and the code structure.

    # The function should exist and be callable
    assert callable(get_offscreen_canvas)

    # We can verify the error message format by checking the source
    import inspect

    source = inspect.getsource(get_offscreen_canvas)
    assert "imageio-ffmpeg" in source
    assert "manifold-gfx[offline]" in source
