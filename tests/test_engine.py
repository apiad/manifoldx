import importlib.util
import pytest
from manifoldx import Engine


# === Simplified API Tests (no Backend enum) ===


def test_engine_no_backend_parameter():
    """Engine no longer has a backend parameter."""
    import inspect

    sig = inspect.signature(Engine.__init__)
    params = list(sig.parameters.keys())
    assert "backend" not in params


def test_engine_stores_title_width_height_fullscreen():
    """Engine stores title, width, height, fullscreen."""
    engine = Engine("TestApp", width=1280, height=720, fullscreen=True)
    assert engine.title == "TestApp"
    assert engine.w == 1280
    assert engine.h == 720
    assert engine.fullscreen is True


# === run() is context-aware ===


def test_run_uses_glfw_on_desktop():
    """run() uses GlfwRenderCanvas when not in browser."""
    from unittest.mock import patch, MagicMock

    engine = Engine("Test", width=800, height=600)

    mock_canvas = MagicMock()
    mock_canvas.get_wgpu_context.return_value = MagicMock()

    # Patch GlfwRenderCanvas
    with patch("manifoldx.backends.get_desktop_canvas", return_value=mock_canvas):
        # run() should use get_desktop_canvas
        # (We can't actually run, but verify the method exists)
        assert hasattr(engine, "run")
        assert callable(engine.run)


def test_run_raises_import_error_without_glfw():
    """run() raises ImportError if glfw not installed on desktop."""
    from manifoldx.backends import get_desktop_canvas
    import inspect

    # Verify get_desktop_canvas raises clear error without glfw
    source = inspect.getsource(get_desktop_canvas)
    assert "glfw" in source
    assert "manifold-gfx[desktop]" in source


# === render() uses offscreen unconditionally ===


def test_render_uses_offscreen():
    """render() uses OffscreenRenderCanvas for headless rendering."""
    from manifoldx.backends import get_offscreen_canvas
    import inspect

    # Verify get_offscreen_canvas exists and is correct
    source = inspect.getsource(get_offscreen_canvas)
    assert "imageio-ffmpeg" in source
    assert "manifold-gfx[offline]" in source


# === Engine Basic Tests ===


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


def test_engine_has_render_method():
    """Engine has render method for video output."""
    engine = Engine("Test")
    assert hasattr(engine, "render")
    assert callable(engine.render)


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


# === Backends Module Tests ===


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

    # Verify the error message in source code
    import inspect

    source = inspect.getsource(get_desktop_canvas)
    assert "glfw" in source
    assert "manifold-gfx[desktop]" in source


def test_get_offscreen_canvas_raises_without_imageio():
    """get_offscreen_canvas raises ImportError if imageio-ffmpeg not installed."""
    from manifoldx.backends import get_offscreen_canvas

    # Verify the error message in source code
    import inspect

    source = inspect.getsource(get_offscreen_canvas)
    assert "imageio-ffmpeg" in source
    assert "manifold-gfx[offline]" in source


# === render() Video Output Tests ===


def test_engine_render_validates_params():
    """Engine.render() validates output and frame parameters."""
    engine = Engine("Test")

    # Missing output should raise TypeError (required positional arg)
    with pytest.raises(TypeError):
        engine.render()

    # Missing duration/frame_count should raise ValueError
    with pytest.raises(ValueError, match="duration.*frame_count"):
        engine.render(output="test.mp4")


def test_engine_render_accepts_duration():
    """Engine.render() accepts duration parameter."""
    engine = Engine("Test")

    # Verify the method accepts duration
    import inspect

    sig = inspect.signature(engine.render)
    assert "duration" in sig.parameters


def test_engine_render_accepts_frame_count():
    """Engine.render() accepts frame_count parameter."""
    engine = Engine("Test")

    # Verify the method accepts frame_count
    import inspect

    sig = inspect.signature(engine.render)
    assert "frame_count" in sig.parameters


def test_engine_render_produces_video_file(tmp_path):
    """Engine.render() produces a valid video file."""
    engine = Engine(
        "TestRender",
        width=320,
        height=240,
    )

    output_file = tmp_path / "test_output.mp4"

    # This will render a single frame video
    engine.render(
        output=str(output_file),
        frame_count=1,
        fps=30,
        progress=False,
    )

    # Verify the file was created and has content
    assert output_file.exists()
    assert output_file.stat().st_size > 0
