"""pytest configuration - mock display-dependent modules for headless testing."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_rendercanvas(monkeypatch):
    """Mock rendercanvas to avoid window creation in tests."""
    mock_canvas = MagicMock()
    mock_canvas.get_wgpu_context.return_value = MagicMock(
        configure=MagicMock(),
        get_current_texture=MagicMock(
            return_value=MagicMock(texture=MagicMock(), texture_view=MagicMock())
        ),
    )

    mock_glfw = MagicMock()
    mock_glfw.GlfwRenderCanvas.return_value = mock_canvas
    mock_glfw.loop = MagicMock()

    # Patch before importing manifoldx
    import sys

    sys.modules["rendercanvas"] = mock_glfw
    sys.modules["rendercanvas.glfw"] = mock_glfw
    sys.modules["rendercanvas.glfw"] = mock_glfw

    yield mock_glfw
