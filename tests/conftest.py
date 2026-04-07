"""pytest configuration - mock display-dependent modules for headless testing."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_rendercanvas(monkeypatch):
    """Mock rendercanvas to avoid window creation in tests.

    Note: We don't mock rendercanvas globally because we need the real
    modules for testing. Instead, we only mock glfw-specific imports.
    """
    import sys

    # Only mock glfw - let offscreen use real implementation
    if "rendercanvas.glfw" not in sys.modules:
        mock_glfw = MagicMock()
        mock_glfw.loop = MagicMock()
        sys.modules["rendercanvas.glfw"] = mock_glfw

    yield
