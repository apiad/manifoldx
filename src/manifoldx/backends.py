"""Lazy backend imports for rendering canvases.

This module provides functions to create render canvases for different backends.
Each function performs lazy imports to provide clear error messages when
dependencies are missing.

Usage:
    from manifoldx.backends import get_desktop_canvas, get_offscreen_canvas

    # Desktop (requires: pip install manifold-gfx[desktop])
    canvas = get_desktop_canvas(width=800, height=600, fullscreen=False, title="MyApp")

    # Offscreen (requires: pip install manifold-gfx[offline])
    canvas = get_offscreen_canvas(width=1920, height=1080)
"""

from __future__ import annotations


def get_desktop_canvas(
    width: int,
    height: int,
    fullscreen: bool,
    title: str,
) -> "BaseRenderCanvas":
    """Create a desktop canvas using GLFW.

    Requires: pip install manifold-gfx[desktop]

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    fullscreen : bool
        Whether to create a fullscreen window.
    title : str
        Window title.

    Returns
    -------
    BaseRenderCanvas
        The GLFW render canvas.

    Raises
    ------
    ImportError
        If glfw is not installed.
    RuntimeError
        If not running in a desktop environment.
    """
    # Lazy import - raises clear error if glfw not installed
    try:
        from rendercanvas.glfw import GlfwRenderCanvas
    except ImportError:
        raise ImportError(
            "Desktop backend requires glfw.\nInstall with: pip install manifold-gfx[desktop]"
        )

    # Create the canvas
    canvas = GlfwRenderCanvas(size=(width, height), title=title)

    # Handle fullscreen via GLFW
    if fullscreen:
        import glfw

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_monitor(
            canvas._window,
            monitor,
            0,
            0,
            mode.size[0],
            mode.size[1],
            mode.refresh_rate,
        )

    return canvas


def get_offscreen_canvas(width: int, height: int) -> "BaseRenderCanvas":
    """Create an offscreen canvas for headless rendering.

    Requires: pip install manifold-gfx[offline]

    Parameters
    ----------
    width : int
        Canvas width in pixels.
    height : int
        Canvas height in pixels.

    Returns
    -------
    BaseRenderCanvas
        The offscreen render canvas.

    Raises
    ------
    ImportError
        If imageio-ffmpeg is not installed.
    """
    # First check for imageio_ffmpeg - raises clear error if not installed
    try:
        import imageio_ffmpeg  # noqa: F401 - just checking availability
    except ImportError:
        raise ImportError(
            "Offline rendering requires imageio-ffmpeg.\n"
            "Install with: pip install manifold-gfx[offline]"
        ) from None

    # Now import the offscreen canvas (only if imageio-ffmpeg is available)
    from rendercanvas.offscreen import OffscreenRenderCanvas

    return OffscreenRenderCanvas(size=(width, height), format="rgba-u8")


# Type hint for the canvas return type
BaseRenderCanvas = None  # Will be set when rendercanvas is imported
