import asyncio
import wgpu

from rendercanvas.glfw import GlfwRenderCanvas


class Engine:
    def __init__(self, name: str, h: int = 600, w: int = 800, fullscreen: bool = False):
        self.name = name
        self.h = h
        self.w = w
        self.fullscreen = fullscreen
        self._running = False
        self._render_canvas = None
        self._wgpu_context = None
        self._adapter = None
        self._device = None
        self._present_mode = "fifo"
        self._startup_callbacks = []
        self._shutdown_callbacks = []
        self._update_callbacks = []

    def startup(self, func):
        self._startup_callbacks.append(func)
        return func

    def shutdown(self, func):
        self._shutdown_callbacks.append(func)
        return func

    def update(self, func):
        self._update_callbacks.append(func)
        return func

    def quit(self):
        self._running = False

    def _init_webgpu(self):
        # Use rendercanvas's GlfwRenderCanvas
        self._render_canvas = GlfwRenderCanvas()

        # Get the wgpu context from the canvas
        self._wgpu_context = self._render_canvas.get_wgpu_context()

        # Request adapter and device (use sync API to avoid deprecation warnings)
        self._adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._device = self._adapter.request_device_sync()

        # Configure the swap chain
        self._wgpu_context.configure(
            device=self._device,
            format=wgpu.TextureFormat.bgra8unorm,
        )

    def _render_frame(self):
        """Render a single frame: acquire, encode, render, present."""
        # Get the next frame's texture view
        texture_view = self._wgpu_context.get_current_texture()

        # Create command encoder
        command_encoder = self._device.create_command_encoder()

        # Create render pass with clear color
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture_view,
                    "resolve_target": None,
                    "clear_value": (0.1, 0.1, 0.2, 1.0),  # Dark blue background
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )

        # End render pass (no actual rendering yet, just clear)
        render_pass.end()

        # Submit command buffer
        self._device.queue.submit([command_encoder.finish()])

        # Present the frame (swaps buffer for FIFO)
        self._wgpu_context.present()

    def run(self):
        self._init_webgpu()

        self._running = True
        for callback in self._startup_callbacks:
            callback()

        while self._running:
            # Call update callbacks
            for callback in self._update_callbacks:
                callback()

            # Render frame
            self._render_frame()

            # Check if window should close
            if self._render_canvas.is_closing:
                self._running = False

        for callback in self._shutdown_callbacks:
            callback()
