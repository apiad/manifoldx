import asyncio
import glfw
import wgpu


class Engine:
    def __init__(self, name: str, h: int = 600, w: int = 800, fullscreen: bool = False):
        self.name = name
        self.h = h
        self.w = w
        self.fullscreen = fullscreen
        self._running = False
        self._window = None
        self._adapter = None
        self._device = None
        self._canvas = None
        self._swap_chain = None
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
        # Get canvas context using rendercanvas's WgpuContext
        from rendercanvas.contexts import WgpuContext
        
        # Get platform (x11 or wayland)
        platform = "x11"
        if glfw.get_platform() == glfw.PLATFORM_WAYLAND:
            platform = "wayland"
        
        # Get display for the platform
        display = None
        if platform == "wayland":
            display = glfw.get_wayland_display()
        
        # Create present_info dict
        present_info = {
            "window": self._window,
            "platform": platform,
        }
        if display:
            present_info["display"] = display
        
        # Create canvas context
        self._canvas = WgpuContext(present_info)
        
        # Request adapter and device
        self._adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self._device = asyncio.run(self._adapter.request_device())
        
        # Configure swap chain
        self._canvas.configure(
            device=self._device,
            format=wgpu.TextureFormat.bgra8unorm,
        )

    def _render_frame(self):
        """Render a single frame: acquire, encode, render, present."""
        # Get the next frame's texture view
        texture_view = self._canvas.get_current_texture()
        
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
        self._canvas.present()

    def run(self):
        glfw.init()
        self._window = glfw.create_window(self.w, self.h, self.name, None, None)
        glfw.make_context_current(self._window)

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
            
            # Poll GLFW events
            glfw.poll_events()
            
            # Check if window should close
            if glfw.window_should_close(self._window):
                self._running = False
        
        for callback in self._shutdown_callbacks:
            callback()

        if self._window:
            glfw.destroy_window(self._window)
        glfw.terminate()
