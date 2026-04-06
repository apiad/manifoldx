import asyncio
import wgpu
import numpy as np
from time import perf_counter_ns

from rendercanvas.glfw import GlfwRenderCanvas

# Import ECS components
from manifoldx.ecs import EntityStore
from manifoldx.commands import CommandBuffer, Command, CommandType
from manifoldx.systems import SystemRegistry
from manifoldx.resources import GeometryRegistry, MaterialRegistry
from manifoldx.renderer import RenderPipeline
from manifoldx.components import Transform, Mesh, Material


class Engine:
    def __init__(self, name: str, h: int = 600, w: int = 800, fullscreen: bool = False,
                 max_entities: int = 100_000):
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
        
        # === ECS Infrastructure ===
        self.store = EntityStore(max_entities)
        self.commands = CommandBuffer()
        self.systems = SystemRegistry()
        
        # Resource registries
        self._geometry_registry = GeometryRegistry(self._device)
        self._material_registry = MaterialRegistry(self._device)
        
        # Render pipeline
        self._render_pipeline = RenderPipeline(self.store, self._device)
        
        # Configurable timestep
        self._use_fixed_dt = False
        self._fixed_dt_value = 1/60
        self._last_time = None
        
        # Register built-in components
        Transform.register(self.store)
        Mesh.register(self.store)
        Material.register(self.store)

    def startup(self, func):
        self._startup_callbacks.append(func)
        return func

    def shutdown(self, func):
        self._shutdown_callbacks.append(func)
        return func

    def update(self, func):
        self._update_callbacks.append(func)
        return func

    def system(self, func):
        """Decorator to register a system function."""
        self.systems.register(func)
        return func

    def quit(self):
        self._running = False
    
    # === Timestep Configuration ===
    def set_fixed_timestep(self, dt: float):
        """Use fixed timestep for deterministic simulations."""
        self._use_fixed_dt = True
        self._fixed_dt_value = dt
        
    def use_wall_clock(self):
        """Use actual elapsed time (default for animations)."""
        self._use_fixed_dt = False
        
    def _compute_dt(self):
        """Compute delta time based on configuration."""
        if self._use_fixed_dt:
            return self._fixed_dt_value
        
        current_time = perf_counter_ns()
        if self._last_time is None:
            self._last_time = current_time
            return self._fixed_dt_value
            
        dt = (current_time - self._last_time) / 1_000_000_000
        self._last_time = current_time
        return dt
    
    # === Spawn & Destroy ===
    def spawn(self, *args, n: int, **kwargs):
        """Spawn entities by emitting SPAWN command."""
        # Handle Mesh, Material, and Transform component objects
        processed_kwargs = {}
        for name, value in kwargs.items():
            if hasattr(value, 'get_data'):
                # It's a component object (Mesh, Material, Transform) - get data from it
                processed_kwargs[name] = value.get_data(n, self._geometry_registry)
            elif np.isscalar(value):
                # Broadcast scalars to arrays
                processed_kwargs[name] = np.full((n,), value, dtype=np.float32)
            else:
                processed_kwargs[name] = value
                
        # Emit SPAWN command
        self.commands.append(Command(
            CommandType.SPAWN,
            {'n': n, 'components': processed_kwargs}
        ))
        
    def destroy(self, indices):
        """Destroy entities matching condition by emitting DESTROY command."""
        if hasattr(indices, '__len__') and len(indices) > 0:
            self.commands.append(Command(
                CommandType.DESTROY,
                {'indices': np.asarray(indices)}
            ))

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
        self._last_time = perf_counter_ns()
        
        for callback in self._startup_callbacks:
            callback()

        while self._running:
            dt = self._compute_dt()
            
            # 1. Clear command buffer for this frame
            self.commands.clear()
            
            # 2. Run all user systems (they emit commands)
            self.systems.run_all(self, dt)
            
            # 3. Execute command buffer (apply all spawn/destroy/update)
            self.commands.execute(self.store)
            
            # 4. RENDER PIPELINE (runs after commands)
            self._render_pipeline.run(self, dt)
            
            # 5. Render frame to screen
            self._render_frame()

            # Check if window should close
            if self._render_canvas.is_closing:
                self._running = False

        for callback in self._shutdown_callbacks:
            callback()
