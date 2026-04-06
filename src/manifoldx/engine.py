import asyncio
import wgpu
import numpy as np
from time import perf_counter_ns

from rendercanvas.glfw import GlfwRenderCanvas, loop as glfw_loop

# Import ECS components
import manifoldx.ecs as ecs
from manifoldx.ecs import EntityStore
from manifoldx.commands import CommandBuffer, Command, CommandType
from manifoldx.systems import SystemRegistry
from manifoldx.resources import GeometryRegistry, MaterialRegistry
from manifoldx.renderer import RenderPipeline
from manifoldx.components import Transform, Mesh, Material
from manifoldx.camera import Camera


class Engine:
    def __init__(
        self,
        name: str,
        h: int = 600,
        w: int = 800,
        fullscreen: bool = False,
        max_entities: int = 100_000,
        check: bool = True,
    ):
        self.name = name
        self.h = h
        self.w = w
        self.fullscreen = fullscreen
        self.check = check  # Enable/disable validation warnings
        ecs.ENABLE_VALIDATION = check  # Set global flag for ECS validation
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
        self._fixed_dt_value = 1 / 60
        self._last_time = None
        self._start_time = None
        self.elapsed: float = 0.0

        # Register built-in components
        Transform.register(self.store)
        Mesh.register(self.store)
        Material.register(self.store)

        # Camera for MVP matrices
        self._camera = Camera()
        self.camera = self._camera

        # External lights (not in ECS)
        self._lights = []

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
        """Decorator to register a system function.

        Parses Query type hints to determine which components to query.
        """
        import inspect
        from manifoldx.systems import Query

        # Parse type hints to extract component names
        component_names = []
        hints = func.__annotations__
        for param_name, hint in hints.items():
            if isinstance(hint, Query):
                # Query[Transform] -> Query with components tuple
                comps = hint.components
                if not isinstance(comps, tuple):
                    comps = (comps,)
                for c in comps:
                    if isinstance(c, str):
                        component_names.append(c)
                    elif hasattr(c, "__name__"):
                        component_names.append(c.__name__)

        self.systems.register(func, component_names)
        return func

    def component(self, cls):
        """Decorator to register a component class with this engine."""
        from manifoldx.ecs import _make_component_class

        return _make_component_class(cls, self)

    def set_lights(self, lights: list):
        """Set external lights (passed to renderer, not in ECS)."""
        self._lights = lights

    def add_light(self, light):
        """Add a single light to the external lights list."""
        self._lights.append(light)

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
            self._start_time = current_time
            return self._fixed_dt_value

        dt = (current_time - self._last_time) / 1_000_000_000
        self._last_time = current_time
        self.elapsed = (current_time - self._start_time) / 1_000_000_000
        return dt

    # === Spawn & Destroy ===
    def spawn(self, *args, n: int = 1, **kwargs):
        """Spawn entities - register components immediately, then emit SPAWN command."""
        # Handle Mesh, Material, and Transform component objects
        processed_kwargs = {}

        # Convert positional args to kwargs by class name
        for arg in args:
            name = type(arg).__name__
            kwargs[name] = arg

        for name, value in kwargs.items():
            # Check if it's a custom component class (has _component_registry)
            if hasattr(value, "_component_registry") and hasattr(
                value, "_component_fields"
            ):
                # It's a custom component class like Cube
                # Register the component name immediately (not in command)
                if name not in self.store._components:
                    # Calculate total size of all fields
                    total_cols = 0
                    for field_name in value._component_fields:
                        comp_def = value._component_registry.get(field_name)
                        if comp_def:
                            shape = comp_def.shape
                            total_cols += int(np.prod(shape)) if shape else 1

                    # Register as single component
                    self.store.register_component(name, np.dtype("f4"), (total_cols,))
                    # Store field info for later access
                    value._component_name = name
                    value._component_start_idx = {}

                    col_idx = 0
                    for field_name in value._component_fields:
                        comp_def = value._component_registry.get(field_name)
                        if comp_def:
                            shape = comp_def.shape
                            size = int(np.prod(shape)) if shape else 1
                            value._component_start_idx[field_name] = (col_idx, size)
                            col_idx += size

            if hasattr(value, "get_data"):
                # It's a component object (Mesh, Material, Transform) - get data from it
                # Pass appropriate registry based on component type
                if name == "Material":
                    processed_kwargs[name] = value.get_data(n, self._material_registry)
                else:
                    processed_kwargs[name] = value.get_data(n, self._geometry_registry)
            elif np.isscalar(value):
                # Broadcast scalars to arrays
                processed_kwargs[name] = np.full((n,), value, dtype=np.float32)
            else:
                processed_kwargs[name] = value

        # Also register built-in Transform component if not already
        if "Transform" not in self.store._components:
            Transform.register(self.store)

        # NOW spawn the entities immediately (not deferred)
        if n > 0:
            self.store.spawn(n, **processed_kwargs)

    def destroy(self, indices):
        """Destroy entities matching condition by emitting DESTROY command."""
        if indices is None:
            return

        # If boolean array, convert to indices
        indices = np.asarray(indices)
        if indices.dtype == np.bool_:
            indices = np.where(indices)[0]

        if hasattr(indices, "__len__") and len(indices) > 0:
            self.commands.append(Command(CommandType.DESTROY, {"indices": indices}))

    def _init_webgpu(self):
        # Use rendercanvas's GlfwRenderCanvas
        self._render_canvas = GlfwRenderCanvas()

        # Get the wgpu context from the canvas
        self._wgpu_context = self._render_canvas.get_wgpu_context()

        # Request adapter and device (use sync API to avoid deprecation warnings)
        self._adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        self._device = self._adapter.request_device_sync()

        # Configure the swap chain
        self._wgpu_context.configure(
            device=self._device,
            format=wgpu.TextureFormat.bgra8unorm,
        )

        # Depth texture created lazily in _draw_frame to match canvas size
        self._depth_texture = None
        self._depth_texture_view = None
        self._depth_texture_size = (0, 0)

    def _draw_frame(self):
        """Draw callback invoked by rendercanvas event loop each frame."""
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
        # Ensure render pipeline is initialized
        self._render_pipeline._ensure_pipeline(
            self._device, wgpu.TextureFormat.bgra8unorm
        )

        # Also update registries with device reference
        self._geometry_registry._device = self._device
        self._material_registry._device = self._device

        # Get the next frame's texture and create view
        texture = self._wgpu_context.get_current_texture()
        texture_view = texture.create_view()

        # Recreate depth texture if canvas size changed
        tex_size = (texture.size[0], texture.size[1])
        if tex_size != self._depth_texture_size:
            self._depth_texture = self._device.create_texture(
                size=(tex_size[0], tex_size[1], 1),
                format=wgpu.TextureFormat.depth24plus,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self._depth_texture_view = self._depth_texture.create_view()
            self._depth_texture_size = tex_size

        # Create command encoder
        command_encoder = self._device.create_command_encoder()

        # Create render pass with clear color and depth attachment
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture_view,
                    "resolve_target": None,
                    "clear_value": (0.1, 0.1, 0.2, 1.0),  # Dark blue background
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self._depth_texture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )

        # Issue draw calls via render pipeline
        self._render_pipeline.render(self, render_pass)

        # End render pass
        render_pass.end()

        # Submit command buffer
        self._device.queue.submit([command_encoder.finish()])

        # Request next frame
        self._render_canvas.request_draw()

    def run(self):
        self._init_webgpu()

        self._running = True
        self._start_time = perf_counter_ns()
        self._last_time = self._start_time
        self.elapsed = 0.0

        for callback in self._startup_callbacks:
            callback()

        # Register draw callback and run the canvas event loop
        self._render_canvas.request_draw(self._draw_frame)
        glfw_loop.run()

        for callback in self._shutdown_callbacks:
            callback()
