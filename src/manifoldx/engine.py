import asyncio
import wgpu
import sys
import numpy as np
from time import perf_counter_ns

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
        title: str,
        *,
        height: int = 600,
        width: int = 800,
        fullscreen: bool = False,
        max_entities: int = 100_000,
        check_numerics: bool = True,
    ):
        self.title = title
        self.h = height
        self.w = width
        self.fullscreen = fullscreen
        self.check = check_numerics  # Enable/disable validation warnings
        ecs.ENABLE_VALIDATION = check_numerics  # Set global flag for ECS validation

        # Private stuff
        self._running = False
        self._render_canvas = None
        self._wgpu_context = None
        self._adapter = None
        self._device = None
        self._event_loop = None
        self._present_mode = "fifo"
        self._texture_format = wgpu.TextureFormat.bgra8unorm
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
        # Stop the event loop
        if hasattr(self, "_event_loop") and self._event_loop is not None:
            self._event_loop.stop()
        # Also close the canvas
        if self._render_canvas is not None:
            try:
                self._render_canvas.close()
            except Exception:
                pass

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
            if hasattr(value, "_component_registry") and hasattr(value, "_component_fields"):
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

    def _draw_frame(self):
        """Draw callback invoked by rendercanvas event loop each frame."""
        # Check if we should stop - must be FIRST to avoid crashes
        if not self._running:
            return False  # Stop the event loop

        # Double-check canvas state
        try:
            if self._render_canvas.get_closed():
                return False
        except Exception:
            return False

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
        self._render_pipeline._ensure_pipeline(self._device, wgpu.TextureFormat.bgra8unorm)

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

        # Request next frame only if still running
        if self._running:
            self._render_canvas.request_draw()

    def run(self):
        """Run the engine with real-time rendering.

        Automatically uses:
        - PyodideRenderCanvas if running in browser (emscripten)
        - GlfwRenderCanvas otherwise (desktop)

        Note: Use render() for video output instead.
        """
        # Import backends module for lazy canvas creation
        from manifoldx.backends import get_desktop_canvas

        # Create canvas (GLFW on desktop, handled by backends module)
        self._render_canvas = get_desktop_canvas(
            width=self.w,
            height=self.h,
            fullscreen=self.fullscreen,
            title=self.title,
        )

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

        # Depth texture created lazily in _draw_frame to match canvas size
        self._depth_texture = None
        self._depth_texture_view = None
        self._depth_texture_size = (0, 0)

        self._running = True
        self._start_time = perf_counter_ns()
        self._last_time = self._start_time
        self.elapsed = 0.0

        for callback in self._startup_callbacks:
            callback()

        # Register draw callback and run the canvas event loop
        from rendercanvas.glfw import loop as glfw_loop

        self._event_loop = glfw_loop
        self._render_canvas.request_draw(self._draw_frame)
        try:
            glfw_loop.run()
        except Exception as e:
            print(f"Event loop error: {e}")
        finally:
            for callback in self._shutdown_callbacks:
                callback()

    def render(
        self,
        output: str,
        *,
        fps: int = 30,
        duration: float | None = None,
        frame_count: int | None = None,
        codec: str = "h264",
        quality: str = "high",
        progress: bool = True,
    ):
        """Render to video file.

        Uses OffscreenRenderCanvas for headless rendering.

        Parameters
        ----------
        output : str
            Output video file path (required).
        fps : int
            Frames per second (default: 30).
        duration : float
            Duration in seconds (alternative to frame_count).
        frame_count : int
            Number of frames to render (alternative to duration).
        codec : str
            Video codec: "h264", "hevc", or "mp4v" (default: "h264").
        quality : str
            Quality preset: "low", "medium", "high" (default: "high").
        progress : bool
            Show progress bar (default: True).
        """
        # Validate parameters
        if output is None:
            raise ValueError("output path required for render()")
        if duration is None and frame_count is None:
            raise ValueError("Either duration or frame_count must be specified")

        # Calculate frame count from duration if needed
        if frame_count is None:
            frame_count = int(duration * fps)

        dt = 1.0 / fps

        # Import backends module for lazy canvas creation
        from manifoldx.backends import get_offscreen_canvas

        # Create offscreen canvas for headless rendering
        self._render_canvas = get_offscreen_canvas(width=self.w, height=self.h)

        # Initialize WebGPU
        self._wgpu_context = self._render_canvas.get_wgpu_context()
        self._adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._device = self._adapter.request_device_sync()

        # Get preferred texture format from canvas context
        texture_format = self._wgpu_context.get_preferred_format(self._adapter)
        self._texture_format = texture_format
        self._wgpu_context.configure(device=self._device, format=texture_format)

        # Depth texture (created lazily per frame)
        self._depth_texture = None
        self._depth_texture_view = None
        self._depth_texture_size = (0, 0)

        # Set up video writer with imageio-ffmpeg
        writer = self._get_video_writer(output, fps, self.w, self.h, codec, quality)

        # Initialize timing
        self._start_time = perf_counter_ns()
        self._last_time = self._start_time
        self.elapsed = 0.0

        # Run startup callbacks
        for callback in self._startup_callbacks:
            callback()

        # Progress tracking
        from tqdm import tqdm

        pbar = tqdm(total=frame_count, desc="Rendering video")

        # === RENDER LOOP ===
        for frame_idx in range(frame_count):
            # Clear command buffer
            self.commands.clear()

            # Run all user systems with fixed timestep
            self.systems.run_all(self, dt)

            # Execute command buffer
            self.commands.execute(self.store)

            # Run render pipeline
            self._render_pipeline.run(self, dt)
            self._render_pipeline._ensure_pipeline(self._device, wgpu.TextureFormat.bgra8unorm)

            # Update registries with device reference
            self._geometry_registry._device = self._device
            self._material_registry._device = self._device

            # Get current texture and create view
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

            # Begin render pass
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": texture_view,
                        "resolve_target": None,
                        "clear_value": (0.1, 0.1, 0.2, 1.0),
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

            # Issue draw calls
            self._render_pipeline.render(self, render_pass)

            # End render pass
            render_pass.end()

            # Submit command buffer
            self._device.queue.submit([command_encoder.finish()])

            # Capture frame from canvas
            frame = self._render_canvas.draw()

            # Convert RGBA to RGB (most video codecs don't support alpha)
            frame_rgb = frame[:, :, :3].copy()  # Make contiguous

            # Write to video
            writer.send(frame_rgb)

            # Update progress
            pbar.update(1)

            # Update elapsed time
            self.elapsed = (frame_idx + 1) * dt

        # Clean up
        writer.close()
        writer = None  # type: ignore
        pbar.close()

        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            callback()

        print(f"Video saved: {output}")

    def _get_video_writer(self, output, fps, width, height, codec, quality):
        """Set up imageio-ffmpeg video writer."""
        import imageio_ffmpeg

        # Quality presets (1-10 scale for imageio-ffmpeg)
        quality_map = {
            "low": 1,
            "medium": 3,
            "high": 5,
            "ultra": 8,
        }
        quality_value = quality_map.get(quality, 5)

        # Get writer with ffmpeg
        writer = imageio_ffmpeg.write_frames(
            output,
            size=(width, height),
            fps=fps,
            codec="libx264" if codec == "h264" else codec,
            quality=quality_value,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
        )

        # Seed the generator
        writer.send(None)

        return writer
