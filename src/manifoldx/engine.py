import asyncio
import argparse
import wgpu
import sys
import numpy as np
from pathlib import Path
from time import perf_counter_ns

# Import ECS components
import manifoldx.ecs as ecs
from manifoldx.ecs import EntityStore
from manifoldx.commands import CommandBuffer, Command, CommandType
from manifoldx.systems import SystemRegistry
from manifoldx.resources import GeometryRegistry, MaterialRegistry, VolumeRegistry
from manifoldx.renderer import RenderPipeline
from manifoldx.components import Component, Transform, Mesh, Material
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
        # Event-driven system (replaces _startup/_shutdown/_update_callbacks)
        from manifoldx.events import EventBus, FrameWaiters

        self._event_bus = EventBus()
        self._aio_loop = asyncio.new_event_loop()
        self._frame_waiters = FrameWaiters(self._aio_loop)
        # Task error spool — populated by add_done_callback when an async
        # handler raises (other than CancelledError). Drained by
        # _pump_aio_loop, which re-raises the first error per the v1
        # "errors crash the engine" policy.
        self._task_errors: list[BaseException] = []

        # === ECS Infrastructure ===
        self.store = EntityStore(max_entities)
        self.commands = CommandBuffer()
        self.systems = SystemRegistry()

        # Resource registries
        self._geometry_registry = GeometryRegistry(self._device)
        self._material_registry = MaterialRegistry(self._device)
        self._volume_registry = VolumeRegistry(self._device)

        # Render pipeline
        self._render_pipeline = RenderPipeline(self.store, self._device)

        # Compute systems — declarative GPU work registered via engine.compute(cls).
        from manifoldx.compute import ComputeRunner
        self._compute_runner = ComputeRunner(self)
        self._last_dt: float = 1 / 60
        self._frame_index: int = 0

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

        # Lazily-constructed label atlas, materialized on first label render.
        self._label_atlas = None

    def get_label_atlas(self):
        """Lazily construct the label atlas on first use.

        Used by the renderer's label pass and by user code that needs to
        register strings ahead of time.
        """
        if self._label_atlas is None:
            from manifoldx.viz.text import LabelTextureAtlas
            self._label_atlas = LabelTextureAtlas()
        return self._label_atlas

    def on(self, event: str):
        """Register a sync or async handler for an event.

        Usage:
            @engine.on("startup")
            def init(payload): ...

            @engine.on("frame")
            async def each_frame(payload): ...
        """
        return self._event_bus.on(event)

    def emit(self, event: str, payload=None) -> None:
        """Queue an event for delivery at the start of the next frame."""
        self._event_bus.emit(event, payload)

    def tick(self):
        """Return a future that resolves at the next frame boundary."""
        return self._frame_waiters.add_tick()

    def delay(self, seconds: float):
        """Return a future that resolves after `seconds` of engine.elapsed."""
        return self._frame_waiters.add_delay(seconds, self.elapsed)

    def elapsed_at(self, target: float):
        """Return a future that resolves once engine.elapsed >= target."""
        return self._frame_waiters.add_elapsed_at(target)

    async def run_blocking(self, fn, *args, **kwargs):
        """Run a blocking callable in the default executor and await its result."""
        import functools

        return await self._aio_loop.run_in_executor(
            None, functools.partial(fn, *args, **kwargs)
        )

    def _on_task_done(self, task) -> None:
        """Done-callback wired onto every async-handler task.

        Called synchronously when the task completes. We surface non-cancel
        exceptions into _task_errors so _pump_aio_loop can re-raise them on
        the next pump (per the v1 "errors crash the engine" policy).
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._task_errors.append(exc)

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

    def register_volume(
        self,
        data: "np.ndarray",
        *,
        name: str | None = None,
    ) -> int:
        """Register a 3D scalar field for volume rendering.

        Args:
            data: numpy array of shape (Nz, Ny, Nx), dtype=float32,
                  C-contiguous. data[k, j, i] is the voxel at integer
                  coordinates (i, j, k) along local-space (x, y, z).
            name: optional diagnostic label.

        Returns:
            Integer handle to pass into `Volume(volume_id=handle)`.
        """
        return self._volume_registry.register(data, name=name)

    def update_volume(self, handle: int, data: "np.ndarray") -> None:
        """Replace voxel data for a registered volume. Same shape required.

        Bumps the dirty bit; the renderer re-uploads on the next frame.
        """
        self._volume_registry.update(handle, data)

    def bind_compute_volume(self, handle: int, kernel_field: str) -> None:
        """v2: bind a Phase-2 compute kernel's `Writes[Volume3D]` field
        to this volume's storage texture. v1 raises NotImplementedError —
        the API is reserved here so v2 can land without a breaking change.
        """
        raise NotImplementedError(
            "bind_compute_volume is reserved for v2; v1 supports only "
            "CPU-uploaded volumes via register_volume()/update_volume()."
        )

    def compute(self, cls):
        """Register a Compute subclass to run as part of the per-frame loop.

        Usage:
            class Gravity(Compute):
                ...

            engine.compute(Gravity)

        Validates the kernel synchronously when the wgpu device is available:
        compiles the WGSL via `cls().compile()` and feeds it through
        `device.create_shader_module()` so transpile + validation errors
        surface at registration time, not on first frame.
        """
        if self._device is not None:
            instance = cls()
            wgsl = instance.compile()
            try:
                self._device.create_shader_module(code=wgsl)
            except Exception as e:
                from manifoldx.compute.transpile import ComputeShaderCompileError
                raise ComputeShaderCompileError(
                    category="wgpu-validation",
                    message=str(e),
                    filename=getattr(cls, "__module__", "<class>"),
                    line=0, col=0, source_line=None,
                )
        self._compute_runner.register(cls)
        return cls

    def set_lights(self, lights: list):
        """Set external lights (passed to renderer, not in ECS)."""
        self._lights = lights

    def add_light(self, light):
        """Add a single light to the external lights list."""
        self._lights.append(light)

    def quit(self):
        self._running = False
        self.shutdown_events()
        # Stop the rendercanvas event loop
        if hasattr(self, "_event_loop") and self._event_loop is not None:
            self._event_loop.stop()
        # Also close the canvas
        if self._render_canvas is not None:
            try:
                self._render_canvas.close()
            except Exception:
                pass

    def shutdown_events(self) -> None:
        """Fire 'shutdown', cancel async tasks, drain the loop, close it.

        Idempotent: safe to call from quit() and from finalization paths.
        """
        if self._aio_loop.is_closed():
            return

        # 1. Sync 'shutdown' handlers run inline; async ones get scheduled
        #    as tasks on the loop.
        self._event_bus.dispatch_immediate(self, "shutdown", {})

        # 2. Cancel every outstanding task so while-True coroutines get
        #    CancelledError on their next await.
        for task in list(asyncio.all_tasks(loop=self._aio_loop)):
            task.cancel()

        # 3. Cancel any outstanding waiter futures so coroutines blocked
        #    on engine.tick / delay / elapsed_at unblock with CancelledError.
        self._frame_waiters.cancel_all()

        # 4. Pump the loop one final time to let try/finally cleanup run.
        #    Done callbacks fire here, populating _task_errors for any
        #    non-cancel exceptions raised during cleanup.
        try:
            self._aio_loop.run_until_complete(asyncio.sleep(0))
        finally:
            pending = self._task_errors
            self._task_errors = []
            self._aio_loop.close()

        # 5. Surface the first non-cancel exception (if any) per the v1
        #    "errors propagate" policy.
        if pending:
            raise pending[0]

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
            # Auto-register Component subclasses on first encounter so callers
            # don't have to call `engine.store.register_component(...)` by hand.
            # Classes decorated with `@engine.component` register themselves at
            # decorator time via _make_component_class, so they don't need a
            # branch here — they fall through to the get_data path below.
            if isinstance(value, Component):
                type(value).register(self.store)

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

    # === Canvas Initialization ===

    def _init_canvas(self, canvas):
        """Initialize WebGPU context from a canvas (shared by run() and render())."""
        self._render_canvas = canvas

        # Get the wgpu context from the canvas
        self._wgpu_context = canvas.get_wgpu_context()

        # Request adapter and device (use sync API to avoid deprecation warnings)
        self._adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._device = self._adapter.request_device_sync()

        # Get preferred texture format from canvas context
        texture_format = self._wgpu_context.get_preferred_format(self._adapter)
        self._texture_format = texture_format

        # Configure the swap chain
        self._wgpu_context.configure(device=self._device, format=texture_format)

        # Update registries with device reference
        self._geometry_registry._device = self._device
        self._material_registry._device = self._device
        self._volume_registry._device = self._device

        # Depth texture created lazily to match canvas size
        self._depth_texture = None
        self._depth_texture_view = None
        self._depth_texture_size = (0, 0)

        # Initialize timing
        self._start_time = perf_counter_ns()
        self._last_time = self._start_time
        self.elapsed = 0.0

    # === Frame Rendering (shared by run() and render()) ===

    def _draw_frame(self):
        """Draw a single frame. Backend-agnostic - used by both run() and render()."""
        # Check if we should stop (only for run() with event loop)
        if not self._running:
            return False  # Stop the event loop

        # Double-check canvas state
        try:
            if self._render_canvas.get_closed():
                return False
        except Exception:
            return False

        dt = self._compute_dt()
        self._last_dt = dt

        # Step 2: resolve frame waiters (tick / delay / elapsed_at)
        self._frame_waiters.resolve(self.elapsed)

        # Clear command buffer ONCE at the head of the frame so events,
        # async handlers, and systems all contribute to the same buffer
        # that gets flushed at step 6.
        self.commands.clear()

        # Step 3: drain pending events (frame N-1's emits + this frame's 'frame')
        frame_payload = {
            "dt": dt,
            "elapsed": self.elapsed,
            "frame": self._frame_index,
        }
        # Inject the inline 'frame' event ahead of the user-emitted queue,
        # so frame handlers see CURRENT-frame data (not last-frame's).
        self._event_bus._pending.insert(0, ("frame", frame_payload))
        self._event_bus.dispatch_pending(self)

        # Step 4: pump asyncio loop (async handlers + waiter wakers).
        self._pump_aio_loop()

        # Step 5: run user systems (may emit commands).
        self.systems.run_all(self, dt)

        # Step 6: flush command buffer (events + handlers + systems).
        self.commands.execute(self.store)

        # Step 7: GPU compute systems.
        self._compute_runner.run_all(dt)
        self._frame_index += 1

        # Step 8: render pipeline.
        self._render_pipeline.run(self, dt)

        # Ensure render pipeline is initialized
        self._render_pipeline._ensure_pipeline(self._device, self._texture_format)

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

        return True  # Continue rendering

    def _pump_aio_loop(self) -> None:
        """Drive the engine's asyncio loop until quiescent.

        Runs all currently-runnable callbacks: woken futures from waiter
        resolution, freshly-scheduled handler tasks, and I/O completions.
        If any async handler raised an exception during this pump, the
        first one is re-raised so the frame loop crashes per the v1
        error policy. (Done tasks are no longer in asyncio.all_tasks(),
        so we capture exceptions via add_done_callback in _invoke_async
        and spool them on engine._task_errors.)
        """
        self._aio_loop.run_until_complete(asyncio.sleep(0))
        if self._task_errors:
            exc = self._task_errors.pop(0)
            raise exc

    def _run_loop(self):
        """Called each frame by the event loop (run() mode)."""
        self._draw_frame()
        # Re-queue for next frame if still running
        if self._running:
            self._render_canvas.request_draw(self._run_loop)

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
        canvas = get_desktop_canvas(
            width=self.w,
            height=self.h,
            fullscreen=self.fullscreen,
            title=self.title,
        )

        # Initialize WebGPU context
        self._init_canvas(canvas)

        self._running = True

        # Fire built-in 'startup' event before the first frame.
        self._event_bus.dispatch_immediate(self, "startup", {})

        # Register draw callback and run the canvas event loop
        from rendercanvas.glfw import loop as glfw_loop

        self._event_loop = glfw_loop
        self._render_canvas.request_draw(self._run_loop)
        try:
            glfw_loop.run()
        except Exception as e:
            print(f"Event loop error: {e}")
        finally:
            self._running = False
            self.shutdown_events()

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

        # Use fixed timestep for video rendering
        dt = 1.0 / fps

        # Import backends module for lazy canvas creation
        from manifoldx.backends import get_offscreen_canvas

        # Create offscreen canvas for headless rendering
        canvas = get_offscreen_canvas(width=self.w, height=self.h)

        # Initialize WebGPU context
        self._init_canvas(canvas)

        self._running = True

        # Set up video writer with imageio-ffmpeg
        writer = self._get_video_writer(output, fps, self.w, self.h, codec, quality)

        # Fire built-in 'startup' event before the first frame.
        self._event_bus.dispatch_immediate(self, "startup", {})

        # Progress tracking
        pbar = None
        if progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=frame_count, desc="Rendering video", unit="frames")
            except ImportError:
                pass

        # === RENDER LOOP ===
        for frame_idx in range(frame_count):
            # Draw the frame (backend-agnostic)
            self._draw_frame()

            # Capture frame from canvas
            frame = self._render_canvas.draw()

            # Convert RGBA to RGB (most video codecs don't support alpha)
            frame_rgb = frame[:, :, :3].copy()  # Make contiguous

            # Write to video
            writer.send(frame_rgb)

            # Update progress
            if pbar:
                pbar.update(1)

            # Update elapsed time
            self.elapsed = (frame_idx + 1) * dt

        # Clean up
        writer.close()
        writer = None  # type: ignore
        if pbar:
            pbar.close()

        self._running = False

        # 'shutdown' dispatch + asyncio teardown wired in Task 8.

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

    def cli(
        self,
        *,
        fps: int = 30,
        duration: float = 60,
        output: str | None = None,
        quality: str = "high",
    ) -> None:
        """Command-line interface for the engine.

        Usage:
            python example.py           # Interactive window
            python example.py --render  # Render 60s video
            python example.py --render --fps 60 --duration 120
            python example.py --render --output custom.mp4

        Parameters
        ----------
        fps : int
            Frames per second (default: 30).
        duration : float
            Video duration in seconds (default: 60).
        output : str | None
            Output filename. If None, inferred from script name (default).
        quality : str
            Video quality: "low", "medium", "high" (default: "high").
        """
        parser = argparse.ArgumentParser(description=self.title)
        parser.add_argument(
            "--render",
            action="store_true",
            help="Render to video instead of showing window",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=fps,
            help=f"Frames per second (default: {fps})",
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=duration,
            help=f"Video duration in seconds (default: {duration})",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output filename (default: <script>.mp4)",
        )
        parser.add_argument(
            "--quality",
            type=str,
            default=quality,
            choices=["low", "medium", "high"],
            help=f"Video quality (default: {quality})",
        )

        args = parser.parse_args()

        if args.render:
            # Infer output filename from script name
            if args.output is None:
                script_path = Path(sys.argv[0]).resolve()
                output_path = script_path.with_suffix(".mp4")
            else:
                output_path = args.output

            self.render(
                output=str(output_path),
                fps=args.fps,
                duration=args.duration,
                quality=args.quality,
            )
        else:
            self.run()
