"""Altair-style declarative shim layer for sci-viz primitives v1, plan 4.

Wraps the imperative ECS API in a grammar-of-graphics shape so the 80% case
fits in 10 lines:

    chart = (
        mxv.points(positions=p, color=mxv.color(s, cmap="viridis"), size=r)
        + mxv.axes(extent=10)
        + mxv.legend("Speed")
    )

    @chart.simulate
    def step(dt):
        p[:] += v * dt

    chart.cli()

Design: `.knowledge/analysis/2026-05-06-sci-viz-primitives-v1-plan-4-altair-shim-design.md`.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np


# =============================================================================
# Channel objects
# =============================================================================


class Channel:
    """Base for declarative channel wrappers (`mxv.color(...)`, etc.)."""

    data: np.ndarray

    def __init__(self, data):
        self.data = np.asarray(data)


class ColorChannel(Channel):
    """Color encoding: per-instance scalar mapped through a colormap LUT."""

    def __init__(
        self,
        data,
        *,
        cmap: str = "viridis",
        domain: Optional[tuple[float, float]] = None,
        title: str = "",
    ):
        super().__init__(data)
        self.cmap = cmap
        self.domain = domain
        self.title = title


class SizeChannel(Channel):
    """Size encoding: per-instance world-space radius for sprites."""

    def __init__(self, data, *, units: str = "world"):
        super().__init__(data)
        self.units = units


class PositionChannel(Channel):
    """Position encoding: (N, 3) world-space positions."""

    def __init__(self, data):
        super().__init__(data)


def _as_channel(value, cls: type, **defaults) -> Channel:
    """Auto-wrap a bare array in the given channel class with default config."""
    if isinstance(value, Channel):
        return value
    if value is None:
        return None
    return cls(value, **defaults)


# Public factory aliases — `mxv.color(...)`, `mxv.size(...)`, `mxv.position(...)`.
def color(data, **kwargs) -> ColorChannel:
    return ColorChannel(data, **kwargs)


def size(data, **kwargs) -> SizeChannel:
    return SizeChannel(data, **kwargs)


def position(data, **kwargs) -> PositionChannel:
    return PositionChannel(data, **kwargs)


def _infer_domain(arr: np.ndarray) -> tuple[float, float]:
    """Auto-derive (vmin, vmax) from data when the user didn't pin one."""
    if arr.size == 0:
        return (0.0, 1.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        hi = lo + 1.0  # avoid degenerate domain when all values are equal
    return (lo, hi)


# =============================================================================
# Mark base
# =============================================================================


class Mark:
    """Base class for declarative marks. Each mark spawns its entities into
    a target Engine when `apply(engine)` is called by `Chart.build()`."""

    def apply(self, engine) -> None:
        raise NotImplementedError

    def __add__(self, other: "Mark | Chart") -> "Chart":
        chart = Chart()
        chart.marks.append(self)
        if isinstance(other, Chart):
            chart.marks.extend(other.marks)
            chart.simulate_callbacks.extend(other.simulate_callbacks)
        else:
            chart.marks.append(other)
        return chart


# =============================================================================
# Concrete marks
# =============================================================================


class PointsMark(Mark):
    """Sprite point cloud — wraps Plan 1's PointCloud + ColormapMaterial path.

    Per-instance position, color (via colormap), and size (radius). Color
    accepts a bare numpy array (auto-wrapped with cmap='viridis' and a
    domain inferred from the data) or a `mxv.color(...)` channel object
    that pins cmap and domain explicitly.

    The user's arrays are referenced, not copied; per-frame mutation in
    `@chart.simulate` propagates to the GPU because a sync system copies
    the live arrays into ECS storage each frame.
    """

    def __init__(
        self,
        *,
        positions,
        color=None,
        size=None,
    ):
        # Normalize each channel — bare array → wrapped Channel; None stays None.
        self.positions = _as_channel(positions, PositionChannel)
        self.color = _as_channel(color, ColorChannel)
        self.size = _as_channel(size, SizeChannel)

        if self.positions is None:
            raise ValueError("points(...) requires a `positions` channel")

        pos_arr = self.positions.data
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            raise ValueError(
                f"positions must be (N, 3); got shape {pos_arr.shape}"
            )
        self._n = pos_arr.shape[0]

    def apply(self, engine) -> None:
        from manifoldx.components import Material, Transform
        from manifoldx.viz import (
            ColormapMaterial,
            PointCloud,
            Radius,
            ScalarValue,
        )

        n = self._n
        pos_arr = np.asarray(self.positions.data, dtype=np.float32)

        # Color → ColormapMaterial. Domain auto-inferred from data when None.
        if self.color is not None:
            color_arr = np.asarray(self.color.data, dtype=np.float32)
            domain = self.color.domain or _infer_domain(color_arr)
            material = ColormapMaterial(
                cmap=self.color.cmap, vmin=domain[0], vmax=domain[1]
            )
        else:
            color_arr = np.zeros(n, dtype=np.float32)
            material = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)

        # Size → Radius. Defaults to 1.0 when no size channel.
        if self.size is not None:
            size_arr = np.asarray(self.size.data, dtype=np.float32)
        else:
            size_arr = np.full(n, 1.0, dtype=np.float32)

        engine.spawn(
            PointCloud(),
            Material(material),
            Transform(pos=pos_arr),
            ScalarValue(value=color_arr),
            Radius(radius=size_arr),
            n=n,
        )

        # Live-data sync: each frame, copy the user's arrays into ECS
        # component storage so per-frame mutations propagate to the GPU.
        # We capture the arrays + the slice of entity indices via closure.
        first_entity = int(np.argmax(engine.store._alive)) if n > 0 else 0
        # `argmax` returns the first True; this works because entities are
        # spawned contiguously starting at the first free slot. For Plan 4
        # v1 this is sufficient — multi-mark scenes spawn each mark's
        # entities in their own contiguous block.
        entity_slice = slice(first_entity, first_entity + n)

        positions_ref = self.positions.data  # alias the user's array
        color_ref = self.color.data if self.color is not None else None
        size_ref = self.size.data if self.size is not None else None

        store = engine.store

        from manifoldx.systems import Query

        def _sync(query: Query[Transform], dt: float):
            # Copy mutable user arrays into the corresponding component slots.
            store._components["Transform"][entity_slice, 0:3] = positions_ref
            if color_ref is not None:
                store._components["ScalarValue"][entity_slice, 0] = color_ref
            if size_ref is not None:
                store._components["Radius"][entity_slice, 0] = size_ref

        _sync.__name__ = f"_points_sync_{id(self)}"
        engine.system(_sync)


def points(*, positions, color=None, size=None) -> "Chart":
    """Declarative point cloud mark. Returns a Chart with one mark so
    callers can immediately access `.engine`, `.simulate`, `.run()`, etc.
    or compose with `+`."""
    chart = Chart()
    chart.marks.append(PointsMark(positions=positions, color=color, size=size))
    return chart


# =============================================================================
# Chart
# =============================================================================


class Chart:
    """A composed scene of marks + simulation callbacks. Lazy-builds an Engine
    on first terminal call (`run`, `render`, `cli`, or `.engine`)."""

    def __init__(self):
        self.marks: list[Mark] = []
        self.simulate_callbacks: list[Callable] = []
        self._engine = None
        self._title: str = "manifoldx"
        self._width: int = 800
        self._height: int = 600

    # --- composition ---------------------------------------------------------

    def __add__(self, other: "Mark | Chart") -> "Chart":
        out = Chart()
        out.marks.extend(self.marks)
        out.simulate_callbacks.extend(self.simulate_callbacks)
        if isinstance(other, Chart):
            out.marks.extend(other.marks)
            out.simulate_callbacks.extend(other.simulate_callbacks)
        else:
            out.marks.append(other)
        out._title = self._title
        out._width = self._width
        out._height = self._height
        return out

    # --- live data -----------------------------------------------------------

    def simulate(self, func: Callable) -> Callable:
        """Decorator: register a per-frame simulation callback.

        The callback is invoked once per frame with `dt` as its sole argument.
        Multiple callbacks run in registration order before any channel-sync
        systems and before the renderer reads component data.
        """
        self.simulate_callbacks.append(func)
        # If the engine is already built, attach immediately.
        if self._engine is not None:
            self._attach_simulate(func)
        return func

    # --- build / terminal ----------------------------------------------------

    @property
    def engine(self):
        """Lazily build the Engine and return it."""
        if self._engine is None:
            self.build()
        return self._engine

    def build(self):
        """Construct an Engine, apply each mark, wire simulate callbacks."""
        if self._engine is not None:
            return self._engine

        from manifoldx.engine import Engine
        from manifoldx.components import Transform

        engine = Engine(self._title, width=self._width, height=self._height)
        self._engine = engine

        for mark in self.marks:
            mark.apply(engine)

        for cb in self.simulate_callbacks:
            self._attach_simulate(cb)

        return engine

    def _attach_simulate(self, func: Callable) -> None:
        """Wire one user-provided simulate callback into the engine as a system."""
        from manifoldx.systems import Query
        from manifoldx.components import Transform

        # Compile the user's `def step(dt): ...` into an
        # `@engine.system def _wrapper(query, dt): step(dt)` shape so it
        # ticks every frame against the existing system runner.
        engine = self._engine

        def _wrapper(query: Query[Transform], dt: float):
            func(dt)

        # Preserve the original function name for any introspection.
        _wrapper.__name__ = f"_chart_simulate_{func.__name__}"
        engine.system(_wrapper)

    def run(self) -> None:
        """Open an interactive window and start the render loop."""
        self.build()
        self._engine.run()

    def render(
        self,
        output: str = "scene.mp4",
        *,
        duration: float = 60.0,
        fps: int = 30,
        quality: str = "high",
    ) -> None:
        """Render to a video file (offline, no window)."""
        self.build()
        self._engine.render(
            output=output, duration=duration, fps=fps, quality=quality
        )

    def cli(self) -> None:
        """Parse argv and dispatch to run() or render() — same shape as
        the existing `engine.cli()` so demos can drop in this method
        verbatim."""
        self.build()
        self._engine.cli()
