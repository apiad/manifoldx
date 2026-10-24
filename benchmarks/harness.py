"""Timing helpers and the case registry shared by every suite.

A case is a function of one size `n` that returns milliseconds, or a dict with
"ms" and any extra numbers worth printing (calls per frame, bytes moved). Cases
register with `@case(suite, name, sizes=..., smoke=...)`; `smoke` is the tiny
size the test suite runs so cases do not rot.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
import timeit
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

CASES: dict[str, list["Case"]] = {}


@dataclass
class Case:
    suite: str
    name: str
    fn: Callable[[int], float | dict]
    sizes: list[int]
    smoke: int
    doc: str = ""


@dataclass
class Result:
    suite: str
    case: str
    n: int
    ms: float | None
    extra: dict = field(default_factory=dict)
    error: str | None = None


def case(suite: str, name: str, *, sizes: list[int], smoke: int):
    """Register `fn(n)` as case `name` of `suite`, timed at each of `sizes`."""

    def wrap(fn):
        CASES.setdefault(suite, []).append(
            Case(suite, name, fn, list(sizes), smoke, (fn.__doc__ or "").strip().split("\n")[0])
        )
        return fn

    return wrap


# Repeats of `measure`; the test suite lowers them to 1 so smoke runs stay quick.
REPEAT = 5


def measure(fn: Callable[[], object], repeat: int | None = None, number: int | None = None) -> float:
    """Best of `repeat` runs, in ms per call. Each run makes `number` calls, picked so
    a run lasts at least 0.2 s (at most 20 calls) when not given."""
    repeat = REPEAT if repeat is None else repeat
    if number is None:
        number = max(1, min(timeit.Timer(fn).autorange()[0], 20)) if REPEAT > 1 else 1
    return 1e3 * min(timeit.repeat(fn, number=number, repeat=repeat)) / number


def headless_engine(max_entities: int, size: int = 64):
    """An Engine for timing: small target, so the software rasterizer of a GPU-less
    machine adds little to the CPU-side cost being measured."""
    import manifoldx as mx

    return mx.Engine("bench", width=size, height=size, max_entities=max(max_entities, 16))


def start(engine) -> None:
    """Give a populated engine its offscreen canvas and a fixed timestep."""
    engine._ensure_offscreen(1)
    engine.set_fixed_timestep(1 / 30)


def frame_ms(engine, frames: int = 5, warmup: int = 2) -> float:
    """Median CPU time of Engine._draw_frame (simulation, render prep, encoding,
    submit) in ms. canvas.draw(), which waits for the GPU and reads back, runs
    between frames and is not counted."""
    for _ in range(warmup):
        engine._draw_frame()
        engine._render_canvas.draw()
    times = []
    for _ in range(frames if REPEAT > 1 else 1):
        t0 = time.perf_counter()
        engine._draw_frame()
        times.append(time.perf_counter() - t0)
        engine._render_canvas.draw()
    return 1e3 * float(np.median(times))


@contextmanager
def count_calls(owner, name: str):
    """Count calls of `owner.name` (a function or method) inside the block.
    Yields a dict with "calls"; a missing attribute counts nothing."""
    counter = {"calls": 0}
    original = getattr(owner, name, None)
    if original is None:
        yield counter
        return

    def counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    setattr(owner, name, counted)
    try:
        yield counter
    finally:
        setattr(owner, name, original)


def profile(fn: Callable[[], object], top: int = 15) -> str:
    """cProfile one call of `fn`; the `top` functions by own time."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("tottime").print_stats(top)
    return out.getvalue()
