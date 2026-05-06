"""Compute systems — first-class GPU work as ECS extension.

Phase 1 of the compute-systems design. Users subclass `Compute`, declare
component bindings via class-level annotations, and override `compile()`
to return raw WGSL. The engine extracts the bind-group layout from the
annotations, compiles the shader once, and dispatches the compute pipeline
each frame as part of the run loop.

Phase 2 (separate spec) will add a Python-as-shader DSL on top of this:
the user overrides `def main(self, i)` with traced Python, and the base
class's default `compile()` transpiles `main` to WGSL. The Phase-1 API
shape (class shape, annotations, marker types, bind layout) is identical
between phases — only the kernel-body language differs.

Spec: `.knowledge/analysis/2026-05-06-compute-systems-design.md`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np


# =============================================================================
# Marker types
# =============================================================================
#
# Reads[X] / Writes[X] / ReadsWrites[X] / Uniform[T] are subscriptable type
# markers used in Compute subclass annotations. Subscripting just stashes the
# parameter; Compute.__init_subclass__ walks the annotations and records the
# binding direction. The engine never instantiates these — they are pure
# class-level metadata.


class _MarkerMeta(type):
    """Metaclass so subclasses get a working __class_getitem__."""

    def __getitem__(cls, parameter):
        # Return a parameterized marker. The class identity is what matters
        # for direction (Reads / Writes / ReadsWrites / Uniform); the
        # parameter is preserved so a future code-generator can read it.
        return _ParameterizedMarker(cls, parameter)


class _ParameterizedMarker:
    __slots__ = ("base", "parameter")

    def __init__(self, base, parameter):
        self.base = base
        self.parameter = parameter

    def __repr__(self):
        return f"{self.base.__name__}[{self.parameter!r}]"


class Reads(metaclass=_MarkerMeta):
    """Marker for read-only component bindings (`storage<read>`)."""


class Writes(metaclass=_MarkerMeta):
    """Marker for write-only component bindings (`storage<read_write>`).

    The shader is expected to write each entity's slot. Since wgpu doesn't
    distinguish read-only-write from read-write at the binding level,
    `Writes` and `ReadsWrites` map to the same WGSL access mode; the
    distinction is documentation for readers.
    """


class ReadsWrites(metaclass=_MarkerMeta):
    """Marker for read-write component bindings (`storage<read_write>`)."""


class Uniform(metaclass=_MarkerMeta):
    """Marker for scalar uniform parameters (packed into a single uniform buffer).

    Defaults can be:
    - A literal value (constant for the pipeline's lifetime).
    - A sentinel string (`"frame_dt"`, `"entity_count"`, etc.) — re-uploaded
      each frame, looked up in `_AUTO_BOUND_UNIFORMS`.
    """


# =============================================================================
# Auto-bound uniform / dispatch symbol registries
# =============================================================================


_AUTO_BOUND_UNIFORMS: Dict[str, Callable[[Any], float]] = {
    "frame_dt": lambda engine: float(getattr(engine, "_last_dt", 1 / 60)),
    "entity_count": lambda engine: int(np.sum(engine.store._alive)),
    "frame_index": lambda engine: int(getattr(engine, "_frame_index", 0)),
}


_DISPATCH_SYMBOLS: Dict[str, Callable[[Any], int]] = {
    "entity_count": lambda engine: int(np.sum(engine.store._alive)),
    "max_entities": lambda engine: int(engine.store.max_entities),
}


# =============================================================================
# Compute base class
# =============================================================================


class Compute:
    """Base class for declarative GPU compute systems.

    Subclass and:
    - Declare component bindings via class-level annotations
      (`Reads[X]`, `Writes[X]`, `ReadsWrites[X]`).
    - Declare uniform parameters via `Uniform[T]` annotations (with optional
      class-level defaults, either literals or sentinel strings).
    - Override class-level `workgroup_size: int` (default 64) and
      `dispatch` (default `"entity_count"`).
    - Override `compile()` to return a WGSL string. (Phase 2: override
      `main(self, i)` instead and the base class's default `compile()`
      will transpile it.)

    Then register with the engine: `engine.compute(MyCompute)`.
    """

    workgroup_size: int = 64
    dispatch: Any = "entity_count"  # str symbol | int | callable(engine) → int

    # Populated by __init_subclass__ from the class annotations.
    _reads: Dict[str, Any] = {}
    _writes: Dict[str, Any] = {}
    _uniforms: Dict[str, Any] = {}
    _uniform_defaults: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        reads: Dict[str, Any] = {}
        writes: Dict[str, Any] = {}
        uniforms: Dict[str, Any] = {}
        uniform_defaults: Dict[str, Any] = {}

        annotations = cls.__dict__.get("__annotations__", {})
        for name, ann in annotations.items():
            if name.startswith("_"):
                continue
            if isinstance(ann, _ParameterizedMarker):
                base = ann.base
                if base is Reads:
                    reads[name] = ann.parameter
                elif base is Writes or base is ReadsWrites:
                    writes[name] = ann.parameter
                elif base is Uniform:
                    uniforms[name] = ann.parameter
                    if name in cls.__dict__:
                        uniform_defaults[name] = cls.__dict__[name]
                else:
                    raise TypeError(
                        f"{cls.__name__}: annotation {name!r} uses an "
                        f"unrecognized marker {base.__name__}"
                    )
            else:
                # Non-marker annotations (workgroup_size: int, etc.) — ignore.
                continue

        cls._reads = reads
        cls._writes = writes
        cls._uniforms = uniforms
        cls._uniform_defaults = uniform_defaults

    # --- API ---------------------------------------------------------------

    def compile(self) -> str:
        """Return the WGSL source for this compute kernel.

        Phase 1: subclasses override this to return a raw WGSL string.
        Phase 2 will provide a default implementation that traces
        `self.main` to WGSL automatically.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override compile() with raw WGSL "
            f"source (Phase 1) or override main() and rely on the base-class "
            f"transpiler (Phase 2)."
        )

    # --- Internals (used by the engine) ------------------------------------

    @classmethod
    def _bind_group_layout(cls) -> list[dict]:
        """Compute the bind-group layout from the class annotations.

        Slot 0 is always the packed uniform buffer. Slots 1..K are Reads
        in declaration order; slots K+1.. are Writes / ReadsWrites in
        declaration order. Returns a list of dicts; the engine consumes
        them to build the actual wgpu bind-group layout.
        """
        layout: list[dict] = [
            {
                "binding": 0,
                "name": "_uniforms",
                "kind": "uniform",
                "access": "read",
            }
        ]
        slot = 1
        for name in cls._reads:
            layout.append(
                {"binding": slot, "name": name, "kind": "storage", "access": "read"}
            )
            slot += 1
        for name in cls._writes:
            layout.append(
                {
                    "binding": slot,
                    "name": name,
                    "kind": "storage",
                    "access": "read_write",
                }
            )
            slot += 1
        return layout


__all__ = [
    "Compute",
    "Reads",
    "Writes",
    "ReadsWrites",
    "Uniform",
]
