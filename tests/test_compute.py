"""Unit tests for the Compute base class and marker types.

Phase 1 of the compute-systems design — class-with-annotations declarative
shape. No GPU here; integration tests live in test_compute_integration.py.
"""
import numpy as np
import pytest


# --- Marker types -----------------------------------------------------------


def test_marker_types_are_subscriptable():
    """Reads / Writes / ReadsWrites / Uniform support [X] subscripting."""
    from manifoldx.compute import Reads, ReadsWrites, Uniform, Writes

    # Subscripting any marker just stores the parameter for later inspection.
    r = Reads[int]
    w = Writes[int]
    rw = ReadsWrites[int]
    u = Uniform[float]
    assert r is not None
    assert w is not None
    assert rw is not None
    assert u is not None


def test_compute_base_class_collects_annotations():
    """Subclassing Compute walks annotations into _reads, _writes, _uniforms."""
    from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform

    class MyCompute(Compute):
        positions: Reads[int]              # placeholder; real test uses Component fields
        velocities: ReadsWrites[int]
        G: Uniform[float] = 1.0
        dt: Uniform[float] = "frame_dt"

    # Annotations get walked into structured lists at subclass time.
    assert "positions" in MyCompute._reads
    assert "velocities" in MyCompute._writes
    assert "G" in MyCompute._uniforms
    assert "dt" in MyCompute._uniforms

    # Defaults survived.
    assert MyCompute._uniform_defaults["G"] == 1.0
    assert MyCompute._uniform_defaults["dt"] == "frame_dt"


def test_compute_workgroup_size_and_dispatch_defaults():
    """workgroup_size and dispatch fall back to documented defaults."""
    from manifoldx.compute import Compute, Reads

    class MyCompute(Compute):
        x: Reads[int]

    assert MyCompute.workgroup_size == 64       # documented default
    assert MyCompute.dispatch == "entity_count"  # documented default


def test_compute_compile_must_be_implemented():
    """Subclasses must override compile(); the base raises NotImplementedError."""
    from manifoldx.compute import Compute, Reads

    class IncompleteCompute(Compute):
        x: Reads[int]

    with pytest.raises(NotImplementedError):
        IncompleteCompute().compile()


def test_compute_compile_returns_string_when_overridden():
    from manifoldx.compute import Compute, Reads

    class MyCompute(Compute):
        x: Reads[int]

        def compile(self) -> str:
            return "@compute @workgroup_size(64) fn main() {}"

    assert "compute" in MyCompute().compile()


# --- Bind group layout ------------------------------------------------------


def test_bind_group_layout_orders_reads_then_writes():
    """Bindings: 0 = uniform; 1..K = Reads (decl order); K+1.. = Writes."""
    from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform, Writes

    class MyCompute(Compute):
        a: Reads[int]
        b: Reads[int]
        c: Writes[int]
        d: ReadsWrites[int]
        e: Uniform[float] = 0.0

    layout = MyCompute._bind_group_layout()
    # Uniform always at 0.
    assert layout[0]["name"] == "_uniforms"
    # Reads at 1, 2 in declaration order.
    assert layout[1]["name"] == "a"
    assert layout[1]["access"] == "read"
    assert layout[2]["name"] == "b"
    # Writes / ReadsWrites at 3, 4.
    assert layout[3]["name"] == "c"
    assert layout[3]["access"] == "read_write"
    assert layout[4]["name"] == "d"
    assert layout[4]["access"] == "read_write"


# --- Auto-bound uniform / dispatch symbols ----------------------------------


def test_auto_bound_uniform_resolution():
    """String uniform defaults like 'frame_dt' resolve via the engine each frame."""
    from manifoldx.compute import _AUTO_BOUND_UNIFORMS

    # The registry must include the documented sentinels.
    assert "frame_dt" in _AUTO_BOUND_UNIFORMS
    assert "entity_count" in _AUTO_BOUND_UNIFORMS
    assert "frame_index" in _AUTO_BOUND_UNIFORMS


def test_dispatch_symbol_resolution():
    """String dispatch values like 'entity_count' resolve via the same engine state."""
    from manifoldx.compute import _DISPATCH_SYMBOLS

    assert "entity_count" in _DISPATCH_SYMBOLS


# --- Component gpu_only flag ------------------------------------------------


def test_component_gpu_only_flag():
    """Components subclassing with gpu_only=True carry the flag at the class level."""
    from manifoldx.components import Component
    from manifoldx.types import Vector3

    class GpuVel(Component, gpu_only=True):
        vector: Vector3

    assert getattr(GpuVel, "_gpu_only", False) is True

    class CpuVel(Component):
        vector: Vector3

    assert getattr(CpuVel, "_gpu_only", False) is False
