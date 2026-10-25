"""Compute results at the edges of the store's used slots.

Pins what a dispatch must keep when it moves only part of each component
array: entities in the last used slot, entities respawned into freed slots,
a full store, and `gpu_only` components whose state lives on the GPU.
Skips when no offscreen wgpu backend is available.
"""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Component, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform
from manifoldx.compute.shader import u32, vec3


class SlotVelocity(Component):
    vector: vec3


class SlotTicks(Component, gpu_only=True):
    count: float


class Drift(Compute):
    transforms: ReadsWrites[Transform]
    velocities: ReadsWrites[SlotVelocity]
    n: Uniform[float] = "entity_count"  # type: ignore[assignment]
    workgroup_size = 64
    dispatch = "entity_count"

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos: vec3 = self.transforms[i].pos
        self.transforms[i].pos = pos + self.velocities[i].vector


class Tick(Compute):
    """Counts frames in a gpu_only component and copies the count into pos."""

    transforms: ReadsWrites[Transform]
    ticks: ReadsWrites[SlotTicks]
    n: Uniform[float] = "entity_count"  # type: ignore[assignment]
    workgroup_size = 64
    dispatch = "entity_count"

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        c: float = self.ticks[i].count + 1.0
        self.ticks[i].count = c
        self.transforms[i].pos = vec3(c, c, c)


def _engine(max_entities):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("test", width=64, height=64, max_entities=max_entities)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def _spawn(engine, pos, vel):
    pos = np.asarray(pos, np.float32)
    return engine.spawn(
        Transform(pos=pos),
        SlotVelocity(vector=np.asarray(vel, np.float32)),
        n=len(pos),
    )


def _run(engine, frames=1):
    for _ in range(frames):
        engine._compute_runner.run_all(1 / 30)


def _pos(engine):
    return engine.store._components["Transform"][:, :3]


def test_entity_in_last_used_slot_is_updated_and_read_back():
    engine = _engine(1000)
    _spawn(engine, np.arange(15).reshape(5, 3), np.ones((5, 3)))
    engine.compute(Drift)
    _run(engine, 2)

    np.testing.assert_allclose(_pos(engine)[:5], np.arange(15).reshape(5, 3) + 2.0)
    np.testing.assert_array_equal(_pos(engine)[5:], 0.0)


def test_entities_spawned_after_the_first_frame_are_uploaded():
    engine = _engine(1000)
    _spawn(engine, np.zeros((3, 3)), np.ones((3, 3)))
    engine.compute(Drift)
    _run(engine)
    _spawn(engine, np.full((4, 3), 10.0), np.full((4, 3), 2.0))
    _run(engine)

    np.testing.assert_allclose(_pos(engine)[:3], 2.0)
    np.testing.assert_allclose(_pos(engine)[3:7], 12.0)


def test_respawn_into_freed_slots_uses_the_new_data():
    engine = _engine(1000)
    _spawn(engine, np.zeros((6, 3)), np.ones((6, 3)))
    engine.compute(Drift)
    _run(engine)
    engine.store.destroy(np.array([1, 4]))
    _spawn(engine, np.full((2, 3), 100.0), np.full((2, 3), -1.0))
    np.testing.assert_array_equal(np.flatnonzero(engine.store._alive), np.arange(6))
    _run(engine)

    pos = _pos(engine)
    np.testing.assert_allclose(pos[[0, 2, 3, 5]], 2.0)
    np.testing.assert_allclose(pos[[1, 4]], 99.0)


def test_cpu_writes_between_frames_reach_the_kernel():
    engine = _engine(1000)
    _spawn(engine, np.zeros((4, 3)), np.ones((4, 3)))
    engine.compute(Drift)
    _run(engine)
    engine.store._components["Transform"][2, :3] = 50.0
    _run(engine)

    np.testing.assert_allclose(_pos(engine)[[0, 1, 3]], 2.0)
    np.testing.assert_allclose(_pos(engine)[2], 51.0)


def test_full_store():
    engine = _engine(8)
    _spawn(engine, np.zeros((8, 3)), np.ones((8, 3)))
    engine.compute(Drift)
    _run(engine, 3)

    np.testing.assert_allclose(_pos(engine), 3.0)


def test_gpu_only_component_keeps_state_on_the_gpu():
    engine = _engine(1000)
    engine.spawn(Transform(pos=np.zeros((5, 3), np.float32)), SlotTicks(count=0.0), n=5)
    engine.compute(Tick)
    _run(engine, 3)

    np.testing.assert_allclose(_pos(engine)[:5], 3.0)
    # Never read back: the CPU copy keeps what spawn wrote.
    np.testing.assert_array_equal(engine.store._components["SlotTicks"], 0.0)
