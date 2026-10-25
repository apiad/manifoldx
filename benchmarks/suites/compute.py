"""GPU compute dispatch cost against store capacity, with a fixed number of live entities."""

import numpy as np

from manifoldx.components import Component, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform
from manifoldx.compute.shader import u32, vec3

from benchmarks.harness import case, headless_engine, measure

ALIVE = 1_000


class BenchVelocity(Component):
    vector: vec3


class BenchDrift(Compute):
    transforms: ReadsWrites[Transform]
    velocities: ReadsWrites[BenchVelocity]
    dt: Uniform[float] = "frame_dt"  # type: ignore[assignment]
    n: Uniform[float] = "entity_count"  # type: ignore[assignment]
    workgroup_size = 64
    dispatch = "entity_count"

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos: vec3 = self.transforms[i].pos
        self.transforms[i].pos = pos + self.velocities[i].vector * self.dt


@case("compute", "dispatch_by_capacity", sizes=[10_000, 100_000, 1_000_000], smoke=64)
def dispatch_by_capacity(n):
    """One kernel dispatch (upload, run, readback) with 1000 live entities and max_entities=n."""
    alive = min(ALIVE, n // 2)
    engine = headless_engine(n)
    engine.spawn(
        Transform(pos=np.zeros((alive, 3), np.float32)),
        BenchVelocity(vector=np.ones((alive, 3), np.float32)),
        n=alive,
    )
    engine.compute(BenchDrift)
    engine._ensure_offscreen(1)
    runner = engine._compute_runner
    runner.run_all(1 / 30)  # compile
    return measure(lambda: runner.run_all(1 / 30))
