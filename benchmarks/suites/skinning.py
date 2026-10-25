"""Posing many skinned entities every frame."""

import numpy as np

from benchmarks import scenes
from benchmarks.harness import case, measure

JOINTS = 20  # a humanoid's order of magnitude


@case("skinning", "set_joints_each", sizes=[100, 1_000, 5_000], smoke=4)
def set_joints_each(n):
    """n EntityHandle.set_joints calls of 20 joints, as a crowd poses each frame."""
    from manifoldx.engine import EntityHandle

    engine, first = scenes.skinned(n)
    engine._ensure_offscreen(1)
    handles = [EntityHandle(engine, first.index + k, first._geometry_id) for k in range(n)]
    mats = np.tile(np.eye(4, dtype=np.float32), (n, JOINTS, 1, 1))
    for h, m in zip(handles, mats):
        h.set_joints(m)
    engine._draw_frame()  # creates the joint buffer, so later sets upload

    def run():
        for h, m in zip(handles, mats):
            h.set_joints(m)

    return measure(run)


@case("skinning", "set_joints_batched", sizes=[100, 1_000, 5_000], smoke=4)
def set_joints_batched(n):
    """One engine.set_joints call posing n entities of 20 joints each."""
    engine, first = scenes.skinned(n)
    engine._ensure_offscreen(1)
    indices = np.arange(first.index, first.index + n)
    mats = np.tile(np.eye(4, dtype=np.float32), (n, JOINTS, 1, 1))
    engine.set_joints(indices, mats)
    engine._draw_frame()  # creates the joint buffer, so later sets upload
    return measure(lambda: engine.set_joints(indices, mats))
