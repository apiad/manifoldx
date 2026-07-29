"""Reusable sky helpers for space scenes."""

from __future__ import annotations

import numpy as np

from manifoldx import random as _random
from manifoldx.components import Transform, Material
from manifoldx.viz import PointCloud, ColormapMaterial, ScalarValue, Radius


def starfield(engine, count: int = 1500, radius: float = 400.0, seed: int = 42):
    """Spawn `count` unlit star points on a sky sphere of `radius`.

    Uses the point-sprite path (grey colormap, per-star brightness + size), so it
    is a cheap static backdrop for any space scene. Returns the entity handle.
    """
    pos = _random.positions_on_sphere(count, radius=radius, rng=seed)
    bright = _random.scalars_uniform(count, low=0.45, high=1.0, rng=seed + 1)
    radii = _random.scalars_uniform(count, low=radius * 0.0010, high=radius * 0.0028, rng=seed + 2)
    return engine.spawn(
        PointCloud(),
        Material(ColormapMaterial(cmap="gray", vmin=0.0, vmax=1.0, lit=False)),
        Transform(pos=pos.astype(np.float32)),
        ScalarValue(value=bright),
        Radius(radius=radii),
        n=count,
    )
