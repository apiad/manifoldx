"""Gradient: a composable scalar -> RGB color ramp (piecewise-linear, clamped)."""

from __future__ import annotations

import numpy as np


def _to_rgb(color) -> tuple[float, float, float]:
    if isinstance(color, str):
        h = color.lstrip("#")
        if len(h) == 3:                       # shorthand #rgb -> #rrggbb
            h = "".join(ch * 2 for ch in h)
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    return (float(color[0]), float(color[1]), float(color[2]))


class Gradient:
    """Maps scalar values to RGB by linear interpolation between color stops.

    stops: [(position, color), ...] with color a "#rrggbb" string or (r, g, b) in [0, 1].
    Values below the first / above the last stop clamp to the endpoint colors.
    """

    def __init__(self, stops):
        parsed = sorted(((float(p), _to_rgb(c)) for p, c in stops), key=lambda s: s[0])
        self._pos = np.array([p for p, _ in parsed], dtype=np.float64)
        self._col = np.array([c for _, c in parsed], dtype=np.float32)  # (S, 3)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        out = np.empty((len(v), 3), dtype=np.float32)
        for k in range(3):
            out[:, k] = np.interp(v, self._pos, self._col[:, k])  # np.interp clamps at the ends
        return out
