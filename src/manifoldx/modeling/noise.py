"""Seeded, deterministic value/gradient-noise fields for displacement.

A field is a callable: points (K, 3) -> values (K,) float32 in ~[-1, 1].
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _resolve_rng(seed) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def perlin(seed=None, freq: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Classic Perlin gradient noise in 3D, seeded via a permutation table."""
    rng = _resolve_rng(seed)
    perm = rng.permutation(256).astype(np.int32)
    perm = np.concatenate([perm, perm])  # doubled to avoid overflow indexing

    # 12 canonical gradient directions.
    grad3 = np.array(
        [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
         [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
         [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]],
        dtype=np.float32,
    )

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def grad(ix, iy, iz, dx, dy, dz):
        h = perm[perm[perm[ix & 255] + (iy & 255)] + (iz & 255)] % 12
        g = grad3[h]
        return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

    def field(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64) * freq
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        xi, yi, zi = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), np.floor(z).astype(np.int32)
        xf, yf, zf = x - xi, y - yi, z - zi
        u, v, w = fade(xf), fade(yf), fade(zf)

        def lerp(a, b, t):
            return a + t * (b - a)

        n000 = grad(xi, yi, zi, xf, yf, zf)
        n100 = grad(xi + 1, yi, zi, xf - 1, yf, zf)
        n010 = grad(xi, yi + 1, zi, xf, yf - 1, zf)
        n110 = grad(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf)
        n001 = grad(xi, yi, zi + 1, xf, yf, zf - 1)
        n101 = grad(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1)
        n011 = grad(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1)
        n111 = grad(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1)

        x00 = lerp(n000, n100, u)
        x10 = lerp(n010, n110, u)
        x01 = lerp(n001, n101, u)
        x11 = lerp(n011, n111, u)
        y0 = lerp(x00, x10, v)
        y1 = lerp(x01, x11, v)
        return lerp(y0, y1, w).astype(np.float32)

    return field


def fbm(seed=None, freq: float = 1.0, octaves: int = 4,
        lacunarity: float = 2.0, gain: float = 0.5) -> Callable[[np.ndarray], np.ndarray]:
    """Fractal Brownian motion: summed octaves of `perlin`."""
    rng = _resolve_rng(seed)
    layers = [(perlin(seed=rng, freq=freq * lacunarity**i), gain**i) for i in range(octaves)]
    norm = sum(a for _, a in layers)

    def field(points: np.ndarray) -> np.ndarray:
        total = np.zeros(len(points), dtype=np.float32)
        for f, amp in layers:
            total += amp * f(points)
        return (total / norm).astype(np.float32)

    return field
