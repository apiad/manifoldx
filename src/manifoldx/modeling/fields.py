"""Composable scalar-field algebra: a fluent Field type + noise/pattern sources.

A Field wraps (points (K,3)) -> (K,) float32 and is itself callable, so any
consumer that samples a field (e.g. Mesh.displace) accepts a Field unchanged.
"""

from __future__ import annotations

import numpy as np


def _resolve_rng(seed) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class Field:
    """A composable scalar field over 3-D space."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return np.asarray(self._fn(p), dtype=np.float32)

    # --- arithmetic ---
    def __add__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) + o(p))

    __radd__ = __add__

    def __sub__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) - o(p))

    def __rsub__(self, o):
        o = _as_field(o)
        return Field(lambda p: o(p) - self(p))

    def __mul__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) * o(p))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _as_field(o)
        return Field(lambda p: self(p) / o(p))

    def __rtruediv__(self, o):
        o = _as_field(o)
        return Field(lambda p: o(p) / self(p))

    def __neg__(self):
        return Field(lambda p: -self(p))

    # --- combinators ---
    def mix(self, other, t):
        other, tf = _as_field(other), _as_field(t)
        return Field(lambda p: self(p) * (1.0 - tf(p)) + other(p) * tf(p))

    def minimum(self, other):
        other = _as_field(other)
        return Field(lambda p: np.minimum(self(p), other(p)))

    def maximum(self, other):
        other = _as_field(other)
        return Field(lambda p: np.maximum(self(p), other(p)))

    def clamp(self, lo, hi):
        return Field(lambda p: np.clip(self(p), lo, hi))

    def remap(self, a, b, c, d):
        return Field(lambda p: c + (self(p) - a) * (d - c) / (b - a))

    def abs(self):
        return Field(lambda p: np.abs(self(p)))

    def power(self, n):
        return Field(lambda p: np.power(self(p), n))

    def scale(self, s):
        return self * s

    def bias(self, b):
        return self + b

    def warp(self, amount, fx=None, fy=None, fz=None):
        fx = _as_field(0.0 if fx is None else fx)
        fy = _as_field(0.0 if fy is None else fy)
        fz = _as_field(0.0 if fz is None else fz)

        def fn(p):
            offset = np.stack([fx(p), fy(p), fz(p)], axis=1) * amount
            return self(p + offset)

        return Field(fn)


def _as_field(x) -> Field:
    if isinstance(x, Field):
        return x
    v = float(x)
    return Field(lambda p, v=v: np.full(len(p), v, dtype=np.float32))


# =============================================================================
# Noise sources
# =============================================================================


def perlin(seed=None, freq: float = 1.0) -> Field:
    """Classic Perlin gradient noise in 3D, seeded via a permutation table."""
    rng = _resolve_rng(seed)
    perm = rng.permutation(256).astype(np.int32)
    perm = np.concatenate([perm, perm])  # doubled to avoid overflow indexing

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

    return Field(field)


def fbm(seed=None, freq: float = 1.0, octaves: int = 4,
        lacunarity: float = 2.0, gain: float = 0.5) -> Field:
    """Fractal Brownian motion: summed octaves of `perlin`."""
    rng = _resolve_rng(seed)
    layers = [(perlin(seed=rng, freq=freq * lacunarity**i), gain**i) for i in range(octaves)]
    norm = sum(a for _, a in layers)

    def field(points: np.ndarray) -> np.ndarray:
        total = np.zeros(len(points), dtype=np.float32)
        for f, amp in layers:
            total += amp * f(points)
        return (total / norm).astype(np.float32)

    return Field(field)
