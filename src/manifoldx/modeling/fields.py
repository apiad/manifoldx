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
