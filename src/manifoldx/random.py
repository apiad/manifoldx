"""Initial-condition generators for ECS demos.

Compresses the boilerplate that every physics example reimplements:
random positions in a box / ball / disk, random gaussian or unit-speed
velocities, tangent / circular-orbit velocities, scalar attribute draws.

All return float32 numpy arrays of the right shape — `(n, 3)` for
positions and velocities, `(n,)` for scalars. Every function takes an
optional `rng` kwarg that accepts:

- `None` (default): a fresh `numpy.random.default_rng()` per call (entropy-
  seeded, run-to-run non-deterministic).
- An `int`: passed to `default_rng(seed)` for reproducible draws.
- An existing `numpy.random.Generator`: used as-is so multiple draws
  share state (advance the same RNG stream).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np


_AxisName = str
_Axis = Union[_AxisName, Tuple[float, float, float], np.ndarray]


def _resolve_rng(rng) -> np.random.Generator:
    """Normalize the `rng` kwarg into a numpy Generator instance."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _resolve_axis(axis: _Axis) -> np.ndarray:
    """Resolve an axis argument to a unit (3,) float32 vector."""
    if isinstance(axis, str):
        unit = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}.get(axis.lower())
        if unit is None:
            raise ValueError(f"axis name must be 'x', 'y', or 'z'; got {axis!r}")
        return np.asarray(unit, dtype=np.float32)
    a = np.asarray(axis, dtype=np.float32)
    if a.shape != (3,):
        raise ValueError(f"axis must be a length-3 vector; got shape {a.shape}")
    n = np.linalg.norm(a)
    if n < 1e-9:
        raise ValueError("axis must have non-zero magnitude")
    return a / n


# =============================================================================
# Positions
# =============================================================================


def positions_uniform(
    n: int,
    *,
    low: Union[float, Tuple[float, float, float]] = -1.0,
    high: Union[float, Tuple[float, float, float]] = 1.0,
    rng=None,
) -> np.ndarray:
    """N positions uniformly distributed in a box `[low, high]^3`.

    `low` and `high` may each be a scalar (cube) or a length-3 tuple
    (axis-aligned box with per-axis bounds).
    """
    g = _resolve_rng(rng)
    lo = np.asarray(low, dtype=np.float32)
    hi = np.asarray(high, dtype=np.float32)
    if lo.ndim == 0:
        lo = np.full(3, float(lo), dtype=np.float32)
    if hi.ndim == 0:
        hi = np.full(3, float(hi), dtype=np.float32)
    return g.uniform(lo, hi, size=(n, 3)).astype(np.float32)


def positions_in_box(n: int, *, half_size: float, rng=None) -> np.ndarray:
    """N positions uniformly inside a cube of half-edge `half_size`."""
    return positions_uniform(n, low=-half_size, high=half_size, rng=rng)


def positions_in_sphere(n: int, *, radius: float = 1.0, rng=None) -> np.ndarray:
    """N positions uniformly distributed inside a ball of given radius.

    Uses the inverse-cube-root radius trick to get true volumetric uniformity.
    """
    g = _resolve_rng(rng)
    # Direction on unit sphere.
    direction = g.standard_normal((n, 3)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # Radius: cube root of uniform → uniform volume density.
    r = radius * np.cbrt(g.uniform(0, 1, n)).astype(np.float32)
    return direction * r[:, None]


def positions_on_sphere(n: int, *, radius: float = 1.0, rng=None) -> np.ndarray:
    """N positions uniformly on the surface of a sphere of given radius."""
    g = _resolve_rng(rng)
    direction = g.standard_normal((n, 3)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * radius


def positions_in_disk(
    n: int,
    *,
    inner: float = 0.0,
    outer: float = 1.0,
    thickness: float = 0.0,
    axis: _Axis = "y",
    rng=None,
) -> np.ndarray:
    """N positions inside an annular disk in the plane perpendicular to `axis`.

    Surface-density-uniform sampling: r² is drawn uniformly in
    [inner², outer²]. With non-zero `thickness`, perpendicular offset is
    drawn from N(0, thickness).
    """
    g = _resolve_rng(rng)
    axis_unit = _resolve_axis(axis)
    r = np.sqrt(g.uniform(inner**2, outer**2, n)).astype(np.float32)
    theta = g.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
    # Pick two basis vectors perpendicular to the axis.
    u, v = _orthonormal_basis(axis_unit)
    # In-plane coordinates.
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pos = x[:, None] * u[None, :] + y[:, None] * v[None, :]
    if thickness > 0.0:
        z = g.normal(0.0, thickness, n).astype(np.float32)
        pos = pos + z[:, None] * axis_unit[None, :]
    return pos.astype(np.float32)


def positions_gaussian(
    n: int,
    *,
    sigma: float = 1.0,
    mean: float = 0.0,
    rng=None,
) -> np.ndarray:
    """N positions drawn from N(mean, sigma) per axis."""
    g = _resolve_rng(rng)
    return (g.standard_normal((n, 3)) * sigma + mean).astype(np.float32)


# =============================================================================
# Velocities
# =============================================================================


def velocities_gaussian(n: int, *, sigma: float = 1.0, rng=None) -> np.ndarray:
    """N velocities with each component drawn from N(0, sigma)."""
    g = _resolve_rng(rng)
    return (g.standard_normal((n, 3)) * sigma).astype(np.float32)


def velocities_uniform(
    n: int,
    *,
    low: Union[float, Tuple[float, float, float]] = -1.0,
    high: Union[float, Tuple[float, float, float]] = 1.0,
    rng=None,
) -> np.ndarray:
    """N velocities with each component drawn uniformly in [low, high]."""
    return positions_uniform(n, low=low, high=high, rng=rng)


def velocities_on_sphere(n: int, *, speed: float = 1.0, rng=None) -> np.ndarray:
    """N velocities with random direction and given target speed."""
    return positions_on_sphere(n, radius=speed, rng=rng)


def velocities_tangent(
    positions: np.ndarray,
    *,
    axis: _Axis = "y",
    speed: float = 1.0,
) -> np.ndarray:
    """Tangent-direction velocities: `axis × position`, normalized to `speed`.

    Useful for swirl / orbital initial conditions where you want all
    particles moving perpendicular to both their position vector and a
    common axis (e.g. spinning a cloud around the y-axis).

    Particles whose position is collinear with the axis fall back to a
    random direction so the result has no NaNs.
    """
    pos = np.asarray(positions, dtype=np.float32)
    axis_unit = _resolve_axis(axis)
    cross = np.cross(axis_unit[None, :], pos)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    # Fall back to a deterministic perpendicular for collinear cases.
    fallback = _orthonormal_basis(axis_unit)[0][None, :]
    safe = np.where(norms > 1e-6, cross / np.maximum(norms, 1e-9), fallback)
    return (safe * speed).astype(np.float32)


def velocities_orbit(
    positions: np.ndarray,
    *,
    GM: float,
    axis: _Axis = "y",
) -> np.ndarray:
    """Circular-orbit velocities for positions in the plane perpendicular to `axis`.

    For each position, the velocity is tangent to a circular orbit of
    radius `r = ||position||_in_plane` around the origin with speed
    `sqrt(GM / r)` (Keplerian). Useful as initial conditions for n-body
    simulations starting in stable rings.
    """
    pos = np.asarray(positions, dtype=np.float32)
    axis_unit = _resolve_axis(axis)
    # In-plane radius: project pos onto the plane.
    along = (pos * axis_unit[None, :]).sum(axis=1, keepdims=True) * axis_unit[None, :]
    in_plane = pos - along
    r = np.linalg.norm(in_plane, axis=1)
    speeds = np.sqrt(GM / np.maximum(r, 1e-9))
    cross = np.cross(axis_unit[None, :], in_plane)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(norms > 1e-6, cross / np.maximum(norms, 1e-9), 0.0)
    return (safe * speeds[:, None]).astype(np.float32)


# =============================================================================
# Scalars
# =============================================================================


def scalars_uniform(
    n: int, *, low: float = 0.0, high: float = 1.0, rng=None
) -> np.ndarray:
    """N scalars uniformly distributed in [low, high]."""
    g = _resolve_rng(rng)
    return g.uniform(low, high, n).astype(np.float32)


def scalars_gaussian(
    n: int, *, sigma: float = 1.0, mean: float = 0.0, rng=None
) -> np.ndarray:
    """N scalars drawn from N(mean, sigma)."""
    g = _resolve_rng(rng)
    return (g.standard_normal(n) * sigma + mean).astype(np.float32)


# =============================================================================
# Internal helpers
# =============================================================================


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to `axis` and to each other."""
    # Pick any reference that isn't parallel to the axis.
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)
    return u.astype(np.float32), v.astype(np.float32)


__all__ = [
    "positions_uniform",
    "positions_in_box",
    "positions_in_sphere",
    "positions_on_sphere",
    "positions_in_disk",
    "positions_gaussian",
    "velocities_gaussian",
    "velocities_uniform",
    "velocities_on_sphere",
    "velocities_tangent",
    "velocities_orbit",
    "scalars_uniform",
    "scalars_gaussian",
]
