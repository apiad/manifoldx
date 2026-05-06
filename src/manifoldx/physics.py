"""Vectorized simulation primitives for ECS demos.

Compresses the all-pairs-distance / gravity / boundary / collision
patterns that the imperative physics examples (nbody, gas, boids,
point_cloud_demo) reimplement by hand. All functions operate on
numpy arrays — no engine coupling, no ECS — so they're equally
useful inside `@engine.system` callbacks and outside (data prep,
notebooks, tests).

Conventions:
- `positions` / `velocities` are `(N, 3)` float32 arrays.
- All-pairs `diff[i, j]` is `pos[j] - pos[i]` (force from j on i
  points along `+diff[i, j]`).
- Boundary and collision helpers MUTATE `velocities` in place to
  match the conventional physics-loop pattern (`velocities[walls] *= -1`).
  They return None.
- Gravity helpers RETURN a new `(N, 3)` acceleration array; the caller
  integrates: `velocities += accel * dt`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# =============================================================================
# All-pairs primitives
# =============================================================================


@dataclass
class Pairs:
    """All-pairs differences and distances.

    Fields:
    - `diff[i, j] = positions[j] - positions[i]`, shape (N, N, 3).
    - `dist[i, j] = ||diff[i, j]||`, shape (N, N).
    - `dist_safe`: max(dist, softening); equal to `dist` when softening=0.
    """

    diff: np.ndarray
    dist: np.ndarray
    dist_safe: np.ndarray


def all_pairs(positions: np.ndarray, *, softening: float = 0.0) -> Pairs:
    """Compute all-pairs differences and distances for N positions.

    Returns a `Pairs` dataclass — useful when you need both the diff
    tensor and the scalar distance grid (most physics interactions).
    """
    pos = np.asarray(positions, dtype=np.float32)
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    if softening > 0.0:
        dist_safe = np.maximum(dist, softening)
    else:
        dist_safe = dist
    return Pairs(diff=diff, dist=dist, dist_safe=dist_safe)


# =============================================================================
# Gravity
# =============================================================================


def gravity(
    positions: np.ndarray,
    *,
    masses: Optional[np.ndarray] = None,
    G: float = 1.0,
    softening: float = 0.0,
) -> np.ndarray:
    """All-pairs Newtonian gravity. Returns acceleration `(N, 3)`.

    Acceleration on body i is sum over j != i of:
        a_i += G * m_j * (pos[j] - pos[i]) / (||r||² + ε²)^(3/2)

    `masses` defaults to ones (test-mass interpretation). If supplied,
    the returned acceleration on body i already accounts for the inertia
    of body i (so the caller doesn't need to divide by m_i).
    """
    pos = np.asarray(positions, dtype=np.float32)
    n = pos.shape[0]
    if masses is None:
        masses_arr = np.ones(n, dtype=np.float32)
    else:
        masses_arr = np.asarray(masses, dtype=np.float32)

    pairs = all_pairs(pos, softening=softening)
    diff = pairs.diff  # (N, N, 3)
    dist_safe = pairs.dist_safe  # (N, N)

    # Avoid division by zero on the diagonal — set diagonal to 1 before
    # the divide so we don't generate inf, then zero it out so it doesn't
    # contribute to the sum.
    dist_cubed = dist_safe * dist_safe * dist_safe
    np.fill_diagonal(dist_cubed, 1.0)
    inv_r3 = 1.0 / dist_cubed
    np.fill_diagonal(inv_r3, 0.0)
    # Acceleration on i from j: G * m_j * diff[i, j] * inv_r3[i, j].
    weight = G * masses_arr[None, :] * inv_r3  # (N, N)
    accel = (weight[:, :, None] * diff).sum(axis=1)
    return accel.astype(np.float32)


def central_gravity(
    positions: np.ndarray,
    *,
    GM: float,
    softening: float = 0.0,
    center: tuple = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Acceleration toward a single central mass at `center`.

    `a = -GM * (pos - center) / (||r||² + ε²)^(3/2)`.

    Faster than `gravity(...)` for the common single-source case
    (e.g. protoplanetary disks, satellites) since it skips the
    all-pairs tensor.
    """
    pos = np.asarray(positions, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32)
    r_vec = pos - c[None, :]
    r2 = (r_vec * r_vec).sum(axis=1) + softening * softening
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    return (-GM * r_vec * inv_r3[:, None]).astype(np.float32)


# =============================================================================
# Boundaries
# =============================================================================


def box_boundary(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    half_size: float,
    dt: float = 0.0,
    mode: str = "reflect",
) -> None:
    """Reflect velocities at the walls of a cube `[-half_size, +half_size]^3`.

    Predicts the next position (`positions + velocities * dt`) and flips
    the corresponding velocity component for any axis that would step
    past a wall. Mutates `velocities` in place; returns None.

    `dt=0` checks the current position rather than the predicted next one.
    """
    if mode != "reflect":
        raise ValueError(f"box_boundary mode must be 'reflect'; got {mode!r}")
    next_pos = positions + velocities * dt
    lo = -half_size
    hi = half_size
    velocities[next_pos < lo] = np.abs(velocities[next_pos < lo])
    velocities[next_pos > hi] = -np.abs(velocities[next_pos > hi])


def sphere_boundary(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    radius: float,
    mode: str = "reflect",
    strength: float = 1.0,
    dt: float = 0.0,
) -> None:
    """Confine particles inside a sphere of given radius.

    `mode="reflect"`: hard reflection — radial velocity component
    flipped for particles whose predicted next position is outside the
    sphere.
    `mode="soft"`: continuous inward acceleration ∝ overshoot for
    particles outside the radius. `strength` scales the inward pull.
    Both modes mutate `velocities` in place.
    """
    if mode == "reflect":
        next_pos = positions + velocities * dt
        next_r = np.linalg.norm(next_pos, axis=1, keepdims=True)
        outside = (next_r > radius).flatten()
        if not outside.any():
            return
        # Outward radial direction at each outside particle's current pos.
        cur_r = np.linalg.norm(positions, axis=1, keepdims=True)
        radial = positions / np.maximum(cur_r, 1e-9)
        # Reflect: subtract twice the radial velocity component.
        v_radial = (velocities[outside] * radial[outside]).sum(axis=1, keepdims=True)
        velocities[outside] -= 2.0 * v_radial * radial[outside]
    elif mode == "soft":
        cur_r = np.linalg.norm(positions, axis=1, keepdims=True)
        overshoot = np.maximum(cur_r - radius, 0.0)
        # Inward pull: -position direction × overshoot × strength.
        inward = -positions / np.maximum(cur_r, 1e-9)
        velocities += (inward * overshoot * strength * max(dt, 1.0)).astype(np.float32)
    else:
        raise ValueError(
            f"sphere_boundary mode must be 'reflect' or 'soft'; got {mode!r}"
        )


# =============================================================================
# Elastic collisions
# =============================================================================


def elastic_collisions(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    radius: float,
    restitution: float = 1.0,
) -> None:
    """Resolve elastic pair collisions for equal-mass particles of given radius.

    Detects overlapping pairs (dist < 2 * radius), computes impulse along
    the collision normal, applies it symmetrically — only for pairs that
    are actually approaching. `restitution=1.0` is a perfectly elastic
    bounce; `0.0` is fully inelastic (relative normal velocity zeroed).
    Mutates `velocities` in place; returns None.
    """
    pos = np.asarray(positions, dtype=np.float32)
    pairs = all_pairs(pos)
    overlap = pairs.dist < (2.0 * radius)
    np.fill_diagonal(overlap, False)
    i_idx, j_idx = np.where(np.triu(overlap))
    if i_idx.size == 0:
        return

    n_vec = pairs.diff[i_idx, j_idx]
    n_dist = pairs.dist[i_idx, j_idx, None]
    n_hat = np.where(n_dist > 1e-6, n_vec / n_dist, 0.0)
    v_rel = velocities[j_idx] - velocities[i_idx]
    v_dot_n = (v_rel * n_hat).sum(axis=1)

    approaching = v_dot_n < 0.0
    if not approaching.any():
        return

    idx_i = i_idx[approaching]
    idx_j = j_idx[approaching]
    v_n = v_dot_n[approaching]
    n_h = n_hat[approaching]

    # Impulse magnitude for equal-mass: -(1+e)/2 * v_rel·n along the normal.
    impulse = -(0.5 * (1.0 + restitution)) * v_n[:, None] * n_h
    np.add.at(velocities, idx_i, -impulse)
    np.add.at(velocities, idx_j, impulse)


__all__ = [
    "Pairs",
    "all_pairs",
    "gravity",
    "central_gravity",
    "box_boundary",
    "sphere_boundary",
    "elastic_collisions",
]
