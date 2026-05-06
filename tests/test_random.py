"""Tests for manifoldx.random — initial-condition generators."""
import numpy as np
import pytest

import manifoldx as mx


# --- Positions --------------------------------------------------------------


def test_positions_uniform_shape_and_dtype():
    pts = mx.random.positions_uniform(100, low=-1.0, high=1.0, rng=7)
    assert pts.shape == (100, 3)
    assert pts.dtype == np.float32


def test_positions_uniform_respects_bounds():
    pts = mx.random.positions_uniform(500, low=-2.0, high=3.0, rng=7)
    assert pts.min() >= -2.0
    assert pts.max() <= 3.0


def test_positions_uniform_per_axis_bounds():
    """low/high may be (3,) tuples for per-axis bounds."""
    pts = mx.random.positions_uniform(
        500, low=(-1.0, 0.0, -10.0), high=(1.0, 5.0, -5.0), rng=7
    )
    assert pts[:, 0].min() >= -1.0 and pts[:, 0].max() <= 1.0
    assert pts[:, 1].min() >= 0.0 and pts[:, 1].max() <= 5.0
    assert pts[:, 2].min() >= -10.0 and pts[:, 2].max() <= -5.0


def test_positions_in_box_alias():
    """positions_in_box(n, half) === positions_uniform(n, -half, +half)."""
    a = mx.random.positions_in_box(100, half_size=2.5, rng=7)
    b = mx.random.positions_uniform(100, low=-2.5, high=2.5, rng=7)
    np.testing.assert_array_equal(a, b)


def test_positions_in_sphere_inside_radius():
    pts = mx.random.positions_in_sphere(1000, radius=3.0, rng=7)
    assert pts.shape == (1000, 3)
    assert pts.dtype == np.float32
    norms = np.linalg.norm(pts, axis=1)
    assert norms.max() <= 3.0


def test_positions_on_sphere_at_radius():
    pts = mx.random.positions_on_sphere(500, radius=2.0, rng=7)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 2.0, atol=1e-5)


def test_positions_in_disk_radial_bounds():
    pts = mx.random.positions_in_disk(
        1000, inner=2.0, outer=5.0, thickness=0.0, axis="y", rng=7
    )
    # In-plane radius (xz-plane when axis="y").
    r_xz = np.linalg.norm(pts[:, [0, 2]], axis=1)
    assert r_xz.min() >= 2.0 - 1e-5
    assert r_xz.max() <= 5.0 + 1e-5
    # Thickness 0 → all points on the y=0 plane.
    np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-5)


def test_positions_in_disk_thickness():
    pts = mx.random.positions_in_disk(
        1000, inner=1.0, outer=2.0, thickness=0.3, axis="y", rng=7
    )
    # |y| should rarely exceed thickness (gaussian, so ±3σ ≈ ±0.9).
    assert np.abs(pts[:, 1]).max() < 1.5


def test_positions_gaussian_mean_and_sigma():
    pts = mx.random.positions_gaussian(10_000, sigma=2.0, rng=7)
    assert abs(pts.mean()) < 0.1
    np.testing.assert_allclose(pts.std(), 2.0, atol=0.1)


# --- Velocities -------------------------------------------------------------


def test_velocities_gaussian_shape_and_stats():
    v = mx.random.velocities_gaussian(10_000, sigma=1.5, rng=7)
    assert v.shape == (10_000, 3)
    assert v.dtype == np.float32
    np.testing.assert_allclose(v.std(), 1.5, atol=0.05)


def test_velocities_on_sphere_have_target_speed():
    v = mx.random.velocities_on_sphere(500, speed=4.0, rng=7)
    speeds = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(speeds, 4.0, atol=1e-4)


def test_velocities_tangent_perpendicular_to_position_and_axis():
    pos = mx.random.positions_in_sphere(200, radius=5.0, rng=7)
    v = mx.random.velocities_tangent(pos, axis=(0, 1, 0), speed=2.0)
    # v is perpendicular to both pos (within the axis-plane) and the axis.
    # We require: v ⋅ axis ≈ 0  (everywhere)
    dots_axis = (v * np.array([0, 1, 0], dtype=np.float32)).sum(axis=1)
    np.testing.assert_allclose(dots_axis, 0.0, atol=1e-4)
    # And the speed matches.
    speeds = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(speeds, 2.0, atol=1e-4)


def test_velocities_orbit_circular_speed_matches_keplerian():
    """For a circular orbit at radius r around mass GM,
    v = sqrt(GM / r). Verify."""
    pos = mx.random.positions_in_disk(
        500, inner=2.0, outer=8.0, thickness=0.0, axis="y", rng=7
    )
    GM = 60.0
    v = mx.random.velocities_orbit(pos, GM=GM, axis=(0, 1, 0))
    r = np.linalg.norm(pos[:, [0, 2]], axis=1)
    expected_speed = np.sqrt(GM / r)
    actual_speed = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(actual_speed, expected_speed, rtol=1e-4)


# --- Scalars ----------------------------------------------------------------


def test_scalars_uniform_shape_and_bounds():
    s = mx.random.scalars_uniform(1000, low=0.5, high=3.0, rng=7)
    assert s.shape == (1000,)
    assert s.dtype == np.float32
    assert s.min() >= 0.5 and s.max() <= 3.0


def test_scalars_gaussian_stats():
    s = mx.random.scalars_gaussian(10_000, sigma=2.0, mean=1.0, rng=7)
    assert s.shape == (10_000,)
    assert s.dtype == np.float32
    assert abs(s.mean() - 1.0) < 0.1
    np.testing.assert_allclose(s.std(), 2.0, atol=0.1)


# --- RNG resolution ---------------------------------------------------------


def test_rng_int_seed_is_deterministic():
    a = mx.random.positions_uniform(50, low=-1, high=1, rng=42)
    b = mx.random.positions_uniform(50, low=-1, high=1, rng=42)
    np.testing.assert_array_equal(a, b)


def test_rng_passed_generator_is_consumed_in_order():
    rng = np.random.default_rng(11)
    a = mx.random.positions_uniform(10, low=-1, high=1, rng=rng)
    b = mx.random.positions_uniform(10, low=-1, high=1, rng=rng)
    # Two consecutive draws from the same generator must differ.
    assert not np.array_equal(a, b)


def test_rng_none_defaults_to_fresh_generator():
    """rng=None calls default_rng() each time — results differ run-to-run."""
    a = mx.random.positions_uniform(10, low=-1, high=1, rng=None)
    b = mx.random.positions_uniform(10, low=-1, high=1, rng=None)
    # Statistically vanishingly unlikely to collide.
    assert not np.array_equal(a, b)
