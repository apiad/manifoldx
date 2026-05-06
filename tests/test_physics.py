"""Tests for manifoldx.physics — vectorized simulation primitives."""
import numpy as np
import pytest

import manifoldx as mx


# --- all_pairs --------------------------------------------------------------


def test_all_pairs_shape_and_diff_convention():
    """diff[i, j] = pos[j] - pos[i]; dist[i, j] = ||pos[j] - pos[i]||."""
    pos = np.array(
        [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        dtype=np.float32,
    )
    pairs = mx.physics.all_pairs(pos)
    assert pairs.diff.shape == (3, 3, 3)
    assert pairs.dist.shape == (3, 3)
    np.testing.assert_array_equal(pairs.diff[0, 1], np.array([3, 4, 0], dtype=np.float32))
    assert pairs.dist[0, 1] == pytest.approx(5.0)
    assert pairs.dist[0, 2] == pytest.approx(5.0)


def test_all_pairs_softening_floors_dist():
    pos = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]], dtype=np.float32)
    pairs = mx.physics.all_pairs(pos, softening=0.1)
    # Raw distance is 0.001; softened to 0.1.
    assert pairs.dist[0, 1] == pytest.approx(0.001, abs=1e-6)
    assert pairs.dist_safe[0, 1] == pytest.approx(0.1)


# --- gravity ----------------------------------------------------------------


def test_gravity_two_body_attraction():
    """Two equal masses at unit separation feel reciprocal attraction."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    accel = mx.physics.gravity(pos, masses=None, G=1.0, softening=0.0)
    # Unit masses, G=1, r=1 → |a| = 1 on each body, pointing toward the other.
    np.testing.assert_allclose(accel[0], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(accel[1], [-1.0, 0.0, 0.0], atol=1e-5)


def test_gravity_with_masses_proportional():
    """Heavier source pulls the test body harder."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    accel_unit = mx.physics.gravity(pos, masses=np.array([1.0, 1.0], dtype=np.float32))
    accel_heavy = mx.physics.gravity(pos, masses=np.array([1.0, 5.0], dtype=np.float32))
    # Body 0's acceleration scales linearly with body 1's mass.
    np.testing.assert_allclose(accel_heavy[0], accel_unit[0] * 5.0, atol=1e-5)


# --- central_gravity --------------------------------------------------------


def test_central_gravity_inverse_square():
    """A test mass at distance r feels |a| = GM / r² toward origin."""
    pos = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    accel = mx.physics.central_gravity(pos, GM=10.0)
    # |a| = 10 / r², pointing toward origin (negative position direction).
    np.testing.assert_allclose(np.linalg.norm(accel[0], axis=-1), 10.0 / 4.0, rtol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(accel[1], axis=-1), 10.0 / 9.0, rtol=1e-4)
    # Direction toward origin: pos[0] = (2,0,0) → accel[0] = (-, 0, 0).
    assert accel[0, 0] < 0.0
    assert accel[1, 1] < 0.0


def test_central_gravity_softening_caps_max_accel():
    """Softening prevents singularities at the center."""
    pos = np.array([[0.001, 0.0, 0.0]], dtype=np.float32)
    accel = mx.physics.central_gravity(pos, GM=1.0, softening=0.5)
    # Without softening, |a| = 1/r² ≈ 1e6; with softening 0.5,
    # |a| = 1 / (r² + ε²)^(3/2) * r ≈ 1 / (0.25)^(3/2) * 0.001 ≈ 0.008.
    assert np.linalg.norm(accel[0]) < 1e3


# --- box_boundary -----------------------------------------------------------


def test_box_boundary_reflects_outgoing_velocities():
    """Particles moving past a wall have their velocity reflected."""
    positions = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)  # safely inside
    velocities = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)  # heading right
    mx.physics.box_boundary(positions, velocities, half_size=1.0, dt=0.5)
    # next_pos.x would be 5.5, way past the wall at x=1; velocity should flip.
    assert velocities[0, 0] < 0.0


def test_box_boundary_leaves_inside_velocities_alone():
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[0.5, -0.3, 0.2]], dtype=np.float32)
    original = velocities.copy()
    mx.physics.box_boundary(positions, velocities, half_size=1.0, dt=0.1)
    np.testing.assert_array_equal(velocities, original)


# --- sphere_boundary --------------------------------------------------------


def test_sphere_boundary_reflect_pushes_back():
    """A particle outside the sphere gets velocity reflected toward center."""
    positions = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)  # outside radius=1
    velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # heading further out
    mx.physics.sphere_boundary(
        positions, velocities, radius=1.0, mode="reflect", dt=0.1
    )
    # Should have inverted the radial component.
    assert velocities[0, 0] < 0.0


def test_sphere_boundary_soft_adds_inward_acceleration():
    """Soft mode applies an inward force ∝ overshoot, mutating velocities."""
    positions = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    mx.physics.sphere_boundary(
        positions, velocities, radius=1.0, mode="soft", strength=3.0, dt=0.1
    )
    # Inward (toward origin) pull should make x-velocity negative.
    assert velocities[0, 0] < 0.0


# --- elastic_collisions -----------------------------------------------------


def test_elastic_collisions_swap_velocities_for_head_on():
    """Equal-mass head-on collision exchanges velocities along the normal."""
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    mx.physics.elastic_collisions(positions, velocities, radius=0.3)
    # Particles overlapping (dist=0.5 < 2*0.3=0.6) and approaching head-on.
    # Equal-mass elastic collision along the x-axis swaps the x-components.
    np.testing.assert_allclose(velocities[0, 0], -1.0, atol=1e-4)
    np.testing.assert_allclose(velocities[1, 0], 1.0, atol=1e-4)


def test_elastic_collisions_no_overlap_is_noop():
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    original = velocities.copy()
    mx.physics.elastic_collisions(positions, velocities, radius=0.3)
    np.testing.assert_array_equal(velocities, original)


def test_elastic_collisions_separating_pair_is_noop():
    """If two overlapping particles are moving apart, no impulse applies."""
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    original = velocities.copy()
    mx.physics.elastic_collisions(positions, velocities, radius=0.3)
    np.testing.assert_array_equal(velocities, original)
