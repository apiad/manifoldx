"""Pure-numpy tests for light-space shadow matrices (no GPU)."""

import numpy as np

from manifoldx.shadow import compute_light_view_proj


def _project(M, p):
    v = M @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
    return v[:3] / v[3]


def test_target_maps_near_ndc_center():
    M = compute_light_view_proj(
        direction=(0, -1, 0), target=(0, 0, 0), extent=10.0, near=0.1, far=50.0
    )
    ndc = _project(M, (0, 0, 0))
    assert abs(ndc[0]) < 1e-4 and abs(ndc[1]) < 1e-4
    assert 0.0 <= ndc[2] <= 1.0


def test_point_inside_extent_is_in_ndc_range():
    M = compute_light_view_proj(
        direction=(0, -1, 0), target=(0, 0, 0), extent=10.0, near=0.1, far=50.0
    )
    ndc = _project(M, (5.0, 0.0, -5.0))
    assert -1.0 <= ndc[0] <= 1.0
    assert -1.0 <= ndc[1] <= 1.0
    assert 0.0 <= ndc[2] <= 1.0


def test_point_outside_extent_is_out_of_range():
    M = compute_light_view_proj(
        direction=(0, -1, 0), target=(0, 0, 0), extent=10.0, near=0.1, far=50.0
    )
    ndc = _project(M, (50.0, 0.0, 0.0))
    assert abs(ndc[0]) > 1.0
