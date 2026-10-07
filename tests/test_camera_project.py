"""Camera.project: world points to pixel coordinates and depth."""
import pytest

from manifoldx.camera import Camera


def _cam():
    return Camera(position=(0, 0, 5), target=(0, 0, 0), fov=90)


def test_origin_projects_to_centre_at_its_distance():
    xy, depth = _cam().project([(0, 0, 0)], 100, 100)
    assert xy[0] == pytest.approx((50, 50))
    assert depth[0] == pytest.approx(5)


def test_right_and_up_follow_screen_axes():
    # fov 90 -> focal 1; a point 1 m right at 5 m depth sits at ndc 0.2
    xy, _ = _cam().project([(1, 0, 0), (0, 1, 0)], 100, 100)
    assert xy[0] == pytest.approx((60, 50))
    assert xy[1] == pytest.approx((50, 40))  # y grows downward in pixels


def test_aspect_scales_x_only():
    xy, _ = _cam().project([(1, 0, 0)], 200, 100)
    assert xy[0] == pytest.approx((100 + 0.2 / 2 * 100, 50))


def test_accepts_a_single_point():
    xy, depth = _cam().project((0, 0, 0), 10, 10)
    assert xy.shape == (1, 2) and depth.shape == (1,)
