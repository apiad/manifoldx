"""Forward kinematics: span, symmetry, bone lengths, grounding, presets."""

import numpy as np
import pytest

from manifoldx.humanoid import PRESETS, Proportions
from manifoldx.humanoid.solve import facing, solve


def _male():
    return Proportions.measured("male")


def test_t_pose_fingertips_reach_half_the_span_at_shoulder_height():
    p = _male()
    fig = solve("t", p, height=1.0)
    assert fig.points["hand_l"][0] == pytest.approx(p.dims["span"] / 2, abs=1e-6)
    assert fig.points["hand_r"][0] == pytest.approx(-p.dims["span"] / 2, abs=1e-6)
    assert fig.points["hand_l"][1] == pytest.approx(fig.joints["shoulder_l"][1], abs=1e-6)


def test_left_and_right_mirror_each_other():
    fig = solve("stand", _male(), height=1.75)
    for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        left, right = fig.joints[f"{j}_l"], fig.joints[f"{j}_r"]
        assert left == pytest.approx(right * np.array([-1, 1, 1]), abs=1e-9)


def test_bones_keep_their_lengths_in_any_pose():
    p = _male()
    fig = solve({"base": "hands_on_hips", "spine": {"flex": 20, "lean": 10}}, p, height=1.75)
    upper = np.linalg.norm(fig.joints["elbow_r"] - fig.joints["shoulder_r"])
    assert upper == pytest.approx(p.dims["upper_arm"] * 1.75, rel=1e-6)


def test_figure_stands_on_the_floor_at_its_height():
    fig = solve("stand", _male(), height=1.62)
    lo, hi = fig.bounds
    assert lo[1] == pytest.approx(0, abs=1e-9)
    assert hi[1] == pytest.approx(1.62, abs=1e-9)


def test_seated_pelvis_rests_near_knee_height():
    fig = solve("seated", _male(), height=1.75)
    assert abs(fig.joints["pelvis"][1] - fig.joints["knee_l"][1]) < 0.05 * 1.75


def test_a_seated_figure_is_shorter_than_its_stature():
    # Scaling to the posed bounds would blow a sitting figure up to full height.
    fig = solve("seated", _male(), height=1.75)
    assert fig.bounds[1][1] < 0.8 * 1.75


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("pose", ["stand", "t", "seated"])
def test_every_preset_is_grounded_finite_and_positive(preset, pose):
    fig = solve(pose, Proportions.styled("female", preset), height=1.0)
    lo, hi = fig.bounds
    assert lo[1] == pytest.approx(0, abs=1e-9)
    if pose != "seated":
        assert hi[1] == pytest.approx(1.0, abs=1e-9)  # head top at the stature
    assert all(np.isfinite(v).all() for v in fig.joints.values())
    assert all((part.radii > 0).all() for part in fig.parts)


def test_yaw_turns_the_face():
    fig = solve("stand", _male(), height=1.75, yaw=90)
    assert fig.points["nose"][0] > fig.points["head"][0]  # facing +X


def test_facing_points_at_a_target():
    assert facing((0, 0), (1, 0)) == pytest.approx(90)
    assert facing((0, 0), (0, 1)) == pytest.approx(0)
    assert facing((1, 0, 1), (1, 5, 0)) == pytest.approx(180)

