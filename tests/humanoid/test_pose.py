"""Poses: library, base plus overrides, both-side shorthands, clear errors."""
import pytest

from manifoldx.humanoid.pose import JOINTS, POSES, Pose


def test_library_names_parse():
    for name in POSES:
        Pose.parse(name)


def test_shorthand_applies_to_both_sides():
    p = Pose.parse({"shoulders": {"abduct": 90}})
    assert p.angles_for("shoulder_l") == {"abduct": 90.0}
    assert p.angles_for("shoulder_r") == {"abduct": 90.0}


def test_overrides_merge_per_angle_on_top_of_the_base():
    p = Pose.parse({"base": "stand", "shoulder_r": {"flex": 70}})
    assert p.angles_for("shoulder_r") == {"abduct": 6.0, "flex": 70.0}
    assert p.angles_for("shoulder_l") == {"abduct": 6.0}


def test_unmentioned_joints_have_no_angles():
    assert Pose.parse("t").angles_for("knee_l") == {}


@pytest.mark.parametrize("spec, bad", [
    ({"sholder_r": {"flex": 10}}, "sholder_r"),
    ({"elbow_l": {"flexx": 10}}, "flexx"),
    ("dab", "dab"),
])
def test_typos_fail_naming_the_bad_key(spec, bad):
    with pytest.raises(ValueError, match=bad):
        Pose.parse(spec)


def test_joint_order_has_parents_first():
    assert JOINTS[0] == "pelvis" and JOINTS.index("spine") < JOINTS.index("shoulder_l")
