"""Posable humanoid mannequins: proportions, poses, forward kinematics."""
from .pose import JOINTS, POSES, Pose
from .proportions import BODIES, HEAD_HEIGHT, PRESETS, Proportions

__all__ = ["BODIES", "HEAD_HEIGHT", "JOINTS", "POSES", "PRESETS", "Pose", "Proportions"]
