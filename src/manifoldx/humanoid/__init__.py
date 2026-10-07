"""Posable humanoid mannequins: proportions, poses, forward kinematics."""
from .mannequin import spawn_mannequin
from .pose import JOINTS, POSES, Pose
from .proportions import BODIES, HEAD_HEIGHT, PRESETS, Proportions
from .solve import Part, PosedFigure, facing, solve

__all__ = ["BODIES", "HEAD_HEIGHT", "JOINTS", "POSES", "PRESETS", "Part", "Pose", "PosedFigure",
           "Proportions", "facing", "solve", "spawn_mannequin"]
