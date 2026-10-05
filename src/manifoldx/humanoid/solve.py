"""Forward kinematics: pose + proportions -> a grounded figure of a given height."""
import math
from dataclasses import dataclass

import numpy as np

from . import _quat as q
from .pose import Pose
from .proportions import Proportions
from .skeleton import Joint, build_parts, build_skeleton, ellipsoid_half_extent


@dataclass(frozen=True)
class Part:
    kind: str
    joint: str
    center: np.ndarray
    rotation: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class PosedFigure:
    joints: dict
    rotations: dict
    points: dict
    parts: list
    bounds: tuple
    height: float


def facing(origin, target) -> float:
    """Yaw in degrees that turns a figure at `origin` to face `target`.

    2-sequences are (x, z); 3-sequences are (x, y, z).
    """
    return math.degrees(math.atan2(target[0] - origin[0], target[-1] - origin[-1]))


def _local_rotation(joint: Joint, angles: dict) -> np.ndarray:
    side = joint.side or 1
    r = q.axis_angle((0, 0, 1), angles.get("abduct", 0.0) * side + angles.get("lean", 0.0))
    r = q.mul(r, q.axis_angle((1, 0, 0), angles.get("flex", 0.0) * joint.flex_sign))
    return q.mul(r, q.axis_angle((0, 1, 0), angles.get("twist", 0.0) * side))


def solve(pose, proportions: Proportions, height: float = 1.75, yaw: float = 0.0) -> PosedFigure:
    pose = Pose.parse(pose)
    d = proportions.dims
    skeleton, specs = build_skeleton(d), build_parts(d)
    root = q.axis_angle((0, 1, 0), yaw)

    pos, rot = {}, {}
    for name, j in skeleton.items():
        local = _local_rotation(j, pose.angles_for(name))
        if j.parent is None:
            pos[name], rot[name] = np.array(j.offset, dtype=np.float64), q.mul(root, local)
        else:
            pos[name] = pos[j.parent] + q.rotate(rot[j.parent], j.offset)
            rot[name] = q.mul(rot[j.parent], local)

    parts = [Part(s.kind, name, pos[name] + q.rotate(rot[name], s.center), rot[name], np.array(s.radii))
             for name, ss in specs.items() for s in ss]
    head = next(p for p in parts if p.kind == "head")
    nose = next(p for p in parts if p.kind == "nose")
    points = {"head": head.center, "nose": nose.center}
    for s in ("l", "r"):
        points[f"hand_{s}"] = pos[f"wrist_{s}"] + q.rotate(rot[f"wrist_{s}"], (0.0, -d["hand"], 0.0))

    ext = [ellipsoid_half_extent(q.to_matrix(p.rotation), p.radii) for p in parts]
    lo = np.min([p.center - e for p, e in zip(parts, ext)], axis=0)
    hi = np.max([p.center + e for p, e in zip(parts, ext)], axis=0)
    # Dims are fractions of stature, so the stature scale is `height` whatever
    # the pose; a seated figure is shorter than its stature. Grounding only shifts.
    shift = np.array([0.0, -lo[1], 0.0])
    k = height

    def place(v):
        return (np.asarray(v) + shift) * k

    return PosedFigure(
        joints={n: place(v) for n, v in pos.items()},
        rotations=rot,
        points={n: place(v) for n, v in points.items()},
        parts=[Part(p.kind, p.joint, place(p.center), p.rotation, p.radii * k) for p in parts],
        bounds=(place(lo), place(hi)),
        height=height,
    )
