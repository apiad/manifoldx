"""Poses as anatomical joint angles in degrees.

flex: forward bend (positive is the anatomical bend; knees bend backward).
abduct: outward from the midline, mirrored per side.
twist: rotation about the bone, mirrored per side.
lean: sideways tilt, for pelvis, spine and neck.
"""
from dataclasses import dataclass, field

_SIDED = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
JOINTS: tuple[str, ...] = ("pelvis", "spine", "neck") + tuple(
    f"{j}_{s}" for s in ("l", "r") for j in _SIDED
)
ANGLES = ("flex", "abduct", "twist", "lean")
PAIRS = {f"{j}s": j for j in _SIDED}

POSES: dict[str, dict] = {
    "stand": {"shoulders": {"abduct": 6}, "elbows": {"flex": 10}},
    "t": {"shoulders": {"abduct": 90}},
    "seated": {"hips": {"flex": 90}, "knees": {"flex": 90}},
    "arms_crossed": {"shoulders": {"flex": 30, "abduct": -25}, "elbows": {"flex": 115}},
    "hands_on_hips": {"shoulders": {"abduct": 35, "flex": -10}, "elbows": {"flex": 100},
                      "wrists": {"flex": -20}},
    "hands_behind": {"shoulders": {"flex": -25, "abduct": -5}, "elbows": {"flex": 30}},
    "lean_rail": {"spine": {"flex": 12}, "shoulders": {"flex": 40}, "elbows": {"flex": 50}},
}


def _expand(spec: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, angles in spec.items():
        joints = [f"{PAIRS[key]}_{s}" for s in ("l", "r")] if key in PAIRS else [key]
        if key not in PAIRS and key not in JOINTS:
            raise ValueError(f"unknown joint {key!r}; joints: {', '.join(JOINTS)}; "
                             f"both sides: {', '.join(PAIRS)}")
        for a in angles:
            if a not in ANGLES:
                raise ValueError(f"unknown angle {a!r} on {key!r}; angles: {', '.join(ANGLES)}")
        for j in joints:
            out.setdefault(j, {}).update({a: float(v) for a, v in angles.items()})
    return out


@dataclass(frozen=True)
class Pose:
    angles: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, spec) -> "Pose":
        """A library name, a dict with an optional `base:` plus overrides, or a Pose."""
        if isinstance(spec, Pose):
            return spec
        if isinstance(spec, str):
            spec = {"base": spec}
        spec = dict(spec or {})
        base = spec.pop("base", None)
        merged: dict[str, dict[str, float]] = {}
        if base is not None:
            if base not in POSES:
                raise ValueError(f"unknown pose {base!r}; library: {', '.join(POSES)}")
            merged = _expand(POSES[base])
        for joint, angles in _expand(spec).items():
            merged.setdefault(joint, {}).update(angles)
        return cls(merged)

    def angles_for(self, joint: str) -> dict[str, float]:
        return dict(self.angles.get(joint, {}))
