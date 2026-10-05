"""Joint table and mannequin volumes derived from a dims table (fractions of stature)."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Joint:
    parent: str | None
    offset: tuple[float, float, float]  # from the parent, rest pose (standing, arms down, facing +Z)
    side: int                           # +1 left (+X), -1 right, 0 centre
    flex_sign: int                      # makes positive flex the anatomical bend


@dataclass(frozen=True)
class PartSpec:
    kind: str                           # torso, limb, joint, head, nose
    center: tuple[float, float, float]  # in the joint's frame
    radii: tuple[float, float, float]   # ellipsoid semi-axes


def build_skeleton(d: dict) -> dict[str, Joint]:
    hip_y, waist_y, neck_y = d["trochanter_height"], d["waist_height"], d["cervicale_height"]
    shoulder_y = d["acromial_height"] - d["biceps_width"] / 2
    j = {
        "pelvis": Joint(None, (0.0, hip_y, 0.0), 0, +1),
        "spine": Joint("pelvis", (0.0, waist_y - hip_y, 0.0), 0, +1),
        "neck": Joint("spine", (0.0, neck_y - waist_y, 0.0), 0, +1),
    }
    for s, side in (("l", 1), ("r", -1)):
        j[f"shoulder_{s}"] = Joint("spine", (side * d["biacromial_breadth"] / 2, shoulder_y - waist_y, 0.0), side, -1)
        j[f"elbow_{s}"] = Joint(f"shoulder_{s}", (0.0, -d["upper_arm"], 0.0), side, -1)
        j[f"wrist_{s}"] = Joint(f"elbow_{s}", (0.0, -d["forearm"], 0.0), side, -1)
        j[f"hip_{s}"] = Joint("pelvis", (side * d["hip_breadth"] / 4, 0.0, 0.0), side, -1)
        j[f"knee_{s}"] = Joint(f"hip_{s}", (0.0, -(hip_y - d["knee_height"]), 0.0), side, +1)
        j[f"ankle_{s}"] = Joint(f"knee_{s}", (0.0, -(d["knee_height"] - d["ankle_height"]), 0.0), side, -1)
    return j


def build_parts(d: dict) -> dict[str, list[PartSpec]]:
    """Rounded volumes per joint. Limb ellipsoids overrun their bone by ~10% so joints close."""
    hip_y, waist_y, neck_y = d["trochanter_height"], d["waist_height"], d["cervicale_height"]
    crotch, axilla, acr = d["crotch_height"], d["axilla_height"], d["acromial_height"]
    hh = d["head_height"]
    chin = 1.0 - hh
    mid = (waist_y + axilla) / 2
    chest_w = max(d["chest_breadth"], 0.85 * d["biacromial_breadth"])
    head_c = (1.0 - hh / 2) - neck_y
    head_depth = 0.45 * hh

    def limb(length, width, depth=None):
        return PartSpec("limb", (0.0, -length / 2, 0.0), (width / 2, length / 2 * 1.1, (depth or width) / 2))

    parts = {
        "pelvis": [PartSpec("torso", (0.0, (crotch + waist_y) / 2 - hip_y, 0.0),
                            (d["hip_breadth"] / 2, (waist_y - crotch) / 2 * 1.3, d["hip_breadth"] * 0.32))],
        "spine": [
            PartSpec("torso", (0.0, (waist_y + mid) / 2 - waist_y, 0.0),
                     (d["waist_breadth"] / 2, (mid - waist_y) / 2 * 1.3, d["waist_breadth"] * 0.32)),
            PartSpec("torso", (0.0, (mid + acr) / 2 - waist_y, 0.0),
                     (chest_w / 2, (acr - mid) / 2 * 1.3, chest_w * 0.34)),
        ],
        "neck": [
            PartSpec("limb", (0.0, (chin - neck_y) / 2 + 0.01, 0.0),
                     (d["neck_width"] / 2, max(chin - neck_y, 0.01) / 2 * 1.4 + 0.01, d["neck_width"] / 2)),
            PartSpec("head", (0.0, head_c, 0.0), (d["head_breadth"] / 2, hh / 2, head_depth)),
            PartSpec("nose", (0.0, head_c - 0.05 * hh, head_depth * 0.95), (0.05 * hh,) * 3),
        ],
    }
    for s in ("l", "r"):
        parts[f"shoulder_{s}"] = [PartSpec("joint", (0.0, 0.0, 0.0), (d["biceps_width"] * 0.58,) * 3),
                                  limb(d["upper_arm"], d["biceps_width"])]
        parts[f"elbow_{s}"] = [PartSpec("joint", (0.0, 0.0, 0.0), (d["forearm_width"] / 2,) * 3),
                               limb(d["forearm"], (d["forearm_width"] + d["wrist_width"]) / 2)]
        parts[f"wrist_{s}"] = [limb(d["hand"], d["hand_breadth"], d["hand_breadth"] * 0.45)]
        parts[f"hip_{s}"] = [limb(hip_y - d["knee_height"], d["hip_breadth"] / 2)]
        parts[f"knee_{s}"] = [PartSpec("joint", (0.0, 0.0, 0.0), (d["knee_width"] / 2,) * 3),
                              limb(d["knee_height"] - d["ankle_height"], d["calf_width"])]
        parts[f"ankle_{s}"] = [PartSpec("limb", (0.0, -d["ankle_height"] / 2, d["foot_length"] * 0.3),
                                        (d["foot_breadth"] / 2, d["ankle_height"] / 2, d["foot_length"] / 2))]
    return parts


def ellipsoid_half_extent(rotation_matrix: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """World-axis half extents of a rotated ellipsoid (exact AABB)."""
    return np.sqrt(((rotation_matrix * radii[None, :]) ** 2).sum(axis=1))
