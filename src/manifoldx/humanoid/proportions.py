"""Body proportions as fractions of stature: ANSUR II means plus stylised presets.

Heights are above the floor. ANSUR II: US Army anthropometric survey, 2012
(4,082 men, 1,986 women; tools.openlab.psu.edu/publicData). Head height, vertex
to chin, is 0.130 of stature (Drillis and Contini 1966); ANSUR II does not
measure it. Limb widths are circumference / pi.
"""
from dataclasses import dataclass, field

HEAD_HEIGHT = 0.130

BODIES: dict[str, dict[str, float]] = {
    "male": {
        "cervicale_height": 0.864, "acromial_height": 0.820, "axilla_height": 0.757,
        "waist_height": 0.602, "trochanter_height": 0.513, "crotch_height": 0.482,
        "knee_height": 0.280, "ankle_height": 0.042,
        "head_breadth": 0.088, "neck_width": 0.072, "biacromial_breadth": 0.237,
        "bideltoid_breadth": 0.291, "chest_breadth": 0.165, "waist_breadth": 0.186,
        "hip_breadth": 0.197, "span": 1.033,
        "upper_arm": 0.191, "forearm": 0.153, "hand": 0.110,
        "hand_breadth": 0.050, "foot_breadth": 0.058, "foot_length": 0.154,
        "biceps_width": 0.065, "forearm_width": 0.056, "wrist_width": 0.032,
        "knee_width": 0.074, "calf_width": 0.071, "ankle_width": 0.042,
    },
    "female": {
        "cervicale_height": 0.857, "acromial_height": 0.820, "axilla_height": 0.761,
        "waist_height": 0.602, "trochanter_height": 0.519, "crotch_height": 0.480,
        "knee_height": 0.286, "ankle_height": 0.039,
        "head_breadth": 0.091, "neck_width": 0.065, "biacromial_breadth": 0.224,
        "bideltoid_breadth": 0.277, "chest_breadth": 0.165, "waist_breadth": 0.184,
        "hip_breadth": 0.217, "span": 1.020,
        "upper_arm": 0.191, "forearm": 0.148, "hand": 0.111,
        "hand_breadth": 0.048, "foot_breadth": 0.057, "foot_length": 0.151,
        "biceps_width": 0.060, "forearm_width": 0.052, "wrist_width": 0.030,
        "knee_width": 0.078, "calf_width": 0.073, "ankle_width": 0.042,
    },
}

# Style presets on top of the measured body. Widths are in head heights, as
# figure-drawing canons give them; a missing key keeps the measured value.
#   heads        total height in head heights
#   crotch       crotch height, fraction of stature
#   shoulders    outer shoulder width (bideltoid), heads
#   waist, hips  breadths, heads
#   limbs        limb and neck thickness, times measured
#   head_ratio   head width / head height
#   extremities  hand and foot size, times measured
#   reach        hanging fingertips, fraction down the leg from the crotch
#                (measured: about 0.19; negative is above the crotch)
# Head counts follow published canons (Loomis's 8-head ideal, 8.5-9 heroic,
# 9-10 fashion, 7-8 anime, 2-3 chibi, about 6 heads at six years, 4-5 for a
# toddler). The disney column and widths outside the canons are estimates.
_KIDS = {"heads": 6.0, "crotch": 0.44, "shoulders": 1.7, "waist": 1.3, "hips": 1.3,
         "limbs": 1.05, "head_ratio": 0.82}
PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "heroic": {
        "male": {"heads": 8.5, "crotch": 0.50, "shoulders": 2.8, "waist": 1.25, "hips": 1.5,
                 "limbs": 1.15, "head_ratio": 0.70, "extremities": 1.05},
        "female": {"heads": 8.5, "crotch": 0.50, "shoulders": 2.1, "waist": 1.0, "hips": 1.75,
                   "limbs": 0.95, "head_ratio": 0.70},
    },
    "disney": {
        "male": {"heads": 7.0, "crotch": 0.48, "shoulders": 3.0, "waist": 1.6, "hips": 1.5,
                 "limbs": 1.1, "head_ratio": 0.80, "extremities": 1.1},
        "female": {"heads": 6.5, "crotch": 0.48, "shoulders": 1.9, "waist": 0.85, "hips": 1.7,
                   "limbs": 0.8, "head_ratio": 0.80, "extremities": 0.8},
    },
    "anime": {
        "male": {"heads": 7.5, "crotch": 0.52, "shoulders": 2.2, "waist": 1.2, "hips": 1.4,
                 "limbs": 0.8, "head_ratio": 0.78, "extremities": 0.9},
        "female": {"heads": 7.5, "crotch": 0.52, "shoulders": 1.8, "waist": 0.95, "hips": 1.6,
                   "limbs": 0.75, "head_ratio": 0.78, "extremities": 0.85},
    },
    "fashion": {
        "male": {"heads": 9.0, "crotch": 0.54, "shoulders": 2.3, "waist": 1.2, "hips": 1.4,
                 "limbs": 0.8, "head_ratio": 0.68},
        "female": {"heads": 9.0, "crotch": 0.54, "shoulders": 1.9, "waist": 1.0, "hips": 1.5,
                   "limbs": 0.75, "head_ratio": 0.68, "extremities": 0.9},
    },
    "child": {"male": _KIDS, "female": _KIDS},
    "toddler": {s: {"heads": 4.5, "crotch": 0.38, "shoulders": 1.5, "waist": 1.3, "hips": 1.25,
                    "limbs": 1.4, "head_ratio": 0.88, "extremities": 1.1, "reach": 0.0}
                for s in ("male", "female")},
    "chibi": {s: {"heads": 2.5, "crotch": 0.30, "shoulders": 1.3, "waist": 1.1, "hips": 1.1,
                  "limbs": 1.8, "head_ratio": 0.95, "extremities": 1.2, "reach": -0.15}
              for s in ("male", "female")},
}

_TORSO_HEIGHTS = ("cervicale_height", "acromial_height", "axilla_height", "waist_height",
                  "trochanter_height")
_LEG_HEIGHTS = ("knee_height", "ankle_height")
_LIMB_WIDTHS = ("neck_width", "biceps_width", "forearm_width", "wrist_width", "knee_width",
                "calf_width", "ankle_width")


def _dims(body: str, style: dict) -> dict[str, float]:
    """Measured dimensions, re-proportioned by a style (an empty style is the identity).

    Heights between chin and crotch stretch linearly to the new torso, heights
    below the crotch to the new legs. Arms are resized so the hanging
    fingertips keep their measured place on the leg (or the style's `reach`).
    """
    b = dict(BODIES[body], head_height=HEAD_HEIGHT)
    # Summed end to end the ANSUR II arm segments overshoot the measured span
    # by about 11%; scale them so the T-pose reaches it.
    k = (b["span"] / 2 - b["biacromial_breadth"] / 2) / (b["upper_arm"] + b["forearm"] + b["hand"])
    for seg in ("upper_arm", "forearm", "hand"):
        b[seg] *= k

    arm0 = b["upper_arm"] + b["forearm"] + b["hand"]
    root0 = 1 - b["acromial_height"] + b["biceps_width"] / 2
    leg0 = b["crotch_height"]
    fingertip = (root0 + arm0 - (1 - leg0)) / leg0

    h0, h = HEAD_HEIGHT, 1 / style.get("heads", 1 / HEAD_HEIGHT)
    c0, c = b["crotch_height"], style.get("crotch", b["crotch_height"])
    torso = ((1 - c) - h) / ((1 - c0) - h0)
    for key in _TORSO_HEIGHTS:
        b[key] = 1 - (h + ((1 - b[key]) - h0) * torso)
    for key in _LEG_HEIGHTS:
        b[key] *= c / c0
    b["crotch_height"], b["head_height"] = c, h

    b["head_breadth"] = style.get("head_ratio", b["head_breadth"] / h0) * h
    if "shoulders" in style:
        widen = style["shoulders"] * h / b["bideltoid_breadth"]
        for key in ("biacromial_breadth", "bideltoid_breadth", "chest_breadth"):
            b[key] *= widen
    if "waist" in style:
        b["waist_breadth"] = style["waist"] * h
    if "hips" in style:
        b["hip_breadth"] = style["hips"] * h
    for key in _LIMB_WIDTHS:
        b[key] *= style.get("limbs", 1.0)
    root = 1 - b["acromial_height"] + b["biceps_width"] / 2
    arm = (1 - c) + style.get("reach", fingertip) * c - root
    for seg in ("upper_arm", "forearm", "hand"):
        b[seg] *= arm / arm0
    ext = style.get("extremities", 1.0)
    b["hand"] *= ext
    for key in ("hand_breadth", "foot_breadth", "foot_length"):
        b[key] *= ext
    return b


@dataclass(frozen=True)
class Proportions:
    body: str
    preset: str | None
    dims: dict = field(compare=False, repr=False)

    @classmethod
    def measured(cls, body: str = "male") -> "Proportions":
        _check(body, BODIES, "body")
        return cls(body, None, _dims(body, {}))

    @classmethod
    def styled(cls, body: str, preset: str) -> "Proportions":
        _check(body, BODIES, "body")
        _check(preset, PRESETS, "preset")
        return cls(body, preset, _dims(body, PRESETS[preset][body]))

    @property
    def heads(self) -> float:
        return 1 / self.dims["head_height"]


def _check(name, table, what):
    if name not in table:
        raise ValueError(f"unknown {what} {name!r}; available: {', '.join(sorted(table))}")
