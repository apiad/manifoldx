"""Proportions: ANSUR II means and stylised presets."""
import math

import pytest

from manifoldx.humanoid.proportions import BODIES, HEAD_HEIGHT, PRESETS, Proportions


@pytest.mark.parametrize("body", ["male", "female"])
def test_measured_reproduces_the_ansur_table(body):
    d = Proportions.measured(body).dims
    for key in ("acromial_height", "trochanter_height", "knee_height", "hip_breadth", "biceps_width"):
        assert d[key] == pytest.approx(BODIES[body][key])
    assert d["head_height"] == pytest.approx(HEAD_HEIGHT)


@pytest.mark.parametrize("body", ["male", "female"])
def test_measured_arms_reach_the_measured_span(body):
    d = Proportions.measured(body).dims
    reach = d["biacromial_breadth"] / 2 + d["upper_arm"] + d["forearm"] + d["hand"]
    assert reach == pytest.approx(BODIES[body]["span"] / 2)


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("body", ["male", "female"])
def test_presets_hit_their_head_count(preset, body):
    p = Proportions.styled(body, preset)
    assert p.heads == pytest.approx(PRESETS[preset][body]["heads"])
    assert all(math.isfinite(v) and v > 0 for v in p.dims.values())


def test_unknown_names_are_rejected_with_the_options():
    with pytest.raises(ValueError, match="male"):
        Proportions.measured("robot")
    with pytest.raises(ValueError, match="chibi"):
        Proportions.styled("male", "manga")
