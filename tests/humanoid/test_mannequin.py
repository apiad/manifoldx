"""A spawned mannequin renders where the solver says it is."""
import pytest

import manifoldx as mx
from manifoldx.humanoid import Proportions, solve
from manifoldx.humanoid.mannequin import spawn_mannequin


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    return mx.Engine("mannequin", width=160, height=120)


def test_spawn_returns_one_entity_per_part():
    e = _engine()
    fig = solve("stand", Proportions.measured("female"), height=1.6)
    ids = spawn_mannequin(e, fig, at=(0.5, 0, -1))
    assert len(ids) == len(fig.parts)


def test_head_projects_onto_the_mannequin_in_the_id_pass():
    e = _engine()
    fig = solve("t", Proportions.styled("male", "heroic"), height=1.8, yaw=30)
    ids = spawn_mannequin(e, fig, at=(0.3, 0, 0))
    e.camera.set_pose(position=(0.3, 1.4, 4.0), target=(0.3, 1.0, 0))
    labels = e.render_frame(supersample=1, pass_="ids", groups=[ids])
    head = fig.points["head"] + (0.3, 0, 0)
    (x, y), = e.camera.project(head, 160, 120)[0]
    assert labels[int(y), int(x)] == 1
    assert labels[2, 2] == 0


@pytest.mark.parametrize("color", ["#cc3333", (0.8, 0.2, 0.2), (0.8, 0.2, 0.2, 1.0)])
def test_spawn_accepts_hex_and_tuple_colours(color):
    # Review F1: a tuple colour used to crash in _shade (no .lstrip on tuple).
    e = _engine()
    fig = solve("stand", Proportions.measured("male"), height=1.7)
    assert len(spawn_mannequin(e, fig, color=color)) == len(fig.parts)
