import numpy as np
import manifoldx as mx
from manifoldx.resources import StandardMaterial


def test_fog_in_standard_shader():
    src = StandardMaterial._compile()
    assert "fog_enabled" in src and "fog_color" in src
    assert "distance(globals.camera_pos, in.world_pos)" in src


def test_fog_inherited_by_vcolor_variant():
    assert "fog_enabled" in StandardMaterial._compile(vertex_colors=True)


def test_enable_fog_sets_params():
    engine = mx.Engine("fog")
    engine.background_color = (0.7, 0.8, 0.9)
    engine.enable_fog(10.0, 50.0)
    assert engine.fog_enabled and engine.fog_start == 10.0 and engine.fog_end == 50.0
    assert np.allclose(engine.fog_color, (0.7, 0.8, 0.9))
