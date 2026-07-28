import numpy as np
from manifoldx.resources import AtmosphereMaterial


def test_glow_subtype_and_shader():
    m = AtmosphereMaterial("#88bbff", intensity=1.5)
    assert m.pipeline_subtype == "glow"
    src = AtmosphereMaterial._compile()
    assert "camera_pos" in src and "@binding(3)" not in src   # unlit -> needs_lights False
    assert "pow(" in src                                       # fresnel term


def test_glow_uniform_is_rgb_intensity():
    d = AtmosphereMaterial((0.5, 0.7, 1.0), intensity=2.0).get_data(3, None)
    assert d.shape == (3, 4)
    assert np.allclose(d[0], [0.5, 0.7, 1.0, 2.0])
