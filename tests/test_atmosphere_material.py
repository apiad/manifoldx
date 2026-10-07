import numpy as np
from manifoldx.resources import AtmosphereMaterial, srgb_to_linear


def test_glow_subtype_and_shader():
    m = AtmosphereMaterial("#88bbff", intensity=1.5)
    assert m.pipeline_subtype == "glow"
    src = AtmosphereMaterial._compile()
    assert "camera_pos" in src and "@binding(3)" not in src   # unlit -> needs_lights False
    assert "pow(" in src                                       # fresnel term


def test_glow_uniform_is_rgb_intensity():
    d = AtmosphereMaterial((0.5, 0.7, 1.0), intensity=2.0).get_data(3, None)
    assert d.shape == (3, 4)
    # colours are sRGB and arrive decoded to linear; the intensity is passed through
    assert np.allclose(d[0], [*srgb_to_linear([0.5, 0.7, 1.0]), 2.0])


def test_atmosphere_is_sun_aware():
    src = AtmosphereMaterial._compile()
    assert "sun_direction" in src                              # day/night driven by the sun
