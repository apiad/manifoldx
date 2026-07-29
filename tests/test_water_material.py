import numpy as np
from manifoldx.resources import WaterMaterial


def test_water_shader_is_sun_aware_and_unlit():
    src = WaterMaterial._compile()
    assert "sun_direction" in src            # day/night + glint follow the sun
    assert "@binding(3)" not in src          # non-lit -> 16-byte uniform path
    assert "pow(" in src                     # fresnel + specular glint


def test_water_uniform_is_deep_color_plus_fresnel():
    d = WaterMaterial((0.05, 0.2, 0.4), fresnel_power=4.0).get_data(2, None)
    assert d.shape == (2, 4)
    assert np.allclose(d[0], [0.05, 0.2, 0.4, 4.0])
