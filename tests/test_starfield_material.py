import numpy as np
from manifoldx.sky import StarfieldMaterial


def test_starfield_daynight_shader():
    m = StarfieldMaterial(ground_radius=10.0, atmo_top=2.0)
    src = StarfieldMaterial._compile()
    assert "sun_direction" in src                 # needs the full globals (lighting) block
    assert "cam_alt" in src and "in_atmo" in src  # altitude-gated so space stays starry
    assert "discard" in src                        # fully daylit -> sky shows through


def test_starfield_uniform_carries_planet_params():
    d = StarfieldMaterial(vmin=0.0, vmax=1.0, ground_radius=10.0, atmo_top=2.0).get_data(4, None)
    assert d.shape == (4, 4)
    assert np.allclose(d[0], [0.0, 1.0, 10.0, 2.0])


def test_starfield_defaults_never_fade():
    # atmo_top == 0 -> plain deep-space starfield (no fade branch active)
    d = StarfieldMaterial().get_data(1, None)
    assert d[0, 3] == 0.0
