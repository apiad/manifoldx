import numpy as np
from manifoldx.resources import AtmosphereScatteringMaterial


def test_scatter_subtype_and_physics_shader():
    m = AtmosphereScatteringMaterial(ground_radius=10.0, atmosphere_radius=10.6)
    assert m.pipeline_subtype == "scatter"
    src = AtmosphereScatteringMaterial._compile()
    assert "sun_direction" in src and "@binding(3)" not in src   # unlit -> 16-byte path
    assert "betaR" in src and "phase" in src.lower()             # Rayleigh/Mie single scattering
    assert "ray_sphere" in src                                    # analytic atmosphere intersection


def test_scatter_uniform_is_radii_intensity_exposure():
    d = AtmosphereScatteringMaterial(10.0, 10.6, intensity=22.0, exposure=1.3).get_data(2, None)
    assert d.shape == (2, 4)
    assert np.allclose(d[0], [10.0, 10.6, 22.0, 1.3])
