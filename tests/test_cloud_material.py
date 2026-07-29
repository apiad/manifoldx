import numpy as np
from manifoldx.resources import CloudMaterial


def test_cloud_subtype_and_shader():
    m = CloudMaterial()
    assert m.pipeline_subtype == "cloud"
    src = CloudMaterial._compile()
    assert "sun_direction" in src                 # sun-lit tops / dark night
    assert "fbm" in src and "vnoise" in src       # procedural coverage
    assert "local_pos" in src                     # object-space sampling (pattern rides the shell)
    assert "discard" in src                        # gaps are transparent


def test_cloud_uniform_is_coverage_softness_freq_opacity():
    d = CloudMaterial(coverage=0.5, softness=0.1, freq=4.0, opacity=0.8).get_data(3, None)
    assert d.shape == (3, 4)
    assert np.allclose(d[0], [0.5, 0.1, 4.0, 0.8])
