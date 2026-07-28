import numpy as np
from manifoldx.modeling import Gradient


def test_gradient_endpoints_and_midpoint():
    g = Gradient([(0.0, "#000000"), (1.0, "#ffffff")])
    assert np.allclose(g([0.0])[0], [0, 0, 0])
    assert np.allclose(g([1.0])[0], [1, 1, 1])
    assert np.allclose(g([0.5])[0], [0.5, 0.5, 0.5], atol=2e-3)


def test_gradient_clamps_outside_range():
    g = Gradient([(0.0, "#000000"), (1.0, "#ffffff")])
    assert np.allclose(g([-5.0])[0], [0, 0, 0])
    assert np.allclose(g([9.0])[0], [1, 1, 1])


def test_gradient_three_stops_and_shape():
    g = Gradient([(0.0, "#ff0000"), (0.5, "#00ff00"), (1.0, "#0000ff")])
    assert np.allclose(g([0.5])[0], [0, 1, 0], atol=2e-3)
    out = g([0.1, 0.9])
    assert out.shape == (2, 3) and out.dtype == np.float32


def test_gradient_accepts_rgb_tuples_and_unsorted_stops():
    g = Gradient([(1.0, (0.0, 0.0, 1.0)), (0.0, (1.0, 0.0, 0.0))])  # unsorted
    assert np.allclose(g([0.0])[0], [1, 0, 0])
    assert np.allclose(g([1.0])[0], [0, 0, 1])
