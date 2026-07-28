import numpy as np
from manifoldx.modeling import noise


def _grid(k=200):
    rng = np.random.default_rng(0)
    return rng.uniform(-3, 3, size=(k, 3)).astype(np.float32)


def test_perlin_deterministic_same_seed():
    pts = _grid()
    a = noise.perlin(seed=42)(pts)
    b = noise.perlin(seed=42)(pts)
    assert np.array_equal(a, b)


def test_perlin_differs_across_seeds():
    pts = _grid()
    a = noise.perlin(seed=1)(pts)
    b = noise.perlin(seed=2)(pts)
    assert not np.allclose(a, b)


def test_perlin_range_and_shape():
    pts = _grid(500)
    vals = noise.perlin(seed=7)(pts)
    assert vals.shape == (500,)
    assert vals.min() >= -1.5 and vals.max() <= 1.5  # gradient noise stays bounded


def test_fbm_deterministic_and_shaped():
    pts = _grid()
    a = noise.fbm(seed=3, octaves=4)(pts)
    b = noise.fbm(seed=3, octaves=4)(pts)
    assert np.array_equal(a, b) and a.shape == (pts.shape[0],)
