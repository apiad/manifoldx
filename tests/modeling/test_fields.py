import numpy as np
from manifoldx.modeling import fields
from manifoldx.modeling.fields import Field


def _pts():
    return np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)


X = Field(lambda p: p[:, 0])          # the x-coordinate as a field


def test_call_returns_float32():
    out = X(_pts())
    assert out.dtype == np.float32 and out.shape == (3,)
    assert np.allclose(out, [0, 1, 2])


def test_arithmetic_operators():
    assert np.allclose((X + 1.0)(_pts()), [1, 2, 3])
    assert np.allclose((1.0 + X)(_pts()), [1, 2, 3])
    assert np.allclose((2.0 * X)(_pts()), [0, 2, 4])
    assert np.allclose((X - X)(_pts()), [0, 0, 0])
    assert np.allclose((10.0 - X)(_pts()), [10, 9, 8])
    assert np.allclose((X / 2.0)(_pts()), [0, 0.5, 1.0])
    assert np.allclose((-X)(_pts()), [0, -1, -2])


def test_combinators():
    assert np.allclose(X.clamp(0, 1)(_pts()), [0, 1, 1])
    assert np.allclose(X.remap(0, 2, 0, 10)(_pts()), [0, 5, 10])
    assert np.allclose(X.minimum(1.0)(_pts()), [0, 1, 1])
    assert np.allclose(X.maximum(1.0)(_pts()), [1, 1, 2])
    assert np.allclose(X.power(2)(_pts()), [0, 1, 4])
    assert np.allclose(X.scale(3).bias(1)(_pts()), [1, 4, 7])
    ten = Field(lambda p: np.full(len(p), 10.0))
    assert np.allclose(X.mix(ten, 0.5)(_pts()), [5.0, 5.5, 6.0])


def test_warp_shifts_sampling():
    ones = Field(lambda p: np.ones(len(p)))
    warped = X.warp(1.0, fx=ones)      # sample X at x + 1 → values shift up by 1
    assert np.allclose(warped(_pts()), [1, 2, 3])


from manifoldx.modeling import noise  # noqa: E402


def _grid(k=200):
    return np.random.default_rng(0).uniform(-3, 3, size=(k, 3)).astype(np.float32)


def test_perlin_fbm_are_fields():
    assert isinstance(fields.perlin(seed=1), Field)
    assert isinstance(fields.fbm(seed=1), Field)


def test_perlin_deterministic_and_bounded():
    pts = _grid(500)
    a, b = fields.perlin(seed=7)(pts), fields.perlin(seed=7)(pts)
    assert np.array_equal(a, b)
    assert a.shape == (500,) and a.min() >= -1.5 and a.max() <= 1.5


def test_noise_shim_matches_fields():
    pts = _grid()
    assert np.array_equal(noise.fbm(seed=3, octaves=4)(pts), fields.fbm(seed=3, octaves=4)(pts))
    assert callable(noise.perlin(seed=2))


def test_ridged_and_billow_range_and_determinism():
    pts = _grid(500)
    for src in (fields.ridged, fields.billow):
        a, b = src(seed=4)(pts), src(seed=4)(pts)
        assert np.array_equal(a, b)
        assert a.shape == (500,)
        assert a.min() >= -1e-4 and a.max() <= 1.0 + 1e-4      # non-negative, bounded
    assert not np.allclose(fields.ridged(seed=1)(pts), fields.ridged(seed=2)(pts))


def test_worley_deterministic_nonnegative():
    pts = _grid(400)
    a, b = fields.worley(seed=5)(pts), fields.worley(seed=5)(pts)
    assert np.array_equal(a, b)
    assert a.shape == (400,) and a.min() >= 0.0
    assert not np.allclose(fields.worley(seed=1)(pts), fields.worley(seed=2)(pts))


def test_worley_f2f1_is_ridge_like():
    w = fields.worley(seed=3, feature="f2f1")
    v = w(_grid(400))
    assert v.min() >= 0.0
    assert v.max() > 0.1          # some cellular structure exists


def test_constant_coord_distance():
    pts = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 2]], dtype=np.float64)
    assert np.allclose(fields.constant(2.5)(pts), [2.5, 2.5, 2.5])
    assert np.allclose(fields.coord("x")(pts), [0, 3, 0])
    assert np.allclose(fields.coord("z")(pts), [0, 0, 2])
    assert np.allclose(fields.distance()(pts), [0, 5, 2])
    assert np.allclose(fields.distance(center=(3, 4, 0))(pts), [5, 0, np.sqrt(3**2 + 4**2 + 2**2)])
