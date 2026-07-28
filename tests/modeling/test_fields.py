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
