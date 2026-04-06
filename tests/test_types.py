"""Tests for manifoldx.types module."""
import numpy as np
import pytest


def test_vector3_creation():
    """Vector3 should be a 3-element array."""
    from manifoldx.types import Vector3
    v = Vector3([1, 2, 3])
    assert v.shape == (3,)


def test_vector3_addition():
    """Vector3 should support vectorized addition."""
    from manifoldx.types import Vector3
    v1 = Vector3([1, 2, 3])
    v2 = Vector3([4, 5, 6])
    result = v1 + v2
    np.testing.assert_array_equal(result, [5, 7, 9])


def test_vector4_creation():
    """Vector4 should be a 4-element array."""
    from manifoldx.types import Vector4
    v = Vector4([1, 2, 3, 4])
    assert v.shape == (4,)


def test_color_from_hex():
    """Color should be created from hex string."""
    from manifoldx.types import Color
    c = Color("#ff0000")
    assert c.r == 1.0
    assert c.g == 0.0
    assert c.b == 0.0
    assert c.a == 1.0


def test_color_from_rgb():
    """Color should be created from RGB values."""
    from manifoldx.types import Color
    c = Color(r=1.0, g=0.0, b=0.0)
    assert c.r == 1.0
    assert c.g == 0.0
    assert c.b == 0.0


def test_color_to_linear():
    """Color should convert sRGB to linear."""
    from manifoldx.types import Color
    c = Color("#888888")  # sRGB mid-gray (0.5333)
    linear = c.to_linear()
    # Gamma correction: linear value should be different from sRGB
    # For gray #888888, sRGB is ~0.533, linear should be ~0.265
    assert linear.r != c.r  # Conversion happened
    assert 0.2 < linear.r < 0.4  # Check reasonable linear value


def test_color_to_srgb():
    """Color should convert linear to sRGB."""
    from manifoldx.types import Color
    c = Color(linear_r=0.2, linear_g=0.0, linear_b=0.0)
    srgb = c.to_srgb()
    # Gamma correction: sRGB value should be greater than linear
    assert srgb.r > 0.2


def test_float_type():
    """Float type marker should exist."""
    from manifoldx.types import Float
    f = Float(5.0)
    assert float(f) == 5.0
