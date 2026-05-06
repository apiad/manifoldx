"""Tests for manifoldx.viz.colormaps module."""
import numpy as np
import pytest

from manifoldx.viz import colormaps


def test_viridis_shape():
    lut = colormaps.get_colormap("viridis")
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8


def test_viridis_endpoints():
    """Viridis spans dark purple (idx 0) to yellow (idx 255)."""
    lut = colormaps.get_colormap("viridis")
    # Dark purple: low R, low G, mid B
    r0, g0, b0, a0 = lut[0]
    assert r0 < 100
    assert g0 < 30
    assert 60 < b0 < 130
    assert a0 == 255
    # Yellow: high R, high G, low B
    r255, g255, b255, a255 = lut[255]
    assert r255 > 200
    assert g255 > 200
    assert b255 < 60
    assert a255 == 255


def test_unknown_colormap_raises():
    with pytest.raises(KeyError, match="unknown colormap"):
        colormaps.get_colormap("not-a-real-cmap")


def test_lookup_normalized_value():
    """Sample at a normalized scalar; verify (R,G,B,A) returned matches LUT."""
    lut = colormaps.get_colormap("viridis")
    rgba = colormaps.lookup("viridis", 0.5)
    assert rgba.shape == (4,)
    # Index 127 or 128 — accept either
    np.testing.assert_array_equal(rgba, lut[127]) if (rgba == lut[127]).all() else np.testing.assert_array_equal(rgba, lut[128])


@pytest.mark.parametrize("name", ["viridis", "magma", "plasma", "inferno", "turbo", "gray"])
def test_all_colormaps_well_formed(name):
    lut = colormaps.get_colormap(name)
    assert lut.shape == (256, 4)
    assert lut.dtype == np.uint8
    # Alpha is always 255
    assert (lut[:, 3] == 255).all()


def test_gray_is_monotonic():
    lut = colormaps.get_colormap("gray")
    rgb = lut[:, :3]
    # All three channels should equal each other and be monotonic increasing
    assert (rgb[:, 0] == rgb[:, 1]).all()
    assert (rgb[:, 1] == rgb[:, 2]).all()
    assert (np.diff(rgb[:, 0].astype(int)) >= 0).all()


def test_available_colormaps_complete():
    available = sorted(colormaps._LUTS.keys())
    assert available == sorted(["viridis", "magma", "plasma", "inferno", "turbo", "gray"])
