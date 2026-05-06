"""Tests for manifoldx.viz.geometry module."""
import numpy as np

from manifoldx.viz import geometry as viz_geometry


def test_sprite_quad_vertex_count():
    """SPRITE_QUAD is a unit quad: 4 vertices."""
    q = viz_geometry.SPRITE_QUAD
    assert q["vertices"].shape == (4, 3)
    assert q["vertices"].dtype == np.float32


def test_sprite_quad_indices():
    """SPRITE_QUAD has 6 indices forming 2 triangles."""
    q = viz_geometry.SPRITE_QUAD
    assert q["indices"].shape == (6,)
    assert q["indices"].dtype == np.uint32
    # Indices reference all 4 vertices
    assert set(q["indices"].tolist()) == {0, 1, 2, 3}


def test_sprite_quad_uv_corners():
    """The four vertices form unit-quad corners in XY (z=0)."""
    q = viz_geometry.SPRITE_QUAD
    v = q["vertices"]
    # All vertices at z=0
    np.testing.assert_array_equal(v[:, 2], np.zeros(4, dtype=np.float32))
    # XY corners are (-1,-1), (1,-1), (1,1), (-1,1) in some order
    xy = set(map(tuple, v[:, :2].tolist()))
    assert xy == {(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)}


def test_sprite_quad_registered_in_geometry_registry():
    """After import, SPRITE_QUAD is a built-in geometry that can be looked up by name."""
    from manifoldx.resources import GeometryRegistry

    reg = GeometryRegistry()
    assert "sprite_quad" in reg
