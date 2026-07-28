import numpy as np
from manifoldx.modeling import Mesh, Gradient, fields


def test_with_colors_stores_and_emits():
    m = Mesh.icosphere(subdivisions=1)
    n = m.positions.shape[0]
    colors = np.tile([0.2, 0.4, 0.6], (n, 1)).astype(np.float32)
    c = m.with_colors(colors)
    assert c.colors.shape == (n, 3)
    geo = c.to_geometry()
    assert "colors" in geo and geo["colors"].shape == (n, 3) and geo["colors"].dtype == np.float32


def test_to_geometry_omits_colors_when_absent():
    geo = Mesh.icosphere(subdivisions=1).to_geometry()
    assert "colors" not in geo


def test_color_by_constant_field_is_uniform():
    m = Mesh.icosphere(subdivisions=2)
    grad = Gradient([(0.0, "#000000"), (1.0, "#ffffff")])
    c = m.color_by(fields.constant(0.5), grad)
    assert c.colors.shape == (m.positions.shape[0], 3)
    assert np.allclose(c.colors, 0.5, atol=2e-3)          # constant field → uniform grey


def test_color_by_height_varies():
    m = Mesh.icosphere(subdivisions=3, radius=1.0)
    grad = Gradient([(-1.0, "#000000"), (1.0, "#ffffff")])
    c = m.color_by(fields.coord("y"), grad)
    # top (y≈+1) is brighter than bottom (y≈-1)
    top = c.colors[m.positions[:, 1] > 0.8].mean()
    bot = c.colors[m.positions[:, 1] < -0.8].mean()
    assert top > bot + 0.3


def test_deformer_preserves_colors():
    m = Mesh.icosphere(subdivisions=2).color_by(fields.coord("y"),
                                                 Gradient([(-1, "#000"), (1, "#fff")]))
    warped = m.twist(angle=0.5, axis="y")
    assert warped.colors is not None
    assert warped.colors.shape == m.colors.shape
    assert np.array_equal(warped.colors, m.colors)         # per-vertex color rides through
