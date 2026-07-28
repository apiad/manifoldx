import numpy as np
from manifoldx.modeling import Mesh, noise


def test_displace_changes_positions_and_is_deterministic():
    base = Mesh.icosphere(subdivisions=3)
    field = noise.fbm(seed=11, octaves=4)
    a = base.displace(field, amount=0.3)
    b = base.displace(field, amount=0.3)
    assert a.positions.shape == base.positions.shape
    assert not np.allclose(a.positions, base.positions)
    assert np.array_equal(a.positions, b.positions)         # deterministic
    assert a.normals is not None                             # normals recomputed


def test_displace_along_fixed_vector():
    base = Mesh.plane(width=2, depth=2, segments=4)
    const = lambda pts: np.ones(len(pts), dtype=np.float32)  # noqa: E731
    out = base.displace(const, amount=0.5, along=(0, 1, 0))
    assert np.allclose(out.positions[:, 1], base.positions[:, 1] + 0.5)


def test_twist_preserves_axis_coordinate_and_count():
    base = Mesh.cylinder(radius=1.0, height=4.0, segments=24)
    out = base.twist(angle=1.0, axis="y")
    assert out.positions.shape == base.positions.shape
    # Twisting about y leaves the y coordinate unchanged.
    assert np.allclose(out.positions[:, 1], base.positions[:, 1])
    # Points off the axis actually move.
    assert not np.allclose(out.positions[:, [0, 2]], base.positions[:, [0, 2]])


def test_twist_zero_angle_is_identity():
    base = Mesh.cylinder(radius=1.0, height=2.0, segments=12)
    out = base.twist(angle=0.0, axis="y")
    assert np.allclose(out.positions, base.positions)


def test_bend_zero_angle_identity_and_count():
    base = Mesh.plane(width=4, depth=1, segments=8)
    out = base.bend(angle=0.0, axis="z", along="x")
    assert out.positions.shape == base.positions.shape
    assert np.allclose(out.positions, base.positions)


def test_bend_curves_the_strip():
    base = Mesh.plane(width=4, depth=1, segments=16)  # flat in XZ, y == 0
    out = base.bend(angle=1.5, axis="z", along="x")
    # Bending about z, driven by x, must lift vertices out of the y == 0 plane.
    assert np.abs(out.positions[:, 1]).max() > 0.1


def test_taper_narrows_one_end():
    base = Mesh.cylinder(radius=1.0, height=2.0, segments=24)
    out = base.taper(factor=0.8, axis="y")
    top = base.positions[:, 1] > 0.9
    bot = base.positions[:, 1] < -0.9
    r_top = np.linalg.norm(out.positions[top][:, [0, 2]], axis=1).mean()
    r_bot = np.linalg.norm(out.positions[bot][:, [0, 2]], axis=1).mean()
    assert r_top > r_bot        # widened at +y, narrowed at -y


def test_taper_zero_is_identity():
    base = Mesh.cylinder(radius=1.0, height=2.0, segments=12)
    out = base.taper(factor=0.0, axis="y")
    assert np.allclose(out.positions, base.positions)
