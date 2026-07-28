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
