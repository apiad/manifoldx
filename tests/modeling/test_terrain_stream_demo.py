import numpy as np
from manifoldx.modeling import Mesh, fields


def test_patch_builder_seams_are_continuous():
    HEIGHT, DEPTH = 8.0, 52.0
    tmpl = Mesh.plane(width=44, depth=DEPTH, segments=40)
    terrain = fields.ridged(seed=5, freq=0.12)

    def patch(z):
        return tmpl.displace(terrain.shift((0, 0, z)), amount=HEIGHT)

    a, b = patch(0.0), patch(DEPTH)
    # a's far edge and b's near edge sample the same world Z → identical heights.
    za, zb = a.positions[:, 2].max(), b.positions[:, 2].min()
    ha = a.positions[np.isclose(a.positions[:, 2], za)]
    hb = b.positions[np.isclose(b.positions[:, 2], zb)]
    assert np.allclose(np.sort(ha[:, 1]), np.sort(hb[:, 1]), atol=1e-4)
