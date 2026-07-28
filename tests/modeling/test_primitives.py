import numpy as np
from manifoldx.modeling import Mesh


def test_icosphere_face_count_grows_by_four():
    base = Mesh.icosphere(subdivisions=0)
    assert base.faces.shape[0] == 20            # icosahedron
    one = Mesh.icosphere(subdivisions=1)
    assert one.faces.shape[0] == 80             # x4 per subdivision


def test_icosphere_vertices_on_radius():
    m = Mesh.icosphere(subdivisions=2, radius=2.0)
    r = np.linalg.norm(m.positions, axis=1)
    assert np.allclose(r, 2.0, atol=1e-5)
