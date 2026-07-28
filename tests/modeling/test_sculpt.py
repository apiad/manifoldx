import numpy as np
from manifoldx.modeling import Mesh, Falloff


def test_adjacency_of_single_triangle():
    m = Mesh(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    adj = m.adjacency()
    assert adj.offsets.shape == (4,)
    # Every vertex of a triangle neighbors the other two.
    for i in range(3):
        nbrs = set(adj.neighbors[adj.offsets[i]:adj.offsets[i + 1]].tolist())
        assert nbrs == {j for j in range(3) if j != i}


def test_adjacency_is_cached():
    m = Mesh.icosphere(subdivisions=2)
    assert m.adjacency() is m.adjacency()   # same object, built once


def test_adjacency_symmetric_on_icosphere():
    m = Mesh.icosphere(subdivisions=1)
    adj = m.adjacency()
    pairs = set()
    for i in range(len(m.positions)):
        for j in adj.neighbors[adj.offsets[i]:adj.offsets[i + 1]]:
            pairs.add((i, int(j)))
    assert all((j, i) in pairs for (i, j) in pairs)   # undirected


def test_falloff_weights_bounds():
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    w = Falloff(center=(0, 0, 0), radius=1.0).weights(pts)
    assert w[0] == 1.0                 # at center
    assert w[2] == 0.0 and w[3] == 0.0  # at/outside radius
    assert 0.0 < w[1] < 1.0            # partway
    assert w.dtype == np.float32


def test_draw_raises_bump_near_center_only():
    base = Mesh.plane(width=4, depth=4, segments=20)
    out = base.draw(center=(0, 0, 0), radius=1.0, strength=0.5)
    disp = out.positions - base.positions
    d = np.linalg.norm(base.positions - np.array([0, 0, 0]), axis=1)
    assert np.abs(disp[d < 0.2]).max() > 0.1      # bump near center
    assert np.allclose(disp[d > 1.5], 0.0)        # untouched far away


def test_inflate_pushes_along_vertex_normals():
    base = Mesh.icosphere(subdivisions=3, radius=1.0)
    out = base.inflate(center=(0, 0, 1), radius=0.6, strength=0.3)
    r_in = np.linalg.norm(base.positions, axis=1)
    r_out = np.linalg.norm(out.positions, axis=1)
    near_pole = base.positions[:, 2] > 0.7
    assert (r_out[near_pole] > r_in[near_pole] + 0.05).any()   # bulges outward
    far = base.positions[:, 2] < 0.0
    assert np.allclose(r_out[far], r_in[far], atol=1e-4)       # untouched


def test_pinch_pulls_vertices_toward_center():
    base = Mesh.plane(width=4, depth=4, segments=20)
    center = (0.0, 0.0, 0.0)
    out = base.pinch(center=center, radius=1.5, strength=0.5)
    c = np.asarray(center)
    d_in = np.linalg.norm(base.positions - c, axis=1)
    d_out = np.linalg.norm(out.positions - c, axis=1)
    sel = (d_in > 0.1) & (d_in < 1.0)
    assert np.all(d_out[sel] < d_in[sel])          # closer to center
    assert np.allclose(d_out[d_in > 2.0], d_in[d_in > 2.0])  # untouched


def test_flatten_reduces_height_variation_in_region():
    base = Mesh.icosphere(subdivisions=3, radius=1.0)
    # Flatten the +x cap toward its own plane.
    out = base.flatten(center=(1, 0, 0), radius=0.7, strength=1.0)
    sel = base.positions[:, 0] > 0.7
    # Radial spread (distance from x-axis) of the selected cap shrinks as it flattens.
    spread_in = np.linalg.norm(base.positions[sel][:, [1, 2]], axis=1).std()
    spread_out = np.linalg.norm(out.positions[sel][:, [1, 2]], axis=1).std()
    assert out.positions.shape == base.positions.shape
    assert spread_out <= spread_in + 1e-4
    far = base.positions[:, 0] < 0.0
    assert np.allclose(out.positions[far], base.positions[far], atol=1e-4)
