"""Rigid GPU skinning: Mesh(joints=), EntityHandle.set_joints, the skinned mesh pass."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh as MeshC, Transform
from manifoldx.modeling import Mesh
from manifoldx.resources import (
    BasicMaterial,
    DirectionalLight,
    FlatMaterial,
    PhongMaterial,
    PointLight,
    StandardMaterial,
)


# --- helpers -----------------------------------------------------------------


def _parts(colors=False, uvs=False):
    """Three boxes side by side on one mesh, joint i on box i."""
    boxes = [Mesh.box(0.5, 0.8, 0.5).translate((x, 0.0, 0.0)) for x in (-0.8, 0.0, 0.8)]
    pos, faces, normals, joints, cols = [], [], [], [], []
    base = 0
    for i, b in enumerate(boxes):
        b = b.recompute_normals()
        pos.append(b.positions)
        normals.append(b.normals)
        faces.append(b.faces + base)
        joints.append(np.full(len(b.positions), i))
        cols.append(np.tile([(0.9, 0.2, 0.2), (0.2, 0.9, 0.2), (0.2, 0.3, 0.9)][i], (len(b.positions), 1)))
        base += len(b.positions)
    p = np.concatenate(pos).astype(np.float32)
    return Mesh(
        positions=p,
        faces=np.concatenate(faces).astype(np.uint32),
        normals=np.concatenate(normals).astype(np.float32),
        uvs=(p[:, [0, 1]] * 0.5 + 0.5).astype(np.float32) if uvs else None,
        colors=np.concatenate(cols).astype(np.float32) if colors else None,
        joints=np.concatenate(joints),
    )


def _rot(axis, angle, pivot):
    """4x4 rotation about `axis` (unit) through `pivot`, column-vector convention."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.asarray(pivot) - r @ np.asarray(pivot)
    return m


def _pose():
    """Rotation about a pivot, non-uniform scale with rotation, and a collapsed part."""
    a = _rot((0, 0, 1), 0.6, (-0.8, -0.4, 0.0))
    b = _rot((1, 1, 0), -0.5, (0.0, 0.0, 0.0)) @ np.diag([1.4, 0.6, 1.0, 1.0])
    b[:3, 3] += (0.0, 0.2, 0.0)
    c = np.zeros((4, 4))
    c[3, 3] = 1.0
    c[:3, 3] = (0.8, 0.0, 0.0)  # every vertex of part 2 goes to one point
    return np.stack([a, b, c]).astype(np.float32)


def _normal_matrix(m):
    """Numpy twin of the shader rule: cofactor of the 3x3 times sign(det), or identity."""
    a, b, c = m[:3, 0], m[:3, 1], m[:3, 2]
    det = float(np.dot(a, np.cross(b, c)))
    if abs(det) < 1e-8:
        return np.eye(3)
    return np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=1) * np.sign(det)


def _cpu_pose(mesh, mats):
    """The skinned mesh posed in numpy, without joints (drawn unskinned)."""
    m = mats.astype(np.float64)[mesh.joints]                       # (N, 4, 4)
    p = np.concatenate([mesh.positions, np.ones((len(mesh.positions), 1))], axis=1)
    pos = np.einsum("nij,nj->ni", m, p)[:, :3]
    nm = np.stack([_normal_matrix(x) for x in m])
    n = np.einsum("nij,nj->ni", nm, mesh.normals)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return Mesh(
        positions=pos.astype(np.float32), faces=mesh.faces, normals=n.astype(np.float32),
        uvs=mesh.uvs, colors=mesh.colors,
    )


def _engine(w=64, h=48):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("skin", width=w, height=h)
    e.add_light(PointLight(color="#ffffff", intensity=25.0, position=(1.5, 2.0, 3.0)))
    e.set_sun(DirectionalLight(color="#ffffff", intensity=2.0, direction=(-0.3, -0.5, -1.0)))
    e.camera.set_pose(position=(0.3, 0.5, 1.8), target=(0, 0, 0))
    return e


def _texture(engine, tmp_path):
    from PIL import Image
    from manifoldx.textures import load_texture
    png = tmp_path / "grad.png"
    g = np.linspace(0, 255, 16).astype(np.uint8)
    rgb = np.stack(np.broadcast_arrays(g[None, :], g[:, None], np.full((16, 16), 90, np.uint8)), axis=-1)
    Image.fromarray(np.ascontiguousarray(rgb)).save(png)
    engine.render_frame(supersample=1)  # creates the device load_texture needs
    return load_texture(engine, png)


def _material(variant, engine, tmp_path):
    if variant == "plain":
        return StandardMaterial(color="#d0a060", roughness=0.6)
    if variant == "vcolor":
        return StandardMaterial(color="#ffffff", roughness=0.6, vertex_colors=True)
    return StandardMaterial(color="#ffffff", roughness=0.6, albedo_map=_texture(engine, tmp_path))


def _mesh_for(variant):
    return _parts(colors=variant == "vcolor", uvs=variant == "textured")


def _render(meshes_and_poses, variant, tmp_path, skinned):
    """Render [(mesh, mats or None, x)] either skinned via set_joints or CPU-posed."""
    e = _engine()
    mat = _material(variant, e, tmp_path)
    geos = {}
    for mesh, mats, x in meshes_and_poses:
        if skinned:
            geo = geos.setdefault(id(mesh), mesh.to_geometry())  # one geometry per Mesh object
            h = e.spawn(MeshC(geo), Material(mat), Transform(pos=(x, 0, 0), scale=(0.6, 0.6, 0.6)))
            if mats is not None:
                h.set_joints(mats)
        else:
            posed = mesh if mats is None else _cpu_pose(mesh, mats)
            e.spawn(MeshC(posed.to_geometry()), Material(mat), Transform(pos=(x, 0, 0), scale=(0.6, 0.6, 0.6)))
    return e.render_frame(supersample=1).astype(np.float64)


def _assert_same(a, b):
    diff = np.abs(a - b).mean()
    assert diff < 1.0, f"mean abs diff {diff:.3f} >= 1/255"
    # The mean alone misses wrong normals on one small part (about 30 pixels
    # off by more than 16 levels in a 64x48 frame); allow a few edge pixels.
    off = int((np.abs(a - b).max(axis=2) > 16).sum())
    assert off <= 8, f"{off} pixels differ by more than 16 levels"


# --- Mesh(joints=) -------------------------------------------------------------


def test_max_joints_is_exported():
    assert mx.MAX_JOINTS == 32


def test_to_geometry_carries_joints_as_uint32():
    m = _parts()
    geo = m.to_geometry()
    assert geo["joints"].dtype == np.uint32
    assert geo["joints"].shape == (len(m.positions),)
    assert set(geo["joints"].tolist()) == {0, 1, 2}


def test_to_geometry_has_no_joints_when_unset():
    assert "joints" not in Mesh.box().to_geometry()


@pytest.mark.parametrize(
    "joints",
    [
        np.zeros(7, dtype=int),             # wrong length
        np.zeros((8, 1), dtype=int),        # not 1-D
        np.full(8, -1),                     # negative
        np.full(8, 32),                     # >= MAX_JOINTS
        np.zeros(8, dtype=np.float32),      # not integers
    ],
)
def test_bad_joints_are_rejected(joints):
    box = Mesh.box()
    with pytest.raises(ValueError, match="joints"):
        Mesh(positions=box.positions, faces=box.faces, joints=joints)


def test_ops_that_keep_vertex_order_keep_joints():
    m = _parts(colors=True)
    kept = [
        m.translate((1, 0, 0)),
        m.with_positions(m.positions * 2),
        m.recompute_normals(),
        m.with_colors(m.colors),
        m.color_by(lambda p: p[:, 0], lambda v: np.stack([v, v, v], axis=1)),
        m.twist(0.3),
        m.bend(0.3),
        m.taper(0.5),
        m.displace(lambda p: p[:, 1], 0.1),
        m.draw((0, 0, 0), 0.5, 0.1),
        m.inflate((0, 0, 0), 0.5, 0.1),
        m.pinch((0, 0, 0), 0.5, 0.1),
        m.flatten((0, 0, 0), 0.5, 0.1),
        m.smooth(1),
        m.ffd().apply(),
        Mesh.from_geometry(m.to_geometry()),
    ]
    for out in kept:
        assert np.array_equal(out.joints, m.joints)


def test_topology_ops_do_not_crash_on_skinned_meshes():
    m = _parts()
    m.subdivide()
    m.decimate(8)
    m.extrude(np.arange(len(m.faces)) < 2, 0.1)
    m.union(Mesh.box().translate((0.3, 0.3, 0.3)))


# --- normal-matrix rule -----------------------------------------------------------


def test_normal_matrix_rule_matches_inverse_transpose():
    rng = np.random.default_rng(7)
    for _ in range(50):
        m = np.eye(4)
        m[:3, :3] = rng.normal(size=(3, 3))
        if rng.random() < 0.5:
            m[:3, 0] *= -1  # mirrored
        expected = np.linalg.inv(m[:3, :3]).T
        n = rng.normal(size=3)
        got = _normal_matrix(m) @ n
        want = expected @ n
        assert np.allclose(got / np.linalg.norm(got), want / np.linalg.norm(want), atol=1e-6)


def test_normal_matrix_rule_is_identity_on_singular():
    m = np.eye(4)
    m[:3, :3] = 0.0
    assert np.array_equal(_normal_matrix(m), np.eye(3))
    m[:3, :3] = np.diag([1.0, 1.0, 1e-12])
    assert np.array_equal(_normal_matrix(m), np.eye(3))


# --- shader transform --------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        StandardMaterial._compile(),
        StandardMaterial._compile(vertex_colors=True),
        StandardMaterial._compile(textured=True),
        BasicMaterial._compile(),
        FlatMaterial._compile(),
        PhongMaterial._compile(),
    ],
)
def test_skin_shader_accepts_every_mesh_material(source):
    from manifoldx.skinning import skin_shader
    out = skin_shader(source)
    assert "fn vs_main(in_raw: VertexInput)" in out
    assert "var in = skin_vertex(in_raw);" in out
    assert "joint: u32" in out


def test_skin_shader_rejects_unexpected_wgsl():
    from manifoldx.skinning import skin_shader
    with pytest.raises(ValueError, match="VertexInput"):
        skin_shader("@vertex fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }")


# --- set_joints errors ---------------------------------------------------------------


def _spawn(e, mesh):
    return e.spawn(MeshC(mesh.to_geometry()), Material(StandardMaterial(color="#ffffff")), Transform())


def test_set_joints_on_unskinned_entity_raises():
    e = mx.Engine("skin", width=32, height=32)
    h = _spawn(e, Mesh.box())
    with pytest.raises(ValueError, match="joints"):
        h.set_joints(np.eye(4)[None])


@pytest.mark.parametrize(
    "mats",
    [np.eye(4), np.zeros((2, 3, 3)), np.zeros((0, 4, 4)), np.tile(np.eye(4), (33, 1, 1))],
)
def test_set_joints_bad_shape_raises(mats):
    e = mx.Engine("skin", width=32, height=32)
    h = _spawn(e, _parts())
    with pytest.raises(ValueError):
        h.set_joints(mats)


def test_set_joints_before_device_is_kept_in_the_mirror():
    e = mx.Engine("skin", width=32, height=32)
    h = _spawn(e, _parts())
    h.set_joints(_pose()[:2])
    joints = e._render_pipeline._joints
    block = joints.block_of[h.index]
    assert block > 0
    assert np.allclose(joints.mirror[block, :2], _pose()[:2].transpose(0, 2, 1))
    assert np.array_equal(joints.mirror[block, 2:], np.tile(np.eye(4, dtype=np.float32), (30, 1, 1)))


def test_a_respawned_entity_does_not_inherit_a_block():
    e = mx.Engine("skin", width=32, height=32)
    h = _spawn(e, _parts())
    h.set_joints(_pose())
    e.destroy([h.index])
    e.commands.execute(e.store)
    h2 = _spawn(e, _parts())
    assert h2.index == h.index
    assert e._render_pipeline._joints.block_of[h2.index] == 0


# --- pixels ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["plain", "vcolor", "textured"])
def test_skinned_render_matches_cpu_posed_render(variant, tmp_path):
    mesh = _mesh_for(variant)
    gpu = _render([(mesh, _pose(), 0.0)], variant, tmp_path, skinned=True)
    cpu = _render([(mesh, _pose(), 0.0)], variant, tmp_path, skinned=False)
    rest = _render([(mesh, None, 0.0)], variant, tmp_path, skinned=False)
    assert np.abs(cpu - rest).mean() > 3.0  # the pose is visible, so the check can fail
    _assert_same(gpu, cpu)


def test_unposed_skinned_entity_renders_its_rest_pose(tmp_path):
    mesh = _parts()
    gpu = _render([(mesh, None, 0.0)], "plain", tmp_path, skinned=True)
    plain = Mesh(positions=mesh.positions, faces=mesh.faces, normals=mesh.normals)
    cpu = _render([(plain, None, 0.0)], "plain", tmp_path, skinned=False)
    _assert_same(gpu, cpu)


def test_two_entities_from_one_mesh_are_posed_independently(tmp_path):
    mesh = _parts()
    other = _pose()[[1, 0, 2]].copy()
    other[2] = _rot((0, 1, 0), 1.0, (0.8, 0, 0))
    scene = [(mesh, _pose(), -0.7), (mesh, other, 0.7)]
    gpu = _render(scene, "plain", tmp_path, skinned=True)
    cpu = _render(scene, "plain", tmp_path, skinned=False)
    _assert_same(gpu, cpu)


def test_fewer_joints_keep_the_previous_higher_matrices(tmp_path):
    mesh = _parts()
    first, second = _pose(), _pose()[[1]]
    e = _engine()
    h = e.spawn(MeshC(mesh.to_geometry()), Material(StandardMaterial(color="#d0a060", roughness=0.6)),
                Transform(scale=(0.6, 0.6, 0.6)))
    h.set_joints(first)
    e.render_frame(supersample=1)
    h.set_joints(second)
    gpu = e.render_frame(supersample=1).astype(np.float64)
    expected = np.concatenate([second, first[1:]])
    cpu = _render([(mesh, expected, 0.0)], "plain", tmp_path, skinned=False)
    _assert_same(gpu, cpu)


def test_unskinned_pipeline_uses_the_material_wgsl_unchanged():
    e = _engine()
    mat = StandardMaterial(color="#d0a060")
    e.spawn(MeshC(Mesh.box().to_geometry()), Material(mat), Transform(pos=(-1, 0, 0)))
    e.spawn(MeshC(_parts().to_geometry()), Material(mat), Transform(pos=(1, 0, 0)))
    e.render_frame(supersample=1)
    sources = e._render_pipeline._shader_sources
    plain = [s for s in sources.values() if "skin_vertex" not in s]
    skinned = [s for s in sources.values() if "skin_vertex" in s]
    assert plain == [StandardMaterial._compile()]
    assert len(skinned) == 1


def test_set_geometry_keeps_the_joints(tmp_path):
    mesh = _parts()
    moved = Mesh(positions=mesh.positions + np.float32(0.3) * np.array([0, 1, 0], np.float32),
                 faces=mesh.faces, normals=mesh.normals, joints=mesh.joints)
    e = _engine()
    h = e.spawn(MeshC(mesh.to_geometry()), Material(StandardMaterial(color="#d0a060", roughness=0.6)),
                Transform(scale=(0.6, 0.6, 0.6)))
    e.render_frame(supersample=1)  # vertex buffer exists, so set_geometry rewrites it
    h.set_geometry(Mesh(positions=moved.positions, faces=moved.faces, normals=moved.normals))
    h.set_joints(_pose())
    gpu = e.render_frame(supersample=1).astype(np.float64)
    cpu = _render([(moved, _pose(), 0.0)], "plain", tmp_path, skinned=False)
    _assert_same(gpu, cpu)
