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
    if abs(det) <= 1e-6 * np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c):
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


# Both ways to pose: one EntityHandle.set_joints per entity, or one batched call.
HOW = ["handle", "batched"]


def _set_joints(e, handles, poses, how):
    """Pose each handle with its (J, 4, 4) matrices, all of one J."""
    if not handles:
        return
    if how == "handle":
        for h, m in zip(handles, poses):
            h.set_joints(m)
    else:
        e.set_joints([h.index for h in handles], np.stack(poses))


def _render(meshes_and_poses, variant, tmp_path, skinned, how="handle"):
    """Render [(mesh, mats or None, x)] either skinned via set_joints or CPU-posed."""
    e = _engine()
    mat = _material(variant, e, tmp_path)
    geos, posed_handles, poses = {}, [], []
    for mesh, mats, x in meshes_and_poses:
        if skinned:
            geo = geos.setdefault(id(mesh), mesh.to_geometry())  # one geometry per Mesh object
            h = e.spawn(MeshC(geo), Material(mat), Transform(pos=(x, 0, 0), scale=(0.6, 0.6, 0.6)))
            if mats is not None:
                posed_handles.append(h)
                poses.append(mats)
        else:
            posed = mesh if mats is None else _cpu_pose(mesh, mats)
            e.spawn(MeshC(posed.to_geometry()), Material(mat), Transform(pos=(x, 0, 0), scale=(0.6, 0.6, 0.6)))
    _set_joints(e, posed_handles, poses, how)
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
    m[:3, :3] = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1e-9]]  # columns nearly coplanar
    assert np.array_equal(_normal_matrix(m), np.eye(3))


def test_normal_matrix_rule_keeps_small_uniform_scale():
    m = np.eye(4)
    m[:3, :3] = np.diag([0.002, 0.002, 0.002])  # det 8e-9, not singular
    assert not np.array_equal(_normal_matrix(m), np.eye(3))
    n = np.array([0.0, 0.6, 0.8])
    got = _normal_matrix(m) @ n
    assert np.allclose(got / np.linalg.norm(got), n)


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


@pytest.mark.parametrize("how", HOW)
def test_set_joints_before_device_is_kept_in_the_mirror(how):
    e = mx.Engine("skin", width=32, height=32)
    h = _spawn(e, _parts())
    _set_joints(e, [h], [_pose()[:2]], how)
    joints = e._render_pipeline._joints
    block = joints.block_of[h.index]
    assert block > 0
    assert np.allclose(joints.mirror[block, :2], _pose()[:2].transpose(0, 2, 1))
    assert np.array_equal(joints.mirror[block, 2:], np.tile(np.eye(4, dtype=np.float32), (30, 1, 1)))


@pytest.mark.parametrize("how", HOW)
def test_a_respawned_entity_does_not_inherit_a_block(how):
    e = mx.Engine("skin", width=32, height=32)
    h = _spawn(e, _parts())
    _set_joints(e, [h], [_pose()], how)
    e.destroy([h.index])
    e.commands.execute(e.store)
    h2 = _spawn(e, _parts())
    assert h2.index == h.index
    assert e._render_pipeline._joints.block_of[h2.index] == 0


@pytest.mark.parametrize("how", HOW)
def test_a_destroyed_entity_frees_its_block_for_the_next_pose(how):
    e = mx.Engine("skin", width=32, height=32)
    a, b = _spawn(e, _parts()), _spawn(e, _parts())
    _set_joints(e, [a, b], [_pose(), _pose()], how)
    joints = e._render_pipeline._joints
    freed = int(joints.block_of[a.index])
    e.destroy([a.index])
    e.commands.execute(e.store)
    assert np.array_equal(joints.mirror[freed], np.tile(np.eye(4, dtype=np.float32), (32, 1, 1)))
    c = _spawn(e, _parts())
    _set_joints(e, [c], [_pose()[:1]], how)
    assert joints.block_of[c.index] == freed
    assert joints.block_of[b.index] not in (0, freed)


def test_batched_set_joints_with_an_unskinned_entity_raises():
    e = mx.Engine("skin", width=32, height=32)
    a, box = _spawn(e, _parts()), _spawn(e, Mesh.box())
    with pytest.raises(ValueError, match="joints"):
        e.set_joints([a.index, box.index], np.stack([_pose(), _pose()]))


@pytest.mark.parametrize(
    "n, mats",
    [(1, _pose()), (2, np.stack([_pose()])), (1, np.zeros((1, 0, 4, 4))),
     (1, np.tile(np.eye(4), (1, 33, 1, 1))), (1, np.zeros((1, 3, 3, 3)))],
)
def test_batched_set_joints_bad_shape_raises(n, mats):
    e = mx.Engine("skin", width=32, height=32)
    hs = [_spawn(e, _parts()) for _ in range(n)]
    with pytest.raises(ValueError, match=r"\(N, J, 4, 4\)"):
        e.set_joints([h.index for h in hs], mats)


def test_batched_set_joints_grows_the_mirror_for_many_new_blocks():
    e = mx.Engine("skin", width=32, height=32)
    geo = _parts().to_geometry()
    first = e.spawn(MeshC(geo), Material(StandardMaterial(color="#ffffff")), Transform(), n=40)
    idx = np.arange(first.index, first.index + 40)
    mats = np.tile(np.eye(4, dtype=np.float32), (40, 3, 1, 1))
    mats[:, 0, 0, 3] = np.arange(40)  # each entity's joint 0 moved by its own x
    e.set_joints(idx, mats)
    joints = e._render_pipeline._joints
    blocks = joints.block_of[idx]
    assert len(np.unique(blocks)) == 40 and blocks.min() > 0
    assert np.allclose(joints.mirror[blocks, 0, 3, 0], np.arange(40))


def test_batched_set_joints_after_the_buffer_exists_uploads_every_pose(tmp_path):
    mesh = _parts()
    other = _pose()[[1, 0, 2]].copy()
    other[2] = _rot((0, 1, 0), 1.0, (0.8, 0, 0))
    e = _engine()
    geo = mesh.to_geometry()
    mat = StandardMaterial(color="#d0a060", roughness=0.6)
    hs = [e.spawn(MeshC(geo), Material(mat), Transform(pos=(x, 0, 0), scale=(0.6, 0.6, 0.6)))
          for x in (-0.7, 0.7)]
    e.set_joints([h.index for h in hs], np.stack([_pose(), other]))
    e.render_frame(supersample=1)  # creates the buffer, so the next call uploads
    e.set_joints([h.index for h in hs], np.stack([other, _pose()]))
    gpu = e.render_frame(supersample=1).astype(np.float64)
    cpu = _render([(mesh, other, -0.7), (mesh, _pose(), 0.7)], "plain", tmp_path, skinned=False)
    _assert_same(gpu, cpu)


# --- pixels ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["plain", "vcolor", "textured"])
@pytest.mark.parametrize("how", HOW)
def test_skinned_render_matches_cpu_posed_render(variant, how, tmp_path):
    mesh = _mesh_for(variant)
    gpu = _render([(mesh, _pose(), 0.0)], variant, tmp_path, skinned=True, how=how)
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


@pytest.mark.parametrize("how", HOW)
def test_two_entities_from_one_mesh_are_posed_independently(how, tmp_path):
    mesh = _parts()
    other = _pose()[[1, 0, 2]].copy()
    other[2] = _rot((0, 1, 0), 1.0, (0.8, 0, 0))
    scene = [(mesh, _pose(), -0.7), (mesh, other, 0.7)]
    gpu = _render(scene, "plain", tmp_path, skinned=True, how=how)
    cpu = _render(scene, "plain", tmp_path, skinned=False)
    _assert_same(gpu, cpu)


@pytest.mark.parametrize("how", HOW)
def test_fewer_joints_keep_the_previous_higher_matrices(how, tmp_path):
    mesh = _parts()
    first, second = _pose(), _pose()[[1]]
    e = _engine()
    h = e.spawn(MeshC(mesh.to_geometry()), Material(StandardMaterial(color="#d0a060", roughness=0.6)),
                Transform(scale=(0.6, 0.6, 0.6)))
    _set_joints(e, [h], [first], how)
    e.render_frame(supersample=1)
    _set_joints(e, [h], [second], how)
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


def test_set_geometry_with_joints_on_an_unskinned_entity_raises():
    e = mx.Engine("skin", width=32, height=32)
    box = Mesh.box()
    h = _spawn(e, box)
    jointed = Mesh(positions=box.positions, faces=box.faces, normals=box.normals,
                   joints=np.zeros(len(box.positions), dtype=int))
    with pytest.raises(ValueError, match="joints"):
        h.set_geometry(jointed)


# --- shadow pass -------------------------------------------------------------------------


def _turned():
    """Part 0 tipped onto its side, part 1 as it rests, part 2 swung forward and turned."""
    a = _rot((0, 0, 1), 1.4, (-0.8, -0.4, 0.0))
    c = _rot((0, 1, 0), 0.8, (0.8, 0.0, 0.0))
    c[:3, 3] += (0.3, 0.0, 0.9)
    return np.stack([a, np.eye(4), c]).astype(np.float32)


def _shadow_render(scene, skinned, labels=None):
    """Render [(mesh, mats or None, x)] over a ground plane with sun shadows.

    Returns the beauty frame, or the id labels when `labels` is True (one
    group per entity, in scene order).
    """
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as err:
        pytest.skip(f"offscreen canvas unavailable: {err}")
    e = mx.Engine("skin-shadow", width=96, height=72)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.6, -1.0, -0.3)))
    e.enable_shadows(target=(0, 0, 0), extent=3.0, resolution=1024, near=0.1, far=20.0,
                     pcf_radius=0, auto_fit=False)
    e.camera.set_pose(position=(0.0, 3.0, 2.6), target=(0.0, -0.5, 0.0))
    mat = StandardMaterial(color="#d0a060", roughness=0.6)
    e.spawn(MeshC(Mesh.plane(6.0, 6.0).translate((0.0, -0.5, 0.0)).to_geometry()),
            Material(StandardMaterial(color="#cfcfcf", roughness=0.9)), Transform())
    geos, handles = {}, []
    for mesh, mats, x in scene:
        t = Transform(pos=(x, 0.0, 0.0), scale=(0.6, 0.6, 0.6))
        if skinned:
            geo = geos.setdefault(id(mesh), mesh.to_geometry())
            h = e.spawn(MeshC(geo), Material(mat), t)
            if mats is not None:
                h.set_joints(mats)
        else:
            posed = mesh if mats is None else _cpu_pose(mesh, mats)
            h = e.spawn(MeshC(posed.to_geometry()), Material(mat), t)
        handles.append(h)
    if labels:
        return e.render_frame(supersample=1, pass_="ids", groups=[[h.index] for h in handles])
    return e.render_frame(supersample=1).astype(np.float64)


def _shadow_mask(scene):
    """Ground pixels the CPU-posed scene darkens with its own shadow."""
    lit = _shadow_render([(m, mats, x + 40.0) for m, mats, x in scene], skinned=False)  # casters out of frame
    shaded = _shadow_render(scene, skinned=False)
    ids = _shadow_render(scene, skinned=False, labels=True)
    return ((lit - shaded).mean(axis=2) > 20) & (ids == 0)


def test_skinned_shadow_matches_cpu_posed_shadow():
    mesh = _parts()
    scene = [(mesh, _turned(), 0.0)]
    gpu = _shadow_render(scene, skinned=True)
    cpu = _shadow_render(scene, skinned=False)
    _assert_same(gpu, cpu)
    # The shadow is the posed one, not the rest pose's: where the posed parts
    # cast shadow on the ground, the unposed figure's frame is lit.
    mask = _shadow_mask(scene)
    assert mask.sum() > 40
    rest = _shadow_render([(mesh, None, 0.0)], skinned=True)
    assert np.abs(rest - gpu).mean(axis=2)[mask].mean() > 20


def test_two_skinned_entities_cast_their_own_shadows():
    mesh = _parts()
    other = _turned()[[2, 1, 0]].copy()
    other[0] = _rot((1, 0, 0), -1.2, (-0.8, -0.4, 0.0))
    scene = [(mesh, _turned(), -0.9), (mesh, other, 0.9)]
    _assert_same(_shadow_render(scene, skinned=True), _shadow_render(scene, skinned=False))


def test_unskinned_shadow_pipeline_uses_the_shadow_wgsl_unchanged():
    from manifoldx.renderer import _SHADOW_SHADER
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as err:
        pytest.skip(f"offscreen canvas unavailable: {err}")
    e = mx.Engine("skin-shadow", width=32, height=32)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.6, -1.0, -0.3)))
    e.enable_shadows()
    mat = StandardMaterial(color="#d0a060")
    e.spawn(MeshC(Mesh.box().to_geometry()), Material(mat), Transform(pos=(-1, 0, 0)))
    e.spawn(MeshC(_parts().to_geometry()), Material(mat), Transform(pos=(1, 0, 0)))
    e.render_frame(supersample=1)
    sources = e._render_pipeline._shadow_shader_sources
    assert [s for (stride, skinned), s in sources.items() if not skinned] == [_SHADOW_SHADER]
    assert len([s for (stride, skinned), s in sources.items() if skinned]) == 1


# --- id pass -----------------------------------------------------------------------------


def test_skinned_ids_match_cpu_posed_ids():
    mesh = _parts()
    scene = [(mesh, _turned(), 0.0)]
    gpu = _shadow_render(scene, skinned=True, labels=True)
    cpu = _shadow_render(scene, skinned=False, labels=True)
    assert (gpu != cpu).sum() <= 4


def test_a_turned_part_is_picked_where_it_is_posed():
    mesh = _parts()
    e = _engine()
    h = e.spawn(MeshC(mesh.to_geometry()), Material(StandardMaterial(color="#d0a060")),
                Transform(scale=(0.6, 0.6, 0.6)))
    up = np.stack([np.eye(4)] * 3)
    up[0, :3, 3] = (0.0, 1.1, 0.0)  # part 0 lifted well above its rest place
    h.set_joints(up)
    labels = e.render_frame(supersample=1, pass_="ids", groups=[[h.index]])
    (px_posed, px_rest), _ = e.camera.project(
        np.array([[-0.8, 1.1, 0.0], [-0.8, 0.0, 0.0]]) * 0.6, e.w, e.h)
    (cx, cy), (rx, ry) = np.floor(px_posed).astype(int), np.floor(px_rest).astype(int)
    assert labels[cy, cx] == 1
    assert labels[ry, rx] == 0


def test_two_skinned_entities_are_picked_where_each_is_posed():
    mesh = _parts()
    other = _turned()[[2, 1, 0]].copy()
    other[0] = _rot((1, 0, 0), -1.2, (-0.8, -0.4, 0.0))
    scene = [(mesh, _turned(), -0.9), (mesh, other, 0.9)]
    gpu = _shadow_render(scene, skinned=True, labels=True)
    cpu = _shadow_render(scene, skinned=False, labels=True)
    assert set(np.unique(cpu)) == {0, 1, 2}
    assert (gpu != cpu).sum() <= 6


def test_shadow_auto_fit_bounds_hold_the_posed_parts():
    e = mx.Engine("skin", width=32, height=32)
    h = _spawn(e, _parts())
    lifted = np.stack([np.eye(4)] * 3)
    lifted[0, :3, 3] = (0.0, 5.0, 0.0)  # part 0 far above the rest pose's box
    h.set_joints(lifted)
    e2 = mx.Engine("skin", width=32, height=32)
    _spawn(e2, _cpu_pose(_parts(), lifted.astype(np.float32)))
    center, radius = e._render_pipeline._scene_bounds(e)
    center2, radius2 = e2._render_pipeline._scene_bounds(e2)
    assert np.allclose(center, center2, atol=1e-4) and abs(radius - radius2) < 1e-4
