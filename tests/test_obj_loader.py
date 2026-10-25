import numpy as np
import pytest


SINGLE_TRIANGLE = """\
# tiny test OBJ
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.0 1.0
f 1/1/1 2/2/1 3/3/1
"""


def test_parses_single_triangle(tmp_path):
    from manifoldx.assets.obj import load_obj

    obj_path = tmp_path / "tri.obj"
    obj_path.write_text(SINGLE_TRIANGLE)

    geo = load_obj(obj_path)
    assert set(geo.keys()) >= {"positions", "normals", "uvs", "indices", "name"}
    assert geo["positions"].shape == (3, 3)
    assert geo["normals"].shape == (3, 3)
    assert geo["uvs"].shape == (3, 2)
    assert geo["indices"].shape == (3,)
    assert geo["indices"].dtype == np.uint32
    assert geo["positions"].dtype == np.float32


QUAD = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1/1 2/2/1 3/3/1 4/4/1
"""


def test_quad_fan_triangulates(tmp_path):
    from manifoldx.assets.obj import load_obj
    obj_path = tmp_path / "quad.obj"
    obj_path.write_text(QUAD)

    geo = load_obj(obj_path)
    assert geo["indices"].shape == (6,)
    assert geo["positions"].shape == (4, 3)


def test_mixed_face_forms_raise(tmp_path):
    from manifoldx.assets.obj import load_obj, ObjParseError
    src = "\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "vn 0 0 1",
        "vt 0 0", "vt 1 0", "vt 0 1",
        "f 1/1/1 2/2/1 3/3/1",
        "f 1//1 2//1 3//1",
    ])
    obj_path = tmp_path / "mixed.obj"
    obj_path.write_text(src)

    with pytest.raises(ObjParseError, match="form changed"):
        load_obj(obj_path)


def test_negative_indices_raise(tmp_path):
    from manifoldx.assets.obj import load_obj, ObjParseError
    src = "\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "f -3 -2 -1",
    ])
    obj_path = tmp_path / "neg.obj"
    obj_path.write_text(src)

    with pytest.raises(ObjParseError, match="negative"):
        load_obj(obj_path)


def test_v_double_slash_n_form_no_uvs(tmp_path):
    from manifoldx.assets.obj import load_obj
    src = "\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "vn 0 0 1",
        "f 1//1 2//1 3//1",
    ])
    obj_path = tmp_path / "no_uvs.obj"
    obj_path.write_text(src)

    geo = load_obj(obj_path)
    assert "uvs" not in geo
    assert "normals" in geo


def test_ignored_directives(tmp_path):
    from manifoldx.assets.obj import load_obj
    src = "\n".join([
        "mtllib teapot.mtl",
        "o Teapot",
        "g body",
        "v 0 0 0", "v 1 0 0", "v 0 1 0",
        "vn 0 0 1",
        "vt 0 0", "vt 1 0", "vt 0 1",
        "usemtl porcelain",
        "s 1",
        "f 1/1/1 2/2/1 3/3/1",
    ])
    obj_path = tmp_path / "noisy.obj"
    obj_path.write_text(src)

    geo = load_obj(obj_path)
    assert geo["positions"].shape == (3, 3)


def test_top_level_export():
    import manifoldx as mx
    assert hasattr(mx, "load_obj")
    assert hasattr(mx, "load_texture")


def test_uvs_convert_obj_bottom_left_origin_to_top_left(tmp_path):
    """OBJ puts v=0 at the bottom of the image; textures are uploaded top row
    first, so the loader must return v' = 1 - v (issue #7)."""
    from manifoldx.assets.obj import load_obj
    obj_path = tmp_path / "quad.obj"
    obj_path.write_text(QUAD)

    geo = load_obj(obj_path)
    uv_at = {tuple(p): tuple(uv) for p, uv in zip(geo["positions"].tolist(), geo["uvs"].tolist())}
    assert uv_at[(0.0, 0.0, 0.0)] == (0.0, 1.0)   # vt 0 0: bottom-left of the image
    assert uv_at[(1.0, 1.0, 0.0)] == (1.0, 0.0)   # vt 1 1: top-right of the image


# --- Pinned output (manifoldx#117): exact arrays for each face form, the
# first-occurrence dedup order, fan triangulation and the error messages.

def _load(tmp_path, src, name="m.obj"):
    from manifoldx.assets.obj import load_obj
    path = tmp_path / name
    path.write_text(src)
    return load_obj(path)


P = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.5]]
T = [[0.0, 0.0], [1.0, 0.25], [1.0, 1.0], [0.5, 1.0]]
N = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
HEAD = (
    "".join(f"v {x} {y} {z}\n" for x, y, z in P)
    + "".join(f"vt {u} {v}\n" for u, v in T)
    + "".join(f"vn {x} {y} {z}\n" for x, y, z in N)
)


def _check(geo, name, pos, idx, uv=None, nrm=None):
    assert list(geo) == ["name", "positions", "indices"] + (["normals"] if nrm is not None else []) + (
        ["uvs"] if uv is not None else []
    )
    assert geo["name"] == name
    assert geo["positions"].dtype == np.float32 and geo["indices"].dtype == np.uint32
    np.testing.assert_array_equal(geo["positions"], np.array([P[i] for i in pos], np.float32))
    np.testing.assert_array_equal(geo["indices"], np.array(idx, np.uint32))
    if uv is not None:
        assert geo["uvs"].dtype == np.float32
        np.testing.assert_array_equal(geo["uvs"], np.array([[T[i][0], 1.0 - T[i][1]] for i in uv], np.float32))
    if nrm is not None:
        assert geo["normals"].dtype == np.float32
        np.testing.assert_array_equal(geo["normals"], np.array([N[i] for i in nrm], np.float32))


def test_pinned_v_form_dedups_in_first_occurrence_order(tmp_path):
    geo = _load(tmp_path, HEAD + "f 3 1 2\nf 3 2 4\nf 4 2 1\n", "plain.obj")
    _check(geo, "plain", pos=[2, 0, 1, 3], idx=[0, 1, 2, 0, 2, 3, 3, 2, 1])


def test_pinned_v_vt_form(tmp_path):
    geo = _load(tmp_path, HEAD + "f 3/1 1/2 2/3\nf 3/1 2/3 4/4\nf 3/2 1/2 2/3\n")
    _check(geo, "m", pos=[2, 0, 1, 3, 2], idx=[0, 1, 2, 0, 2, 3, 4, 1, 2], uv=[0, 1, 2, 3, 1])


def test_pinned_v_vt_vn_form(tmp_path):
    geo = _load(tmp_path, HEAD + "f 3/1/1 1/2/1 2/3/1\nf 3/1/1 2/3/1 4/4/2\nf 3/1/2 1/2/1 2/3/1\n")
    _check(geo, "m", pos=[2, 0, 1, 3, 2], idx=[0, 1, 2, 0, 2, 3, 4, 1, 2],
           uv=[0, 1, 2, 3, 0], nrm=[0, 0, 0, 1, 1])


def test_pinned_v_double_slash_vn_form(tmp_path):
    geo = _load(tmp_path, HEAD + "f 3//1 1//1 2//1\nf 3//2 2//1 4//2\n")
    _check(geo, "m", pos=[2, 0, 1, 2, 3], idx=[0, 1, 2, 3, 2, 4], nrm=[0, 0, 0, 1, 1])


def test_pinned_fan_triangulation_of_a_pentagon_and_a_triangle(tmp_path):
    src = HEAD + "v 0.5 0.5 1\nf 1 2 3 4 5\nf 5 1 3\n"
    geo = _load(tmp_path, src)
    np.testing.assert_array_equal(geo["indices"], [0, 1, 2, 0, 2, 3, 0, 3, 4, 4, 0, 2])
    assert geo["positions"].shape == (5, 3)


def test_pinned_extra_components_and_whitespace(tmp_path):
    """v with a w or a colour, vt with a w, tabs, indents and trailing blanks:
    only the first three (v, vn) or two (vt) numbers count."""
    src = (
        "  v 0 0 0 1 0.5 0.5\nv\t1 0 0 1\nv 1 1 0   \n"
        "vt 0 0 0\nvt 1 0.25 0\nvt 1 1 0\n"
        "\tf 1/1 2/2 3/3  \n"
    )
    geo = _load(tmp_path, src)
    np.testing.assert_array_equal(geo["positions"], np.array(P[:3], np.float32))
    np.testing.assert_array_equal(geo["uvs"], np.array([[0, 1], [1, 0.75], [1, 0]], np.float32))
    np.testing.assert_array_equal(geo["indices"], [0, 1, 2])


@pytest.mark.parametrize("faces, match", [
    ("f 1/1/1 2/2/1 3/3/1\nf 1//1 2//1 3//1\n", r"line 12: face-line form changed from 'v/vt/vn' to 'v//vn'"),
    ("f 1/1 2/2/1 3/3\n", r"line 11: mixed face-line forms within a single face: \['v/vt', 'v/vt/vn'\]"),
    ("f 1 2\n", r"line 11: face needs at least 3 vertices, got 2"),
    ("f 1/1/1/1 2 3\n", r"line 11: malformed face token '1/1/1/1'"),
    ("f 1 -1 2\n", r"line 11: negative face indices not supported in v1"),
])
def test_pinned_error_messages(tmp_path, faces, match):
    from manifoldx.assets.obj import ObjParseError
    with pytest.raises(ObjParseError, match=match):
        _load(tmp_path, HEAD + faces)


def test_pinned_form_change_after_triangles_is_reported_at_its_line(tmp_path):
    from manifoldx.assets.obj import ObjParseError
    src = HEAD + "f 1 2 3\n" * 3 + "f 1/1 2/2 3/3\n"
    with pytest.raises(ObjParseError, match=r"line 14: face-line form changed from 'v' to 'v/vt'"):
        _load(tmp_path, src)


def test_pinned_mixed_triangles_and_quads(tmp_path):
    geo = _load(tmp_path, HEAD + "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/2 2/2/1\n")
    np.testing.assert_array_equal(geo["indices"], [0, 1, 2, 0, 2, 3, 0, 3, 1])
    np.testing.assert_array_equal(geo["normals"][3], N[1])
