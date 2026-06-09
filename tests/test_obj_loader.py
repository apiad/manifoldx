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
