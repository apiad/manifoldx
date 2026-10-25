"""Which render pass draws each entity, in which batch and order (issue #111 pins these)."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx import renderer
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import FlatMaterial, StandardMaterial, cube, sphere
from manifoldx.viz import (
    AxisFrame,
    AxisMaterial,
    ColormapMaterial,
    LabelMaterial,
    PointCloud,
    TextLabel,
    Volume,
    VolumeMaterial,
)

FAR = (1000.0, 0.0, 0.0)  # outside the camera's frustum


def _engine():
    engine = mx.Engine("test", width=32, height=32, max_entities=64)
    try:
        engine._ensure_offscreen(1)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine.camera.fit(5)
    return engine


@pytest.fixture
def captured(monkeypatch):
    """The batches each pass receives on the last frame, as entity indices."""
    out = {}

    def entities(rp, local):
        return [int(rp._capture_alive[i]) for i in local]

    real = renderer.RenderPipeline._render_scene_passes

    def scene(self, engine, render_pass):
        out.clear()
        self._capture_alive = np.where(self._store._alive)[0]
        return real(self, engine, render_pass)

    monkeypatch.setattr(renderer.RenderPipeline, "_render_scene_passes", scene)
    monkeypatch.setattr(renderer._mesh_pass, "render_mesh_batches",
                        lambda rp, e, p, b, m, d, a: out.__setitem__(
                            "mesh", [(k, [int(a[i]) for i in v]) for k, v in b.items()]))
    for name, mod, fn in [("sprite", renderer._sprite_pass, "render_sprite_batches"),
                          ("label", renderer._label_pass, "render_label_pass"),
                          ("axis", renderer._axis_pass, "render_axis_pass")]:
        monkeypatch.setattr(mod, fn, lambda rp, e, p, b, m, d, name=name: out.__setitem__(
            name, [(k, entities(rp, v)) for k, v in b.items()]))
    monkeypatch.setattr(renderer._volume_pass, "render_volume_pass",
                        lambda rp, e, p, b: out.__setitem__(
                            "volume", [(entities(rp, [i])[0], m, v) for i, m, v in b]))
    return out


def _spawn(engine, *components, n=1):
    first = engine.store._alive.sum()
    engine.spawn(*components, n=n)
    return list(range(int(first), int(first) + n))


def _frame(engine):
    engine._draw_frame()
    engine._render_canvas.draw()


def _mesh_key(engine, ent):
    store = engine.store
    geom = int(store._components["Mesh"][ent, 0])
    mid = int(store._components["Material"][ent, 0])
    mat = engine._material_registry.get(mid)
    return (geom, type(mat).__name__, getattr(mat, "pipeline_subtype", None), mid)


def _mixed(engine):
    """Meshes, sprites, an axis, labels and a volume, spawned interleaved."""
    a, b, c = StandardMaterial("#ff0000"), FlatMaterial((0, 1, 0)), StandardMaterial("#0000ff")
    box = cube(1, 1, 1)  # one geometry: same (geometry, material) is one batch
    ents = {}
    ents["cubes_a"] = _spawn(engine, Mesh(box), Material(a), Transform(pos=(0, 0, 0)), n=3)
    ents["sprites"] = _spawn(engine, PointCloud(), Material(ColormapMaterial("viridis", 0, 1)),
                             Transform(pos=(0, 0, 0)), n=2)
    ents["spheres_b"] = _spawn(engine, Mesh(sphere(0.5, 8)), Material(b), Transform(pos=(0, 0, 0)), n=2)
    ents["culled_a"] = _spawn(engine, Mesh(box), Material(a), Transform(pos=FAR), n=2)
    ents["cube_c"] = _spawn(engine, Mesh(box), Material(c), Transform(pos=(0, 0, 0)))
    axis_geom = engine._geometry_registry.get_by_name("axis_line_x")
    ents["axis"] = _spawn(engine, Mesh(axis_geom), Material(AxisMaterial()), Transform(), AxisFrame(extent=1.0))
    # A label with a mesh is still a label; a point cloud with a label material too.
    ents["label_mesh"] = _spawn(engine, TextLabel(), Mesh(box), Material(LabelMaterial()), Transform())
    ents["label_cloud"] = _spawn(engine, TextLabel(), PointCloud(), Material(LabelMaterial()), Transform())
    vol = engine.register_volume(np.ones((4, 4, 4), np.float32))
    ents["volume"] = _spawn(engine, Volume(volume_id=vol), Mesh(box),
                            Material(VolumeMaterial()), Transform())
    ents["cubes_a2"] = _spawn(engine, Mesh(box), Material(a), Transform(pos=(0, 0, 0)), n=2)
    return ents


def test_mixed_scene_routes_each_entity_to_its_pass(captured):
    engine = _engine()
    ents = _mixed(engine)
    _frame(engine)
    engine.destroy([ents["sprites"][0], ents["cubes_a"][1]])
    _frame(engine)
    _frame(engine)

    # Batches in order of each one's first entity; entities in index order; culled ones gone.
    cubes_a = [ents["cubes_a"][0], ents["cubes_a"][2]] + ents["cubes_a2"]
    assert captured["mesh"] == [
        (_mesh_key(engine, cubes_a[0]), cubes_a),
        (_mesh_key(engine, ents["spheres_b"][0]), ents["spheres_b"]),
        (_mesh_key(engine, ents["cube_c"][0]), ents["cube_c"]),
    ]
    sprite_mid = int(engine.store._components["Material"][ents["sprites"][1], 0])
    assert captured["sprite"] == [(sprite_mid, ents["sprites"][1:])]
    label_mids = [int(engine.store._components["Material"][e[0], 0])
                  for e in (ents["label_mesh"], ents["label_cloud"])]
    assert captured["label"] == [(label_mids[0], ents["label_mesh"]), (label_mids[1], ents["label_cloud"])]
    axis_ent = ents["axis"][0]
    axis_key = (int(engine.store._components["Mesh"][axis_ent, 0]),
                int(engine.store._components["Material"][axis_ent, 0]))
    assert captured["axis"] == [(axis_key, ents["axis"])]
    vol_ent = ents["volume"][0]
    vol_mid = int(engine.store._components["Material"][vol_ent, 0])
    assert captured["volume"] == [(vol_ent, vol_mid, int(engine.store._components["Volume"][vol_ent, 0]))]
    assert engine._render_pipeline.drawn == 7


def test_meshes_only_scene_drops_culled_meshes(captured):
    engine = _engine()
    a, b = StandardMaterial("#ff0000"), FlatMaterial((0, 1, 0))
    seen = _spawn(engine, Mesh(cube(1, 1, 1)), Material(a), Transform(pos=(0, 0, 0)), n=3)
    culled = _spawn(engine, Mesh(cube(1, 1, 1)), Material(a), Transform(pos=FAR), n=2)
    spheres = _spawn(engine, Mesh(sphere(0.5, 8)), Material(b), Transform(pos=(0, 0, 0)), n=2)
    _spawn(engine, Mesh(sphere(0.5, 8)), Material(b), Transform(pos=FAR))
    _frame(engine)
    _frame(engine)
    assert captured["mesh"] == [
        (_mesh_key(engine, seen[0]), seen),
        (_mesh_key(engine, spheres[0]), spheres),
    ]
    assert not set(culled) & {e for _, v in captured["mesh"] for e in v}
    assert engine._render_pipeline.drawn == 5
    assert set(captured) == {"mesh"}


def test_translucent_batches_draw_after_opaque_ones(captured):
    engine = _engine()
    glass = StandardMaterial("#ffffff", opacity=0.5)
    clear = _spawn(engine, Mesh(cube(1, 1, 1)), Material(glass), Transform(pos=(0, 0, 0)), n=2)
    solid = _spawn(engine, Mesh(cube(1, 1, 1)), Material(StandardMaterial("#ff0000")),
                   Transform(pos=(0, 0, 0)), n=2)
    _frame(engine)
    _frame(engine)
    assert [v for _, v in captured["mesh"]] == [solid, clear]


def test_sprites_only_scene(captured):
    engine = _engine()
    a, b = ColormapMaterial("viridis", 0, 1), ColormapMaterial("magma", 0, 1)
    first = _spawn(engine, PointCloud(), Material(a), Transform(pos=(0, 0, 0)), n=3)
    second = _spawn(engine, PointCloud(), Material(b), Transform(pos=FAR), n=2)  # sprites are not culled
    more = _spawn(engine, PointCloud(), Material(a), Transform(pos=(0, 0, 0)))
    _frame(engine)
    _frame(engine)
    mid = engine.store._components["Material"]
    assert captured["sprite"] == [(int(mid[first[0], 0]), first + more), (int(mid[second[0], 0]), second)]
    assert set(captured) == {"sprite"}


def test_id_pass_mask_is_meshes_with_geometry_and_no_special_material():
    engine = _engine()
    ents = _mixed(engine)
    _frame(engine)
    alive = np.where(engine.store._alive)[0]
    mask = engine._mesh_path_entities(alive)
    want = set(ents["cubes_a"] + ents["spheres_b"] + ents["culled_a"] + ents["cube_c"] + ents["cubes_a2"])
    assert mask.dtype == bool and len(mask) == len(alive)
    assert set(alive[mask].tolist()) == want
