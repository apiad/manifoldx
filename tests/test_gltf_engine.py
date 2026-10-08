import json

import manifoldx as mx
import numpy as np
import pytest
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gltf import ExportIncomplete
from manifoldx.gui import Panel, Text, ValueDisplay
from manifoldx.resources import BasicMaterial, DirectionalLight, StandardMaterial, cube

from gltf_helpers import load


def scene():
    e = mx.Engine("t", width=64, height=48)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=2.0, direction=(0, -1, 0)))
    e.enable_fog(10, 20)
    e.enable_shadows(resolution=1024, bias=0.002, pcf_radius=2)
    geo = cube(1, 1, 1)
    a = e.spawn(Mesh(geo), Material(StandardMaterial("#ffffff")), Transform(pos=(1, 2, 3)))
    b = e.spawn(Mesh(geo), Material(StandardMaterial("#ff0000")), Transform(pos=(4, 0, 0)))
    return e, a, b


def test_fresh_engine_reports_nothing(tmp_path):
    e = mx.Engine("t", width=64, height=48)
    report = e.export_gltf(tmp_path / "s.glb")
    assert len(report) == 0
    assert json.loads((tmp_path / "s.export-report.json").read_text()) == {"complete": True, "entries": []}


def test_entities_names_extras_and_scene_state(tmp_path):
    e, a, b = scene()
    e.export_gltf(tmp_path / "s.glb", names={a.index: "a"}, extras={a.index: {"pick": "p1"}})
    g = load(tmp_path / "s.glb")
    named = {n.name: n for n in g.nodes}
    assert named["a"].translation == [1, 2, 3] and named["a"].extras == {"pick": "p1"}
    assert named[f"entity_{b.index}"].translation == [4, 0, 0]
    assert len(g.meshes) == 2  # one geometry, two materials
    extras = g.scenes[0].extras["manifoldx"]
    assert extras["fog"] == {"start": 10.0, "end": 20.0, "color": list(e.fog_color)}
    assert extras["shadows"]["resolution"] == 1024 and extras["background"] == list(e.background_color)
    assert any(n.camera is not None for n in g.nodes)
    assert g.extensions["KHR_lights_punctual"]["lights"][0]["type"] == "directional"


def test_code_and_unsupported_things_are_reported(tmp_path):
    e, _, _ = scene()

    @e.system
    def spin(query: mx.Query[Transform], dt: float):
        pass

    @e.on("key_down")
    def keys(payload):
        pass

    e.gui.append(Panel(children=[Text("hi"), ValueDisplay(getter=lambda: "x")]))
    e.spawn(Mesh(cube(1, 1, 1)), Material(BasicMaterial("#00ff00")), Transform())
    report = e.export_gltf(tmp_path / "s.glb")
    kinds = sorted((r.kind, r.effect) for r in report)
    assert kinds == sorted([("system", "dropped"), ("handler", "dropped"), ("gui", "dropped"),
                            ("gui", "dropped"), ("material", "approximated")])
    system = next(r for r in report if r.kind == "system")
    assert system.name.endswith("spin") and system.where.endswith(f"test_gltf_engine.py:{spin.__code__.co_firstlineno}")
    assert any(r.name.startswith("ValueDisplay.getter") for r in report)


def test_strict_raises_and_writes_nothing(tmp_path):
    e, _, _ = scene()
    e.spawn(Mesh(cube(1, 1, 1)), Material(BasicMaterial("#00ff00")), Transform())
    with pytest.raises(ExportIncomplete) as info:
        e.export_gltf(tmp_path / "s.glb", strict=True)
    assert len(info.value.report) == 1
    assert not (tmp_path / "s.glb").exists() and not (tmp_path / "s.export-report.json").exists()


def test_startup_spawns_are_exported(tmp_path):
    from manifoldx.backends import get_offscreen_canvas

    try:
        get_offscreen_canvas(width=8, height=8)
    except Exception:
        pytest.skip("no offscreen wgpu backend")
    e = mx.Engine("t", width=64, height=48)

    @e.on("startup")
    def spawn(_payload):
        e.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial("#ffffff")), Transform())

    report = e.export_gltf(tmp_path / "s.glb")
    assert len(load(tmp_path / "s.glb").meshes) == 1 and len(report) == 0


def test_names_and_extras_can_be_functions(tmp_path):
    e, a, b = scene()
    e.export_gltf(tmp_path / "s.glb", names=lambda i: "first" if i == a.index else None,
                  extras=lambda i: {"pick": f"p{i}"} if i == b.index else None)
    named = {n.name: n for n in load(tmp_path / "s.glb").nodes}
    assert "first" in named and not named["first"].extras  # pygltflib reads absent extras as {}
    assert named[f"entity_{b.index}"].extras == {"pick": f"p{b.index}"}


def test_entities_without_a_mesh_are_reported(tmp_path):
    from manifoldx.viz import ColormapMaterial, PointCloud, VolumeMaterial
    from manifoldx.viz.components import Volume

    e = mx.Engine("t", width=64, height=48)
    vol = e.register_volume(np.zeros((4, 4, 4), np.float32))
    e.spawn(Volume(volume_id=vol), Material(VolumeMaterial()), Transform())
    e.spawn(PointCloud(), Material(ColormapMaterial(cmap="inferno", vmin=0.0, vmax=1.0)), Transform(pos=np.zeros((5, 3), np.float32)), n=5)
    report = e.export_gltf(tmp_path / "s.glb")
    rows = {(r.kind, r.name, r.effect, r.count) for r in report}
    assert ("entity", "VolumeMaterial", "dropped", 1) in rows
    assert ("entity", "ColormapMaterial", "dropped", 5) in rows
    with pytest.raises(ExportIncomplete):
        e.export_gltf(tmp_path / "t.glb", strict=True)


def test_mesh_without_material_is_written_like_the_renderer_draws_it(tmp_path):
    e = mx.Engine("t", width=64, height=48)
    e.spawn(Mesh(cube(1, 1, 1)), Transform())
    report = e.export_gltf(tmp_path / "s.glb")
    assert len(load(tmp_path / "s.glb").meshes) == 1
    assert [(r.kind, r.name, r.effect) for r in report] == [("material", "BasicMaterial", "approximated")]
