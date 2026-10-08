from pathlib import Path

from manifoldx.cli import main

from gltf_helpers import load

APP = str(Path(__file__).parent / "fixtures" / "export_app.py")


def test_exports_instead_of_running(tmp_path):
    out = tmp_path / "s.glb"
    assert main(["export", APP, str(out)]) == 0
    assert load(out).nodes[0].translation == [0, 1, 0]


def test_strict_exit_code(tmp_path):
    out = tmp_path / "s.glb"
    assert main(["export", APP, str(out), "--strict", "--", "--basic"]) == 1
    assert not out.exists()
    assert main(["export", APP, str(out), "--", "--basic"]) == 0 and out.exists()


def test_app_that_never_runs(tmp_path):
    assert main(["export", APP, str(tmp_path / "s.glb"), "--", "--no-run"]) == 2


def test_module_form(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(APP).parent))
    out = tmp_path / "s.glb"
    assert main(["export", "-m", "export_app", str(out)]) == 0 and out.exists()


def test_engine_run_is_restored(tmp_path):
    import manifoldx as mx

    before = mx.Engine.run
    main(["export", APP, str(tmp_path / "s.glb")])
    assert mx.Engine.run is before
