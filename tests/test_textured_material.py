import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


def _make_solid_png(path, rgba):
    Image.new("RGBA", (4, 4), rgba).save(path, format="PNG")


def _make_offscreen_engine(name="textured-test", w=128, h=128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine(name, width=w, height=h)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_textured_sphere_renders():
    """A textured StandardMaterial on a UV sphere fires one frame without
    raising. Smoke test that all the texture plumbing lit up end-to-end."""
    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    from manifoldx.resources import StandardMaterial, PointLight, sphere
    from manifoldx.textures import load_texture

    engine = _make_offscreen_engine("textured-smoke")

    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / "blue.png"
        _make_solid_png(png_path, (40, 70, 220, 255))

        tex = load_texture(engine, png_path)
        sphere_geo = sphere(1.0, segments=16)
        mat = StandardMaterial(color="#ffffff", roughness=0.4, metallic=0.0,
                               albedo_map=tex)

        engine.spawn(Mesh(sphere_geo), Material(mat), Transform(pos=(0, 0, 0)))
        engine.add_light(PointLight(position=(2, 3, 3), color="#ffffff", intensity=25.0))

        engine._draw_frame()


def test_scalar_and_textured_coexist():
    """One scalar and one textured StandardMaterial in the same scene must
    both render — they get separate pipeline-cache entries (Task 4 fix) and
    separate bind groups (Task 7 fix)."""
    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    from manifoldx.resources import StandardMaterial, PointLight, sphere

    engine = _make_offscreen_engine("coexist")

    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / "g.png"
        _make_solid_png(png_path, (0, 255, 0, 255))
        from manifoldx.textures import load_texture
        tex = load_texture(engine, png_path)

        sphere_geo = sphere(1.0, segments=8)
        scalar_mat = StandardMaterial(color="#ff0000", roughness=0.5)
        textured_mat = StandardMaterial(color="#ffffff", roughness=0.5, albedo_map=tex)

        engine.spawn(Mesh(sphere_geo), Material(scalar_mat), Transform(pos=(-1.5, 0, 0)))
        engine.spawn(Mesh(sphere_geo), Material(textured_mat), Transform(pos=(1.5, 0, 0)))
        engine.add_light(PointLight(position=(0, 3, 3), color="#ffffff", intensity=25.0))

        engine._draw_frame()


def test_textured_material_on_no_uv_geometry_raises():
    """A textured StandardMaterial bound to a geometry without UVs raises
    MaterialGeometryMismatchError at pipeline-cache time, not silently
    rendering garbage."""
    import manifoldx as mx
    from manifoldx.components import Transform, Mesh, Material
    from manifoldx.resources import StandardMaterial, PointLight, cube
    from manifoldx.textures import load_texture
    from manifoldx.renderer import MaterialGeometryMismatchError

    engine = _make_offscreen_engine("mismatch")

    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / "w.png"
        _make_solid_png(png_path, (255, 255, 255, 255))
        tex = load_texture(engine, png_path)

        # cube() has no UVs in v1.
        engine.spawn(
            Mesh(cube(1, 1, 1)),
            Material(StandardMaterial(color="#ff0000", albedo_map=tex)),
            Transform(pos=(0, 0, 0)),
        )
        engine.add_light(PointLight(position=(2, 2, 2), color="#ffffff", intensity=20.0))

        with pytest.raises(MaterialGeometryMismatchError, match="UVs"):
            engine._draw_frame()
