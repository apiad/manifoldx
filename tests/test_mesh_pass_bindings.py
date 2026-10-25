"""The mesh pass keeps drawing what changed: material edits, swapped materials,
textures, skinned joints and a grown transforms buffer, frame after frame."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import BasicMaterial, DirectionalLight, StandardMaterial, cube


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("mesh-pass-bindings", width=64, height=64)
    engine._init_canvas(canvas)
    engine._running = True
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    engine.set_sun(DirectionalLight("#ffffff", 3.0, (0, 0, -1)))
    return engine, canvas


def _pixel(engine, canvas, x=32, y=32):
    engine._draw_frame()
    return np.asarray(canvas.draw()).astype(int)[y, x, :3]


def _empty(engine, canvas):
    """True when the centre pixel shows the background (the corner's colour)."""
    engine._draw_frame()
    img = np.asarray(canvas.draw()).astype(int)
    return np.abs(img[32, 32, :3] - img[0, 0, :3]).max() < 8


def _is(rgb, channel):
    others = [c for c in range(3) if c != channel]
    return rgb[channel] > 60 and all(rgb[channel] > 2 * rgb[c] + 10 for c in others)


def test_editing_a_material_colour_reaches_the_next_frame():
    engine, canvas = _engine()
    mat = StandardMaterial("#ff0000", roughness=0.8)
    engine.spawn(Mesh(cube(2, 2, 2)), Material(mat), Transform())
    assert _is(_pixel(engine, canvas), 0)
    assert _is(_pixel(engine, canvas), 0)
    mat.color = "#00ff00"
    assert _is(_pixel(engine, canvas), 1)
    mat.color = "#0000ff"
    assert _is(_pixel(engine, canvas), 2)


def test_set_material_between_two_of_one_type_switches_back_and_forth():
    engine, canvas = _engine()
    red = StandardMaterial("#ff0000", roughness=0.8)
    green = StandardMaterial("#00ff00", roughness=0.8)
    h = engine.spawn(Mesh(cube(2, 2, 2)), Material(red), Transform())
    assert _is(_pixel(engine, canvas), 0)
    h.set_material(green)
    assert _is(_pixel(engine, canvas), 1)
    h.set_material(red)
    assert _is(_pixel(engine, canvas), 0)


def test_two_batches_sharing_a_material_both_follow_its_colour():
    engine, canvas = _engine()
    mat = StandardMaterial("#ff0000", roughness=0.8)
    engine.spawn(Mesh(cube(1, 1, 1)), Material(mat), Transform(pos=(-1.5, 0, 0)))
    engine.spawn(Mesh(cube(1.0, 1.0, 1.0001)), Material(mat), Transform(pos=(1.5, 0, 0)))
    engine._draw_frame()
    canvas.draw()
    mat.color = "#00ff00"
    engine._draw_frame()
    img = np.asarray(canvas.draw()).astype(int)
    assert _is(img[32, 16, :3], 1) and _is(img[32, 48, :3], 1)


def test_textured_batch_binds_its_texture_and_follows_a_swap():
    from manifoldx.textures import texture_from_array

    engine, canvas = _engine()
    geo = cube(2, 2, 2)
    geo["uvs"] = np.zeros((len(geo["positions"]), 2), np.float32)
    mat = StandardMaterial("#ffffff", roughness=0.8)
    engine.spawn(Mesh(geo), Material(mat), Transform())
    _pixel(engine, canvas)  # the device texture_from_array needs
    red = texture_from_array(engine, np.tile(np.array([255, 0, 0, 255], np.uint8), (4, 4, 1)))
    blue = texture_from_array(engine, np.tile(np.array([0, 0, 255, 255], np.uint8), (4, 4, 1)))
    textured = StandardMaterial("#ffffff", roughness=0.8, albedo_map=red)
    engine.spawn(Mesh(geo), Material(textured), Transform(pos=(0, 0, 1.5)))
    assert _is(_pixel(engine, canvas), 0)
    assert _is(_pixel(engine, canvas), 0)
    textured.albedo_map = blue
    assert _is(_pixel(engine, canvas), 2)


def test_skinned_batch_binds_its_joints_and_follows_set_joints():
    engine, canvas = _engine()
    geo = cube(1, 1, 1)
    geo["joints"] = np.zeros(len(geo["positions"]), np.uint32)
    h = engine.spawn(Mesh(geo), Material(BasicMaterial("#ff0000")), Transform())
    assert _is(_pixel(engine, canvas), 0)
    away = np.eye(4, dtype=np.float32)
    away[0, 3] = 3.0
    h.set_joints(away[None])
    assert _empty(engine, canvas)
    # More skinned entities grow the joints and blocks buffers; the first of
    # them is posed back to the centre and must still be drawn there.
    hs = engine.spawn(Mesh(geo), Material(BasicMaterial("#00ff00")), Transform(pos=(2, 2, 0)), n=40)
    back = np.eye(4, dtype=np.float32)
    back[0, 3], back[1, 3] = -2.0, -2.0
    hs.set_joints(back[None])
    assert _is(_pixel(engine, canvas), 1)


def test_a_grown_transforms_buffer_is_rebound():
    engine, canvas = _engine()
    mat = StandardMaterial("#ff0000", roughness=0.8)
    engine.spawn(Mesh(cube(1, 1, 1)), Material(mat), Transform(pos=(-2, 0, 0)))
    assert _empty(engine, canvas)  # nothing at the centre yet
    engine.spawn(Mesh(cube(1, 1, 1)), Material(mat), Transform(pos=(9, 9, 0)), n=50)
    engine.spawn(Mesh(cube(1, 1, 1)), Material(mat), Transform())
    assert _is(_pixel(engine, canvas), 0)
