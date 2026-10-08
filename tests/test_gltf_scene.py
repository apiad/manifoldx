import math

import numpy as np
import pytest

from manifoldx.camera import Camera
from manifoldx.gltf import Node, export_gltf
from manifoldx.ibl import EnvironmentMap
from manifoldx.resources import DirectionalLight, PointLight, SpotLight, StandardMaterial, cube

from gltf_helpers import load, view_bytes


def rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def test_camera_looks_at_its_target(tmp_path):
    cam = Camera(position=(0, 5, 10), target=(0, 0, 0), fov=50, near=0.2, far=900)
    export_gltf(tmp_path / "a.glb", [], camera=cam)
    g = load(tmp_path / "a.glb")
    node = next(n for n in g.nodes if n.camera is not None)
    p = g.cameras[node.camera].perspective
    assert p.yfov == pytest.approx(math.radians(50)) and (p.znear, p.zfar) == (pytest.approx(0.2), 900)
    assert node.translation == [0, 5, 10]
    forward = rotate(node.rotation, (0, 0, -1))
    np.testing.assert_allclose(forward, np.array([0, -5, -10]) / np.linalg.norm([0, 5, 10]), atol=1e-6)
    assert rotate(node.rotation, (0, 1, 0))[1] > 0  # not upside down


def test_lights(tmp_path):
    sun = DirectionalLight(color="#ff8000", intensity=2.4, direction=(0, -1, 0))
    spot = SpotLight(color="#ffffff", intensity=3, position=(1, 2, 3), direction=(1, 0, 0),
                     inner_angle=0.2, outer_angle=0.4, distance=20)
    point = PointLight(color="#ffffff", intensity=5, position=(4, 5, 6))
    export_gltf(tmp_path / "a.glb", [], lights=[sun, spot, point])
    g = load(tmp_path / "a.glb")
    assert "KHR_lights_punctual" in g.extensionsUsed
    lights = g.extensions["KHR_lights_punctual"]["lights"]
    assert [light["type"] for light in lights] == ["directional", "spot", "point"]
    assert lights[0]["color"] == pytest.approx([1.0, 128 / 255, 0.0])
    assert lights[0]["intensity"] == 2.4 and lights[0]["extras"]["manifoldx_intensity"] == 2.4
    assert lights[1]["spot"] == {"innerConeAngle": 0.2, "outerConeAngle": 0.4} and lights[1]["range"] == 20
    assert "range" not in lights[2]
    by_light = {n.extensions["KHR_lights_punctual"]["light"]: n for n in g.nodes if n.extensions}
    np.testing.assert_allclose(rotate(by_light[0].rotation, (0, 0, -1)), (0, -1, 0), atol=1e-6)
    np.testing.assert_allclose(rotate(by_light[1].rotation, (0, 0, -1)), (1, 0, 0), atol=1e-6)
    assert by_light[2].translation == [4, 5, 6]


def test_environment_and_extras(tmp_path):
    env = EnvironmentMap.from_sky(zenith=(0.4, 0.5, 0.7), horizon=(0.6, 0.6, 0.7), ground=(0.3, 0.3, 0.2))
    env.intensity = 0.35
    export_gltf(tmp_path / "a.glb", [Node(cube(1, 1, 1), StandardMaterial("#ffffff"))], environment=env,
                scene_extras={"fog": {"start": 10.0, "end": 20.0, "color": [0.1, 0.2, 0.3]}})
    g = load(tmp_path / "a.glb")
    extras = g.scenes[0].extras["manifoldx"]
    assert extras["fog"]["end"] == 20.0
    e = extras["environment"]
    assert (e["width"], e["height"], e["intensity"]) == (128, 64, 0.35)
    data = np.frombuffer(view_bytes(g, e["bufferView"]), np.float32).reshape(64, 128, 3)
    np.testing.assert_allclose(data, env.data)
