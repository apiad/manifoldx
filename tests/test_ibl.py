import numpy as np
import pytest
from pathlib import Path


def test_from_color_shape():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_color((0.2, 0.3, 0.4))
    assert env.data.shape == (32, 64, 3)
    assert env.data.dtype == np.float32
    np.testing.assert_allclose(env.data[0, 0], [0.2, 0.3, 0.4], atol=1e-6)


def test_from_color_defaults():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_color((1.0, 1.0, 1.0))
    assert env.intensity == 1.0
    assert env.show_skybox is False


def test_from_sky_shape():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_sky(
        zenith=(0.1, 0.2, 0.8),
        horizon=(0.5, 0.6, 0.9),
        ground=(0.05, 0.05, 0.05),
    )
    assert env.data.shape == (64, 128, 3)
    assert env.data.dtype == np.float32


def test_from_sky_top_is_zenith():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_sky(
        zenith=(1.0, 0.0, 0.0),
        horizon=(0.0, 1.0, 0.0),
        ground=(0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(env.data[0, 0], [1.0, 0.0, 0.0], atol=0.1)
    np.testing.assert_allclose(env.data[-1, 0], [0.0, 0.0, 1.0], atol=0.1)


def test_from_image_shape(tmp_path):
    from PIL import Image
    from manifoldx.ibl import EnvironmentMap
    img = Image.fromarray(
        (np.full((32, 64, 3), 128, dtype=np.uint8)),
        mode="RGB",
    )
    p = tmp_path / "test.png"
    img.save(p)
    env = EnvironmentMap.from_image(str(p))
    assert env.data.shape == (32, 64, 3)
    assert env.data.dtype == np.float32
    assert np.all(env.data > 0.2) and np.all(env.data < 0.25)


def test_from_image_exposure(tmp_path):
    from PIL import Image
    from manifoldx.ibl import EnvironmentMap
    img = Image.fromarray(np.full((8, 16, 3), 100, dtype=np.uint8), mode="RGB")
    p = tmp_path / "exp.png"
    img.save(p)
    env1 = EnvironmentMap.from_image(str(p), exposure=1.0)
    env2 = EnvironmentMap.from_image(str(p), exposure=2.0)
    np.testing.assert_allclose(env2.data, env1.data * 2.0, atol=1e-5)
