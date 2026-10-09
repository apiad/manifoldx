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


@pytest.mark.slow
def test_precompute_irradiance_shape():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_color((1.0, 1.0, 1.0))
    env._precompute()
    assert env._irradiance is not None
    assert env._irradiance.shape == (6, 64, 64, 4)
    assert env._irradiance.dtype == np.float16


@pytest.mark.slow
def test_precompute_prefiltered_shape():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_color((0.5, 0.5, 0.5))
    env._precompute()
    assert env._prefiltered is not None
    assert len(env._prefiltered) == 8
    assert env._prefiltered[0].shape == (6, 128, 128, 4)
    assert env._prefiltered[7].shape == (6, 1, 1, 4)


@pytest.mark.slow
def test_precompute_irradiance_non_negative():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_color((0.3, 0.5, 0.7))
    env._precompute()
    assert np.all(env._irradiance.astype(np.float32) >= 0.0)


@pytest.mark.slow
def test_precompute_cached():
    from manifoldx.ibl import EnvironmentMap
    env = EnvironmentMap.from_color((1.0, 1.0, 1.0))
    env._precompute()
    id1 = id(env._irradiance)
    env._precompute()
    assert id(env._irradiance) == id1


def test_brdf_lut_loads():
    from manifoldx.ibl import load_brdf_lut
    lut = load_brdf_lut()
    assert lut.shape == (512, 512, 2)
    assert lut.dtype == np.float32
    assert np.all(lut >= 0.0) and np.all(lut <= 1.0)


def test_engine_set_environment_preset():
    from manifoldx.engine import Engine
    from manifoldx.ibl import EnvironmentMap
    eng = Engine("test", max_entities=16)
    eng.set_environment("studio")
    assert isinstance(eng.environment, EnvironmentMap)


def test_engine_set_environment_object():
    from manifoldx.engine import Engine
    from manifoldx.ibl import EnvironmentMap
    eng = Engine("test", max_entities=16)
    env = EnvironmentMap.from_color((0.5, 0.5, 0.5))
    eng.set_environment(env)
    assert eng.environment is env


def test_engine_set_environment_none():
    from manifoldx.engine import Engine
    eng = Engine("test", max_entities=16)
    eng.set_environment("neutral")
    eng.set_environment(None)
    assert eng.environment is None


def test_engine_environment_intensity():
    from manifoldx.engine import Engine
    from manifoldx.ibl import EnvironmentMap
    eng = Engine("test", max_entities=16)
    env = EnvironmentMap.from_color((0.3, 0.3, 0.3))
    env.intensity = 2.5
    eng.set_environment(env)
    assert eng.environment.intensity == 2.5


def _sky():
    from manifoldx.ibl import EnvironmentMap

    return EnvironmentMap.from_sky(zenith=(0.4, 0.5, 0.75), horizon=(0.6, 0.7, 0.8), ground=(0.3, 0.3, 0.2))


def test_prefiltered_mip0_is_the_cube():
    """Roughness 0 is a mirror: its prefiltered map is the radiance cube itself."""
    from manifoldx.ibl import _equirect_to_cubemap

    env = _sky()
    env._precompute()
    cube = _equirect_to_cubemap(env.data, face_size=128)
    np.testing.assert_array_equal(env._prefiltered[0][..., :3], cube.astype(np.float16))
    assert (env._prefiltered[0][..., 3] == 1).all()


def test_sampling_a_mirror_gives_the_cube_back():
    """What mip 0 used to compute (GGX at roughness 0, sampled) is the cube, up to the
    nearest-texel lookup landing one texel over: never off by more than neighbouring texels
    differ. Checked at a size cheap to test."""
    from manifoldx.ibl import _compute_prefiltered, _equirect_to_cubemap

    cube = _equirect_to_cubemap(_sky().data, face_size=32)
    sampled = _compute_prefiltered(cube, roughness=0.0, out_size=32, samples=64)[..., :3].astype(np.float32)
    one_texel = max(np.abs(np.diff(cube, axis=1)).max(), np.abs(np.diff(cube, axis=2)).max())
    assert np.abs(sampled - cube).max() <= one_texel + 1e-3


def test_precompute_is_cached_on_disk(tmp_path, monkeypatch):
    import manifoldx.ibl as ibl

    monkeypatch.setenv("MANIFOLDX_CACHE_DIR", str(tmp_path))
    first = _sky()
    first._precompute()
    assert len(list(tmp_path.rglob("*.npz"))) == 1

    def boom(*a, **k):
        raise AssertionError("recomputed despite the cache")

    monkeypatch.setattr(ibl, "_compute_prefiltered", boom)
    monkeypatch.setattr(ibl, "_compute_irradiance", boom)
    again = _sky()
    again._precompute()
    np.testing.assert_array_equal(again._irradiance, first._irradiance)
    for a, b in zip(again._prefiltered, first._prefiltered):
        np.testing.assert_array_equal(a, b)


def test_a_different_sky_misses_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFOLDX_CACHE_DIR", str(tmp_path))
    from manifoldx.ibl import EnvironmentMap

    _sky()._precompute()
    EnvironmentMap.from_color((0.3, 0.3, 0.3))._precompute()
    assert len(list(tmp_path.rglob("*.npz"))) == 2
