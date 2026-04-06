"""Test external light system - lights passed to renderer."""

import pytest
import numpy as np
from manifoldx.resources import PointLight, SpotLight, DirectionalLight


class TestLightClasses:
    """Test light classes exist and have correct properties."""

    def test_point_light_creation(self):
        """PointLight should be creatable with color and intensity."""
        light = PointLight(color="#ffaa00", intensity=2.0, position=(1, 2, 3))
        assert light.color == "#ffaa00"
        assert light.intensity == 2.0
        assert np.allclose(light.position, (1, 2, 3))

    def test_spot_light_creation(self):
        """SpotLight should be creatable with position, direction, cone angles."""
        light = SpotLight(
            color="#ffffff",
            intensity=3.0,
            position=(0, 5, 0),
            direction=(0, -1, 0),
            inner_angle=0.5,
            outer_angle=0.8,
        )
        assert light.color == "#ffffff"
        assert light.intensity == 3.0

    def test_directional_light_creation(self):
        """DirectionalLight should be creatable with direction."""
        light = DirectionalLight(color="#ffffaa", intensity=1.5, direction=(0, -1, 0))
        assert light.color == "#ffffaa"
        assert light.intensity == 1.5


class TestLightDataMethods:
    """Test light classes provide data for GPU."""

    def test_point_light_get_data(self):
        """PointLight.get_data() should return array for GPU upload."""
        light = PointLight(color="#ff0000", intensity=1.0, position=(1, 0, 0))
        data = light.get_data()
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32

    def test_light_uniform_type(self):
        """Light classes should define uniform_type() for buffer layout."""
        uniform_type = PointLight.uniform_type()
        assert isinstance(uniform_type, dict)
        assert "color" in uniform_type or "position" in uniform_type


class TestEngineLightAPI:
    """Test engine.set_lights() API."""

    def test_engine_has_set_lights_method(self):
        """Engine should have set_lights() method."""
        from manifoldx import Engine

        # Can't create full engine without GPU, but check method exists
        assert hasattr(Engine, "set_lights") or hasattr(Engine, "add_light")


class TestLightAnimation:
    """Test light animation through update callback."""

    def test_light_position_can_be_animated(self):
        """Light position should be update-able for animation."""
        light = PointLight(color="#ff0000", intensity=1.0, position=(0, 0, 0))

        # Update position (simulate animation)
        light.position = (1, 2, 3)
        assert np.allclose(light.position, (1, 2, 3))
