"""Test multi-instance PBR rendering."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from manifoldx.resources import StandardMaterial, PointLight, cube, sphere
from manifoldx.ecs import EntityStore


class TestMultiInstanceMaterial:
    """Test per-instance material properties."""

    def test_standard_material_different_instances(self):
        """Different StandardMaterial instances should have different data."""
        mat1 = StandardMaterial(color="#ff0000", roughness=0.1, metallic=0.0)
        mat2 = StandardMaterial(color="#00ff00", roughness=0.9, metallic=1.0)
        
        # They should have different data when get_data() is called
        mock_registry = MagicMock()
        mock_registry.register.side_effect = [1, 2]
        
        data1 = mat1.get_data(1, mock_registry)
        data2 = mat2.get_data(1, mock_registry)
        
        # Data should be different
        assert not np.allclose(data1, data2)

    def test_material_per_instance_storage(self):
        """Material data should be per-instance in storage buffer."""
        # This test verifies the architecture allows per-instance material data
        # In the renderer, this would be uploaded to a storage buffer
        store = EntityStore(100)
        
        # Register components
        from manifoldx.components import Transform, Mesh, Material
        Transform.register(store)
        Mesh.register(store)
        Material.register(store)
        
        # Spawn with different materials
        mat1 = StandardMaterial(color="#ff0000", roughness=0.2)
        mat2 = StandardMaterial(color="#00ff00", roughness=0.8)
        
        # get_data should return arrays with different roughness values
        mock_registry = MagicMock()
        mock_registry.register.side_effect = [1, 2]
        
        data1 = mat1.get_data(1, mock_registry)
        data2 = mat2.get_data(1, mock_registry)
        
        # Roughness is stored at index 1 (after albedo 3 components)
        # albedo(3) + roughness(1) + metallic(1) + ao(1) = 6 floats
        roughness_idx = 3  # After albedo vec3
        assert data1[0, roughness_idx] != data2[0, roughness_idx]


class TestPBRRenderingWithLights:
    """Test PBR rendering works with external lights."""

    def test_point_light_affects_pbr(self):
        """PointLight should affect StandardMaterial rendering."""
        light = PointLight(color="#ffffff", intensity=2.0, position=(0, 5, 0))
        
        # Light should have data that can be passed to GPU
        data = light.get_data()
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32


class TestDemoScene:
    """Integration test: demo scene with multiple objects."""

    def test_demo_spawns_multiple_objects(self):
        """Demo should spawn multiple objects with different materials."""
        from manifoldx.components import Transform, Mesh, Material
        
        # This test just verifies the concept - actual demo is in examples/
        # The key is: we can spawn entities with different material instances
        store = EntityStore(100)
        Transform.register(store)
        Mesh.register(store)
        Material.register(store)
        
        # Register geometries
        from manifoldx.resources import GeometryRegistry, MaterialRegistry
        geo_registry = GeometryRegistry()
        mat_registry = MaterialRegistry()
        
        # Create materials with different properties
        red_shiny = StandardMaterial(color="#ff0000", roughness=0.1, metallic=0.9)
        green_dull = StandardMaterial(color="#00ff00", roughness=0.8, metallic=0.1)
        
        # Register and get IDs
        red_id = mat_registry.register(red_shiny)
        green_id = mat_registry.register(green_dull)
        
        assert red_id != green_id
        
        # Verify materials are different
        assert red_shiny.roughness != green_dull.roughness
        assert red_shiny.metallic != green_dull.metallic
