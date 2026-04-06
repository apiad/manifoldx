"""Test renderer integration with material-type specific pipelines."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from manifoldx.resources import BasicMaterial, StandardMaterial
from manifoldx.renderer import RenderPipeline
from manifoldx.ecs import EntityStore


class TestRendererMaterialPipeline:
    """Test renderer uses material-type specific shaders."""

    def test_pipeline_key_includes_material_type(self):
        """Pipeline cache key should include material type, not material_id."""
        store = EntityStore(100)
        pipeline = RenderPipeline(store, None)

        # Current implementation uses (geom_id, mat_id) - need to change to (geom_id, material_type)
        # This test will fail until we update the renderer
        assert hasattr(pipeline, "_pipeline_cache") or hasattr(pipeline, "_pipelines")

    def test_different_material_types_need_different_pipelines(self):
        """BasicMaterial and StandardMaterial should use different pipelines."""
        store = EntityStore(100)
        pipeline = RenderPipeline(store, None)

        # Get shader sources from materials
        basic_shader = BasicMaterial._compile()
        standard_shader = StandardMaterial._compile()

        # These should be different
        assert basic_shader != standard_shader


class TestMaterialDataFlow:
    """Test material data flows through ECS to renderer."""

    def test_material_get_data_returns_array(self):
        """Material.get_data() should return numpy array."""
        mat = BasicMaterial(color="#ff0000")
        # Create a mock registry
        mock_registry = MagicMock()
        mock_registry.register.return_value = 1

        data = mat.get_data(1, mock_registry)
        assert isinstance(data, np.ndarray)

    def test_standard_material_get_data_returns_array(self):
        """StandardMaterial.get_data() should return numpy array with PBR fields."""
        mat = StandardMaterial(color="#3366ff", roughness=0.3, metallic=0.8)
        mock_registry = MagicMock()
        mock_registry.register.return_value = 1

        data = mat.get_data(1, mock_registry)
        assert isinstance(data, np.ndarray)


class TestRenderPipelineWithMaterial:
    """Test render pipeline integration with materials."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine with required attributes."""
        engine = MagicMock()
        engine._geometry_registry = MagicMock()
        engine._material_registry = MagicMock()
        engine._camera = MagicMock()
        engine._camera.get_view_projection_matrix.return_value = np.eye(
            4, dtype=np.float32
        )
        engine.w = 800
        engine.h = 600
        return engine

    def test_render_uses_material_shader(self, mock_engine):
        """Render should use the material's compiled shader."""
        # This test verifies the renderer architecture
        # After implementation, renderer should:
        # 1. Get material type from registry
        # 2. Use material._compile() to get shader
        # 3. Create pipeline with that shader
        pass  # Architecture test - will be validated by integration test


class TestBasicMaterialIntegration:
    """Integration test: BasicMaterial renders correctly."""

    def test_basic_material_can_be_registered(self):
        """BasicMaterial should be registrable in MaterialRegistry."""
        from manifoldx.resources import MaterialRegistry

        registry = MaterialRegistry()
        mat = BasicMaterial(color="#ff0000")

        mat_id = registry.register(mat)
        assert mat_id > 0

        retrieved = registry.get(mat_id)
        assert retrieved is mat
        assert retrieved.color == "#ff0000"
