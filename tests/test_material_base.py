"""Test Material base class with shader compilation interface."""

import pytest
from manifoldx.resources import Material, BasicMaterial, StandardMaterial


class TestMaterialBaseClass:
    """Test Material abstract base class interface."""

    def test_material_is_abstract(self):
        """Material should be abstract (cannot be instantiated directly)."""
        with pytest.raises(TypeError):
            Material()

    def test_binding_slot_class_attribute(self):
        """Material subclasses should have binding_slot class attribute."""
        assert hasattr(BasicMaterial, "binding_slot")
        assert hasattr(StandardMaterial, "binding_slot")
        assert isinstance(BasicMaterial.binding_slot, int)
        assert isinstance(StandardMaterial.binding_slot, int)

    def test_different_binding_slots(self):
        """Different material types should have different binding slots."""
        assert BasicMaterial.binding_slot != StandardMaterial.binding_slot


class TestBasicMaterialCompile:
    """Test BasicMaterial._compile() returns valid WGSL."""

    def test_basic_compile_returns_string(self):
        """BasicMaterial._compile() should return a string."""
        result = BasicMaterial._compile()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compile_returns_valid_wgsl(self):
        """BasicMaterial._compile() should return valid WGSL (has basic structure)."""
        wgsl = BasicMaterial._compile()
        # Should have shader structure
        assert "fn " in wgsl  # Has function definitions
        assert "@vertex" in wgsl  # Has vertex shader
        assert "@fragment" in wgsl  # Has fragment shader


class TestBasicMaterialUniformType:
    """Test BasicMaterial.uniform_type() returns buffer field definitions."""

    def test_uniform_type_returns_dict(self):
        """uniform_type() should return a dictionary."""
        result = BasicMaterial.uniform_type()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_uniform_type_has_color(self):
        """BasicMaterial should have color field."""
        uniform_type = BasicMaterial.uniform_type()
        assert "color" in uniform_type


class TestStandardMaterialCompile:
    """Test StandardMaterial._compile() returns valid WGSL."""

    def test_standard_compile_returns_string(self):
        """StandardMaterial._compile() should return a string."""
        result = StandardMaterial._compile()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compile_returns_pbr_shader(self):
        """StandardMaterial._compile() should return PBR shader."""
        wgsl = StandardMaterial._compile()
        # PBR should have BRDF functions
        assert "DistributionGGX" in wgsl or "GGX" in wgsl
        assert "fresnel" in wgsl.lower() or "F_Schlick" in wgsl


class TestStandardMaterialUniformType:
    """Test StandardMaterial.uniform_type() returns buffer field definitions."""

    def test_uniform_type_returns_dict(self):
        """uniform_type() should return a dictionary."""
        result = StandardMaterial.uniform_type()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_uniform_type_has_pbr_fields(self):
        """StandardMaterial should have albedo, roughness, metallic, ao."""
        uniform_type = StandardMaterial.uniform_type()
        assert "albedo" in uniform_type
        assert "roughness" in uniform_type
        assert "metallic" in uniform_type
        assert "ao" in uniform_type


class TestMaterialInstantiation:
    """Test that materials can be instantiated with parameters."""

    def test_basic_material_accepts_color(self):
        """BasicMaterial should accept color parameter."""
        mat = BasicMaterial(color="#ff0000")
        assert mat.color == "#ff0000"

    def test_standard_material_accepts_pbr_params(self):
        """StandardMaterial should accept roughness, metallic, ao."""
        mat = StandardMaterial(color="#3366ff", roughness=0.3, metallic=0.8, ao=0.9)
        assert mat.color == "#3366ff"
        assert mat.roughness == 0.3
        assert mat.metallic == 0.8
        assert mat.ao == 0.9
