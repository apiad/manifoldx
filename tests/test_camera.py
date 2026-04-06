"""Tests for Camera API - helper matrices (Phase 1)."""
import numpy as np
import pytest


class TestCameraHelperMatrices:
    """Test camera helper matrices and convenience methods."""
    
    def test_get_forward(self):
        """Camera.get_forward() returns normalized forward direction."""
        from manifoldx.camera import Camera
        
        # Default camera looks toward origin from (0, 10, 20)
        # Forward should point from position toward target
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        forward = camera.get_forward()
        
        assert isinstance(forward, np.ndarray), "get_forward must return numpy array"
        assert forward.shape == (3,), "get_forward must return 3D vector"
        assert np.isclose(np.linalg.norm(forward), 1.0, atol=1e-6), \
            "get_forward must return normalized vector"
        # Default camera should look roughly in -Z direction (toward origin)
        assert forward[2] < 0, "Default camera should look in -Z direction"
    
    def test_get_forward_changes_with_target(self):
        """get_forward() updates when target changes."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 0, 0), target=(10, 0, 0))
        
        forward = camera.get_forward()
        
        assert np.isclose(forward[0], 1.0, atol=1e-6), \
            "Forward should point toward +X target"
    
    def test_get_right(self):
        """Camera.get_right() returns normalized right vector."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        right = camera.get_right()
        
        assert isinstance(right, np.ndarray), "get_right must return numpy array"
        assert right.shape == (3,), "get_right must return 3D vector"
        assert np.isclose(np.linalg.norm(right), 1.0, atol=1e-6), \
            "get_right must return normalized vector"
        # Right should be perpendicular to forward
        forward = camera.get_forward()
        dot = np.dot(forward, right)
        assert np.isclose(dot, 0.0, atol=1e-6), \
            "Right must be perpendicular to forward"
    
    def test_get_up(self):
        """Camera.get_up() returns normalized up vector."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        up = camera.get_up()
        
        assert isinstance(up, np.ndarray), "get_up must return numpy array"
        assert up.shape == (3,), "get_up must return 3D vector"
        assert np.isclose(np.linalg.norm(up), 1.0, atol=1e-6), \
            "get_up must return normalized vector"
        # Up should be perpendicular to both forward and right
        forward = camera.get_forward()
        right = camera.get_right()
        assert np.isclose(np.dot(forward, up), 0.0, atol=1e-6), \
            "Up must be perpendicular to forward"
        assert np.isclose(np.dot(right, up), 0.0, atol=1e-6), \
            "Up must be perpendicular to right"
    
    def test_get_view_projection_matrix(self):
        """Camera.get_view_projection_matrix() returns combined VP matrix."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        vp = camera.get_view_projection_matrix(aspect=1.0)
        
        assert isinstance(vp, np.ndarray), "get_view_projection_matrix must return numpy array"
        assert vp.shape == (4, 4), "VP matrix must be 4x4"
        # Matrix should be transposed for WGSL column-major layout
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect=1.0)
        expected_vp = (proj @ view).T
        assert np.allclose(vp, expected_vp, atol=1e-6), \
            "VP matrix should equal proj @ view.T"


class TestCameraSphericalCoords:
    """Test spherical coordinate system for orbit controls."""
    
    def test_camera_tracks_spherical_coords(self):
        """Camera should track azimuth, elevation, distance internally."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        # Should have internal spherical coordinate tracking
        assert hasattr(camera, '_azimuth'), "Camera must track _azimuth"
        assert hasattr(camera, '_elevation'), "Camera must track _elevation"
        assert hasattr(camera, '_distance'), "Camera must track _distance"
    
    def test_spherical_coords_from_default_position(self):
        """Spherical coords computed correctly from default position."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        # Distance should be ~22.36 (sqrt(10^2 + 20^2))
        expected_distance = np.sqrt(10**2 + 20**2)
        assert np.isclose(camera._distance, expected_distance, atol=1e-3)
        
        # Azimuth and elevation should be reasonable values
        assert isinstance(camera._azimuth, (int, float))
        assert isinstance(camera._elevation, (int, float))
    
    def test_get_azimuth_elevation(self):
        """get_azimuth_elevation() returns current angles."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        azimuth, elevation = camera.get_azimuth_elevation()
        
        assert isinstance(azimuth, (int, float)), "azimuth must be numeric"
        assert isinstance(elevation, (int, float)), "elevation must be numeric"
