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


class TestCameraDirectPositioning:
    """Test direct positioning methods (Phase 2)."""
    
    def test_move_to(self):
        """move_to() sets camera position."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        camera.move_to((5, 15, 25))
        
        assert np.allclose(camera.position, [5, 15, 25], atol=1e-6), \
            "move_to should set position"
        # Target unchanged
        assert np.allclose(camera.target, [0, 0, 0], atol=1e-6), \
            "move_to should not change target"
    
    def test_move_to_updates_spherical(self):
        """move_to() updates spherical coordinates."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        # New position at different distance
        camera.move_to((0, 5, 10))
        
        # Distance should be ~11.18 (sqrt(5^2 + 10^2))
        expected_distance = np.sqrt(5**2 + 10**2)
        assert np.isclose(camera._distance, expected_distance, atol=1e-3), \
            "move_to should update distance"
    
    def test_move_by(self):
        """move_by() adds delta to position."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        camera.move_by((1, 2, 3))
        
        assert np.allclose(camera.position, [1, 12, 23], atol=1e-6), \
            "move_by should add delta to position"
    
    def test_look_at(self):
        """look_at() sets the target point."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        camera.look_at((10, 5, 15))
        
        assert np.allclose(camera.target, [10, 5, 15], atol=1e-6), \
            "look_at should set target"
        # Position unchanged
        assert np.allclose(camera.position, [0, 10, 20], atol=1e-6), \
            "look_at should not change position"
    
    def test_look_at_updates_spherical(self):
        """look_at() updates spherical coordinates."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 0, 10), target=(0, 0, 0))
        
        # Move target farther
        camera.look_at((0, 0, 20))
        
        # Distance should now be 10
        assert np.isclose(camera._distance, 10.0, atol=1e-3), \
            "look_at should update distance"
    
    def test_set_pose(self):
        """set_pose() sets both position and target atomically."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        camera.set_pose(position=(5, 15, 25), target=(1, 2, 3))
        
        assert np.allclose(camera.position, [5, 15, 25], atol=1e-6), \
            "set_pose should set position"
        assert np.allclose(camera.target, [1, 2, 3], atol=1e-6), \
            "set_pose should set target"
    
    def test_set_pose_updates_spherical(self):
        """set_pose() updates spherical coordinates."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        camera.set_pose(position=(10, 20, 10), target=(0, 0, 0))
        
        # Distance should be sqrt(10^2 + 20^2 + 10^2) = ~24.49
        expected_distance = np.sqrt(10**2 + 20**2 + 10**2)
        assert np.isclose(camera._distance, expected_distance, atol=1e-3), \
            "set_pose should update spherical coords"
        # Azimuth should be 45 degrees (equal X and Z components)
        assert np.isclose(camera._azimuth, 45.0, atol=1.0), \
            "set_pose should update azimuth"


class TestCameraOrbitControls:
    """Test orbit controls (Phase 3)."""
    
    def test_orbit_horizontal(self):
        """orbit() rotates camera horizontally around target."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        initial_azimuth = camera._azimuth
        camera.orbit(d_azimuth=45)
        
        assert np.isclose(camera._azimuth, initial_azimuth + 45, atol=1e-3), \
            "orbit should change azimuth by 45 degrees"
        # Position should have changed
        assert not np.allclose(camera.position, [0, 10, 20], atol=1e-3), \
            "orbit should change camera position"
    
    def test_orbit_vertical(self):
        """orbit() rotates camera vertically around target."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        initial_elevation = camera._elevation
        camera.orbit(d_elevation=10)
        
        assert np.isclose(camera._elevation, initial_elevation + 10, atol=1e-3), \
            "orbit should change elevation by 10 degrees"
    
    def test_orbit_preserves_distance(self):
        """orbit() should preserve distance to target."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        initial_distance = camera._distance
        camera.orbit(d_azimuth=90, d_elevation=45)
        
        assert np.isclose(camera._distance, initial_distance, atol=1e-3), \
            "orbit should preserve distance"
    
    def test_orbit_elevation_clamped(self):
        """orbit() clamps elevation to prevent gimbal lock."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 1, 10), target=(0, 0, 0))
        
        # Try to orbit past the pole
        camera.orbit(d_elevation=95)
        
        # Elevation should be clamped to ~89
        assert camera._elevation < 90, "Elevation should be clamped below 90"
    
    def test_orbit_combined(self):
        """orbit() handles combined horizontal and vertical rotation."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(10, 0, 0), target=(0, 0, 0))
        
        camera.orbit(d_azimuth=90, d_elevation=0)
        
        # After 90 degree azimuth orbit, camera should be at +Z
        # Allow some tolerance
        assert camera.position[2] > 0, "Camera should be at +Z after 90 azimuth"
    
    def test_orbit_both_axes(self):
        """orbit() rotates around both axes simultaneously."""
        from manifoldx.camera import Camera
        
        camera = Camera(position=(0, 10, 20), target=(0, 0, 0))
        
        initial_pos = camera.position.copy()
        initial_target = camera.target.copy()
        
        camera.orbit(d_azimuth=30, d_elevation=15)
        
        # Position should change
        assert not np.allclose(camera.position, initial_pos, atol=1e-3), \
            "orbit should change position"
        # Target should remain fixed
        assert np.allclose(camera.target, initial_target, atol=1e-6), \
            "orbit should not change target"
