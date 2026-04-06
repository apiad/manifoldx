"""Camera class for 3D rendering."""
import numpy as np


class Camera:
    """Camera with position, target, FOV and projection."""
    
    def __init__(self, position=(0, 10, 20), target=(0, 0, 0), fov=60, up=(0, 1, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.fov = fov
        self.up = np.array(up, dtype=np.float32)
        self._compute_spherical()

    def _compute_spherical(self):
        direction = self.position - self.target
        self._distance = np.linalg.norm(direction)
        direction_normalized = direction / self._distance
        self._azimuth = float(np.degrees(np.arctan2(direction_normalized[0], direction_normalized[2])))
        self._elevation = float(np.degrees(np.arcsin(direction_normalized[1])))
    
    def get_view_matrix(self):
        """Compute view matrix using look-at formula.
        
        Convention: camera looks down -Z in view space (standard OpenGL/WebGPU).
        """
        # Forward direction (from camera toward target)
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        # Right (x-axis): cross(forward, up)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        # True up: cross(right, forward)
        up = np.cross(right, forward)
        
        # Build view matrix: rotation part uses -forward for Z (camera looks down -Z)
        view = np.eye(4, dtype=np.float32)
        view[0, 0:3] = right
        view[1, 0:3] = up
        view[2, 0:3] = -forward  # Negate: camera looks down -Z
        view[0, 3] = -np.dot(right, self.position)
        view[1, 3] = -np.dot(up, self.position)
        view[2, 3] = np.dot(forward, self.position)  # Note: no negate since forward already negated
        
        return view
    
    def get_forward(self):
        direction = self.target - self.position
        return direction / np.linalg.norm(direction)
    
    def get_right(self):
        forward = self.get_forward()
        right = np.cross(forward, self.up)
        return right / np.linalg.norm(right)
    
    def get_up(self):
        right = self.get_right()
        forward = self.get_forward()
        up = np.cross(right, forward)
        return up / np.linalg.norm(up)
    
    def get_view_projection_matrix(self, aspect):
        view = self.get_view_matrix()
        proj = self.get_projection_matrix(aspect)
        return (proj @ view).T
    
    def get_azimuth_elevation(self):
        return (self._azimuth, self._elevation)
    
    def get_projection_matrix(self, aspect, near=0.1, far=100.0):
        """Compute perspective projection matrix for WebGPU.
        
        WebGPU NDC: x,y in [-1,1], z in [0,1].
        Uses the same formula as pylinalg mat_perspective with depth_range=(0,1).
        """
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        # WebGPU depth range [0, 1]:
        # c = -(far*1 - near*0) / (far - near) = -far / (far - near)
        # d = -(far * near * (1 - 0)) / (far - near) = -far*near / (far - near)
        proj[2, 2] = -far / (far - near)
        proj[2, 3] = -(far * near) / (far - near)
        proj[3, 2] = -1.0
        
        return proj


__all__ = ['Camera']
