"""Camera class for 3D rendering."""
import numpy as np


class Camera:
    """Camera with position, target, FOV and projection."""
    
    def __init__(self, position=(0, 0, 5), target=(0, 0, 0), fov=60, up=(0, 1, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.fov = fov
        self.up = np.array(up, dtype=np.float32)
    
    def get_view_matrix(self):
        """Compute view matrix using look-at formula."""
        # Forward (z-axis): normalize(target - position)
        z = self.target - self.position
        z = z / np.linalg.norm(z)
        
        # Right (x-axis): normalize(cross(up, z))
        x = np.cross(self.up, z)
        x = x / np.linalg.norm(x)
        
        # Up (y-axis): cross(z, x)
        y = np.cross(z, x)
        
        # Build view matrix (inverse of camera transform)
        view = np.eye(4, dtype=np.float32)
        view[0, 0:3] = x
        view[1, 0:3] = y
        view[2, 0:3] = z
        view[0, 3] = -np.dot(x, self.position)
        view[1, 3] = -np.dot(y, self.position)
        view[2, 3] = -np.dot(z, self.position)
        
        return view
    
    def get_projection_matrix(self, aspect_ratio, near=0.1, far=100.0):
        """Compute perspective projection matrix."""
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        return proj


__all__ = ['Camera']
