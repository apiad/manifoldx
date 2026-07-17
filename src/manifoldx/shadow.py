"""Light-space matrices for shadow mapping (pure numpy, no GPU)."""

import numpy as np


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(np.dot(f, up)) > 0.999:  # degenerate: sun straight down/up
        up = np.array([0.0, 0.0, 1.0])
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def _ortho(extent, near, far):
    """Symmetric ortho box [-extent, extent]^2, depth near..far, wgpu z in [0,1]."""
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = 1.0 / extent
    M[1, 1] = 1.0 / extent
    M[2, 2] = -1.0 / (far - near)
    M[2, 3] = -near / (far - near)
    M[3, 3] = 1.0
    return M


def compute_light_view_proj(direction, target, extent, near, far, back_distance=None):
    """Ortho light-space view-projection for a directional sun.

    Returns a (4, 4) float32 matrix M with ``M @ [x, y, z, 1]`` -> clip coords
    (wgpu convention: x, y in [-1, 1], z in [0, 1]).
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    target = np.asarray(target, dtype=np.float64)
    if back_distance is None:
        back_distance = far * 0.5
    eye = target - direction * back_distance
    view = _look_at(eye, target, up=np.array([0.0, 1.0, 0.0]))
    proj = _ortho(extent, near, far)
    return (proj @ view).astype(np.float32)
