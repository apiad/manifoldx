"""Quaternion helpers, (x, y, z, w) like Transform.rot."""
import math

import numpy as np


def axis_angle(axis, degrees: float) -> np.ndarray:
    a = math.radians(degrees) / 2
    s = math.sin(a)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(a)])


def mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def rotate(q, v) -> np.ndarray:
    u, w = np.asarray(q[:3]), q[3]
    v = np.asarray(v, dtype=np.float64)
    return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)


def to_matrix(q) -> np.ndarray:
    return np.stack([rotate(q, e) for e in np.eye(3)], axis=1)
