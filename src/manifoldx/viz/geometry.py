"""Built-in geometries for sci-viz primitives.

SPRITE_QUAD: unit quad in XY plane, expanded into a camera-facing
billboard by the vertex shader. UVs are reconstructed from the
quad-local position (xy in [-1, 1]^2) inside the fragment shader.
"""

import numpy as np


# Vertex layout: position only. Normals reconstructed in fragment shader.
_QUAD_VERTICES = np.array(
    [
        [-1.0, -1.0, 0.0],  # 0: bottom-left
        [1.0, -1.0, 0.0],  # 1: bottom-right
        [1.0, 1.0, 0.0],  # 2: top-right
        [-1.0, 1.0, 0.0],  # 3: top-left
    ],
    dtype=np.float32,
)

# Two triangles: (0, 1, 2) and (0, 2, 3) — counter-clockwise winding
_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


SPRITE_QUAD = {
    "vertices": _QUAD_VERTICES,
    "indices": _QUAD_INDICES,
    "name": "sprite_quad",
}
