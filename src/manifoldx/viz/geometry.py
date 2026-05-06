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


# Unit-line geometries for AxisFrame, one per axis direction.
#
# Each axis is a single line from -1 to +1 along its axis. The entity's
# Transform.scale = (extent, 1, 1) (or (1, extent, 1) etc. — usually keyed
# by which AXIS_LINE_* the entity carries, but the engine doesn't enforce
# this) puts the endpoints at the right world distance.
#
# Topology is LineList — 2 vertices, 2 indices.
AXIS_LINE_X = {
    "vertices": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    "indices": np.array([0, 1], dtype=np.uint32),
    "name": "axis_line_x",
}

AXIS_LINE_Y = {
    "vertices": np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    "indices": np.array([0, 1], dtype=np.uint32),
    "name": "axis_line_y",
}

AXIS_LINE_Z = {
    "vertices": np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "indices": np.array([0, 1], dtype=np.uint32),
    "name": "axis_line_z",
}
