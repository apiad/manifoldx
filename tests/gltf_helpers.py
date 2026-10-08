"""Read back .glb files written by manifoldx.gltf, through pygltflib."""
import numpy as np
from pygltflib import GLTF2

_DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def load(path):
    return GLTF2.load_binary(str(path))


def accessor(g, index):
    acc = g.accessors[index]
    view = g.bufferViews[acc.bufferView]
    blob = g.binary_blob()
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    width = _WIDTH[acc.type]
    a = np.frombuffer(blob, dtype=_DTYPES[acc.componentType], count=acc.count * width, offset=start)
    return a if width == 1 else a.reshape(acc.count, width)


def view_bytes(g, index):
    view = g.bufferViews[index]
    start = view.byteOffset or 0
    return g.binary_blob()[start:start + view.byteLength]
