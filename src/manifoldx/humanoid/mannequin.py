"""Spawn a posed figure as rounded volumes (a wooden-mannequin look)."""
import numpy as np

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, sphere

from .solve import PosedFigure

_UNIT_SPHERE = None


def _unit_sphere():
    global _UNIT_SPHERE
    if _UNIT_SPHERE is None:
        _UNIT_SPHERE = sphere(1.0, 24)  # shared so every part batches together
    return _UNIT_SPHERE


def _shade(color: str, k: float) -> str:
    r, g, b = (int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (int(r * k), int(g * k), int(b * k))


def spawn_mannequin(engine, fig: PosedFigure, at=(0.0, 0.0, 0.0), color: str = "#cccccc",
                    roughness: float = 0.6) -> list[int]:
    """Spawn every part of `fig` at world offset `at`; the nose is darker so gaze reads."""
    body = StandardMaterial(color=color, roughness=roughness)
    nose = StandardMaterial(color=_shade(color, 0.7), roughness=roughness)
    offset = np.asarray(at, dtype=np.float64)
    ids = []
    for part in fig.parts:
        handle = engine.spawn(
            Mesh(_unit_sphere()),
            Material(nose if part.kind == "nose" else body),
            Transform(pos=tuple(part.center + offset), rot=tuple(part.rotation), scale=tuple(part.radii)),
        )
        ids.append(handle.index)
    return ids
