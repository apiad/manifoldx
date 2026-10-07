"""Spawn a posed figure as rounded volumes (a wooden-mannequin look)."""
import numpy as np

from manifoldx.components import Material, Mesh, Transform
from manifoldx.gui.style import parse_color
from manifoldx.resources import StandardMaterial, sphere

from .solve import PosedFigure

_UNIT_SPHERE = None


def _unit_sphere():
    global _UNIT_SPHERE
    if _UNIT_SPHERE is None:
        _UNIT_SPHERE = sphere(1.0, 24)  # shared so every part batches together
    return _UNIT_SPHERE


def _shade(color, k: float) -> tuple[float, float, float]:
    """`color` (hex string or float tuple) darkened by `k`."""
    rgb = parse_color(color)[:3] if isinstance(color, str) else tuple(color)[:3]
    return tuple(c * k for c in rgb)


def spawn_mannequin(engine, fig: PosedFigure, at=(0.0, 0.0, 0.0), color="#cccccc",
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
