"""Procedural geometric modeling: numpy-first Mesh value type + operator pipeline."""

from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.sculpt import Falloff
from manifoldx.modeling.ffd import FFD
from manifoldx.modeling import noise

__all__ = ["Mesh", "Falloff", "FFD", "noise"]
