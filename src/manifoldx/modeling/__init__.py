"""Procedural geometric modeling: numpy-first Mesh value type + operator pipeline."""

from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.sculpt import Falloff
from manifoldx.modeling.ffd import FFD
from manifoldx.modeling.fields import Field
from manifoldx.modeling import fields, noise

__all__ = ["Mesh", "Falloff", "FFD", "Field", "fields", "noise"]
