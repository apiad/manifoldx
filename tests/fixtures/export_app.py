"""A tiny app for the export command's tests: --basic adds an approximated material, --no-run skips run()."""

import sys

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import BasicMaterial, StandardMaterial, cube

engine = mx.Engine("fixture", width=64, height=48)
material = BasicMaterial("#00ff00") if "--basic" in sys.argv else StandardMaterial("#ffffff")
engine.spawn(Mesh(cube(1, 1, 1)), Material(material), Transform(pos=(0, 1, 0)))
if "--no-run" not in sys.argv:
    engine.run()
