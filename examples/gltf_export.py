"""Write a procedural terrain to glTF, to open in Blender, Godot, three.js or any glTF viewer.

    uv run python examples/gltf_export.py              # writes terrain.glb in the working directory
    uv run python examples/gltf_export.py out.glb
"""

import sys

from manifoldx.modeling import Gradient, Mesh, fields

AMOUNT = 4.4
relief = (fields.ridged(seed=5, freq=0.22) * 0.8 + fields.fbm(seed=7, freq=0.8) * 0.2).warp(
    1.4, fx=fields.fbm(seed=2, freq=0.25), fz=fields.fbm(seed=9, freq=0.25))
island = fields.distance().remap(0.0, 8.0, 1.0, 0.0).clamp(0.0, 1.0)
land = Mesh.plane(width=16, depth=16, segments=160).displace((relief * island).power(1.3), amount=AMOUNT)
palette = Gradient([(0.00, "#2f5a8c"), (0.05, "#c2b280"), (0.12, "#4a7a3a"), (0.42, "#5f6e38"),
                    (0.62, "#6e5a44"), (0.80, "#8a8078"), (0.95, "#ffffff")])
terrain = land.color_by(fields.coord("y").remap(0.0, AMOUNT, 0.0, 1.0), palette)

out = sys.argv[1] if len(sys.argv) > 1 else "terrain.glb"
report = terrain.to_gltf(out)
print(f"wrote {out}: {len(terrain.positions)} vertices, {len(terrain.faces)} triangles")
report.print()
