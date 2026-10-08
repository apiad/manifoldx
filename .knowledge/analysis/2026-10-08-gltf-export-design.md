# glTF export

**Status:** draft 2026-10-08, for Alex's review. Issue: #23.

## Goal

Get a manifoldx scene or a `manifoldx.modeling` mesh out of manifoldx as a
glTF 2.0 binary (`.glb`) that Blender, Godot, Unity and three.js open, and say
explicitly what the file could not carry. Nothing is dropped silently.

Two clients drive it:

- **uh-twin's web viewer** (syalia-srl/uh-twin#14). Streaming manifoldx frames
  to a browser measured 10.4 fps at 1280x720, bounded by about 41 ms of
  manifoldx CPU per frame, and manifoldx cannot run inside a browser today
  (wgpu-py's `js_webgpu` backend is a stub, pygfx/wgpu-py#753 and #846). A
  static `.glb` rendered by three.js in the visitor's browser avoids both.
- **manifoldx as a procedural authoring tool.** Terrain from noise, buildings
  from CSG and sculpted meshes become usable in any 3D tool.

## Decisions taken with Alex

- **manifoldx ships a writer, not a runtime.** An earlier draft had a JS
  runtime in manifoldx that re-implemented the engine's behaviours in the
  browser. Alex rejected it on 2026-10-08: every new subsystem would need a
  web twin to maintain. glTF fixes the scope from outside: a system or a
  volume can never fit in a `.glb`, so the export does not grow when the
  engine grows. Interactive behaviour belongs to whoever builds a viewer
  (uh-twin builds its own).
- **Report, don't drop silently.** Everything the writer skips or approximates
  is listed by name. The default is to warn and write the file; `strict=True`
  (`--strict`) raises (exits with code 1) instead.
- **No new runtime dependency.** The writer is hand-written on numpy and the
  standard library. `pygltflib` is a test-only dependency.

## Scope

In:

1. `manifoldx.gltf`: the writer, `Node`, `export_gltf(path, nodes)`.
2. `manifoldx.modeling.Mesh.to_gltf(path, material=None)`.
3. `Engine.export_gltf(path, *, names=None, extras=None, strict=False)`
   returning an `ExportReport`.
4. The `manifoldx export` command.
5. `TextureHandle.source`: the image path `load_texture` read.
6. `examples/gltf_export.py`: a procedural terrain written to `.glb`.

Out: animation, skins, morph targets, Draco or meshopt compression, `.gltf`
with external files, importing glTF, normal and metallic-roughness maps
(manifoldx has none yet), touch or web viewers.

## 1. The writer: `manifoldx.gltf`

```python
from manifoldx.gltf import Node, export_gltf

export_gltf("part.glb", [
    Node(geometry, material=StandardMaterial("#c2b69c", roughness=0.9),
         pos=(0, 0, 0), rot=(0, 0, 0, 1), scale=(1, 1, 1),
         name="wall", extras={"pick": "b123"}),
], camera=None, lights=(), environment=None, scene_extras=None) -> ExportReport
```

The writer returns the report of what it approximated or dropped (section
3 describes the report); `Engine.export_gltf` adds the engine-level entries
to it. Light colours are written the way the engine uploads them (hex over
255, no sRGB decode), and spot angles are already radians in manifoldx.

`geometry` is a geometry dict (the form `load_obj`, `cube`, `sphere` and
`Mesh.to_geometry` return) or a `modeling.Mesh`. Everything below is the
mapping the writer applies. manifoldx and glTF agree on axes (right-handed,
y up, metres), quaternion order (x, y, z, w) and the texture origin (top-left,
since #8), so positions, rotations and UVs copy through unchanged.

| manifoldx | glTF | fidelity |
|---|---|---|
| `positions` / `vertices`, `normals`, `uvs`, indices | one mesh primitive: `POSITION` (with min/max), `NORMAL`, `TEXCOORD_0`, `uint32` or `uint16` indices | exact |
| `colors` (linear RGB, as the vertex-colour shader reads them) | `COLOR_0`, written only when the material has `vertex_colors=True` (glTF always multiplies by `COLOR_0`) | exact |
| same geometry and material on many nodes | one glTF mesh, referenced by every node | exact |
| `StandardMaterial` color, roughness, metallic | `pbrMetallicRoughness`; `baseColorFactor` is the colour decoded to linear with `material_rgb`, as glTF requires | exact |
| `StandardMaterial.albedo_map` | `baseColorTexture`; the original JPEG or PNG bytes; sampler `REPEAT`, `LINEAR_MIPMAP_LINEAR`, `LINEAR` | exact |
| `StandardMaterial.ao != 1` | dropped: glTF has occlusion only as a texture | reported |
| `FlatMaterial` | `KHR_materials_unlit` | exact |
| `BasicMaterial`, `PhongMaterial` | PBR, roughness 1, metallic 0; their fixed head-on light is lost | approximated, reported |
| alpha below 1 in a colour | `alphaMode: BLEND` | exact |
| any other material | the node is not written | dropped, reported |
| texture without `source` (created outside `load_texture`) | the node keeps its material without the texture | reported |

Buffers are packed into the single GLB binary chunk with 4-byte alignment.
Node names default to `node_<i>`. `extras` must be JSON-serialisable; the
writer checks it and raises `TypeError` with the node's name otherwise.

## 2. `Mesh.to_gltf`

`modeling.Mesh.to_gltf(path, material=None)` writes one node at the origin.
Without a material it writes a white `StandardMaterial(roughness=1)`, or one
with `vertex_colors=True` when the mesh carries colours. This is the whole
path for "generate a terrain in manifoldx, open it in Blender": no engine, no
device, no GPU.

## 3. `Engine.export_gltf`

```python
report = engine.export_gltf("scene.glb", names={idx: "wall"},
                            extras={idx: {"pick": "b123"}}, strict=False)
```

**When the snapshot is taken.** If the engine has not started and has
`startup` handlers, `export_gltf` starts it the way `render_frame` does
(`_ensure_offscreen`): an offscreen canvas, the device, and the `startup`
event. Without `startup` handlers it reads the store as it is and needs no
device, which keeps most tests runnable in CI. Apps that spawn in `startup`
(uh-twin does, because `load_texture` needs the device) export the same as
apps that spawn before `run()`. No system runs and no frame is drawn; the
file holds the state the app left before `run()`, including the camera pose.

**What it reads.** Every alive entity with `Transform`, `Mesh` and `Material`
on the mesh path becomes a node through section 1. On top of the nodes:

| engine state | glTF | notes |
|---|---|---|
| `camera` (fov, near, far, position, target, up) | a camera node: `perspective.yfov` in radians (manifoldx's `fov` is vertical, in degrees), `znear`, `zfar`; the rotation looks from `position` to `target` | exact |
| `set_sun(DirectionalLight)` | `KHR_lights_punctual` directional light on a node rotated to the light's direction | colour exact; manifoldx's unitless intensity is copied and also stored in the light's `extras.manifoldx_intensity` |
| `add_light`, `set_lights` (point), `set_spot` | `KHR_lights_punctual` point and spot lights, `range` from `distance` when non-zero, cone angles from the spot's inner and outer angles | as above |
| `enable_shadows`, `enable_fog`, `background_color` | `scenes[0].extras.manifoldx`: `{"shadows": {...}, "fog": {...}, "background": [r, g, b]}` | other tools ignore extras; a viewer may read them |
| `set_environment(EnvironmentMap)` | the equirectangular float32 radiance (H, W, 3) as one buffer view; `extras.manifoldx.environment = {"bufferView": n, "width", "height", "intensity"}` | as above |

**The report.** `ExportReport` is a list of entries `(kind, name, where,
effect, detail)`: `kind` is one of `system`, `handler`, `gui`, `compute`,
`task`, `volume`, `entity`, `material`, `texture`, `light`; `where` is
`file:line` when the thing is Python code; `effect` is `dropped` or
`approximated`. It lists:

- every system (`engine.system`), by qualified name and source line;
- every event handler except `startup`, by event and handler;
- every GUI root panel and each widget with a getter or callback;
- every compute kernel and every pending background task;
- every volume, point cloud, text label and axis entity, grouped with a count;
- every material class from section 1 that is dropped or approximated, with
  the number of entities;
- every texture without a source, and `ao != 1`.

`report.print()` writes the grouped summary to stderr;
`report.write_json(path)` writes it as JSON. `export_gltf` always writes the
JSON next to the `.glb` (`scene.glb` gets `scene.export-report.json`) and
prints the summary. With `strict=True` it raises `ExportIncomplete` (which
carries the report) before writing the `.glb` if the report is not empty.

## 4. The `manifoldx export` command

```
uv run manifoldx export app.py scene.glb [--strict] [-- <app args>]
uv run manifoldx export -m uh_twin.app scene.glb [-- <app args>]
```

The command loads the script or module as `__main__` with `runpy`, with
`sys.argv` set to the app's own arguments, and replaces `Engine.run` for that
process with a function that calls `export_gltf` on the engine and returns.
Any app ending in `engine.run()` or `engine.cli()` (which calls `run()`)
exports without changes. If the app finishes without calling `run()`, the
command exits with code 2 and says so. Exit codes: 0 written, 1 strict and
incomplete, 2 the app never reached `run()`. It is registered as
`[project.scripts] manifoldx = "manifoldx.cli:main"`, with `export` as its
only subcommand for now.

## 5. `TextureHandle.source`

`load_texture` stores `source=Path(path)` on the handle. The writer embeds
those bytes when the file is JPEG or PNG and re-encodes to PNG through Pillow
otherwise. PR #22 (mipmaps) edits `load_texture` too; whichever lands second
rebases.

## Testing

- **Writer round-trip** (`tests/test_gltf.py`, no GPU): write nodes with
  every mapped material, read them back with `pygltflib`, compare positions,
  normals, UVs, indices, transforms, material factors, the texture bytes,
  extras, the camera and the lights.
- **Engine export** (`tests/test_gltf_engine.py`, no GPU): a scene spawned
  before export, with a system, a handler, a GUI panel, a `VolumeMaterial`
  entity and a `BasicMaterial` entity; the report lists each one exactly once
  with the right effect, and `strict=True` raises without writing the file.
  One GPU-gated test covers a scene that spawns in `startup` and uses
  `load_texture`.
- **The command**: run it on a small script fixture; check the file, the
  exit codes 0, 1 and 2.
- **Spec conformance**: the Khronos validator (`npx gltf-validator`) reports
  zero errors on the test outputs. Skipped when `node` is missing.
- **Blender**: `blender -b --python-expr` imports the example's `.glb` and
  prints the mesh and material counts. Skipped when `blender` is missing
  (it runs on zion).
- To make sure these checks can fail, break the writer on purpose once
  (swap the quaternion order, drop the 4-byte padding) and watch them go red.

## Open questions

None at the time of writing.
