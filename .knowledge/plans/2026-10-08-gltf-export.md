# glTF export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write manifoldx scenes and modeling meshes to `.glb` files and report by name everything the file cannot carry.

**Architecture:** A dependency-free GLB builder (`gltf/glb.py`) under a writer that maps manifoldx geometry, materials, textures, lights, camera and environment to glTF (`gltf/writer.py`) and returns an `ExportReport` (`gltf/report.py`). `Engine.export_gltf` snapshots the ECS store into writer nodes and adds engine-level report entries (`gltf/engine_export.py`). `manifoldx export` runs any app with `Engine.run` swapped for the export (`cli.py`).

**Tech Stack:** Python 3.13, numpy, the standard library (`struct`, `json`, `runpy`); `pygltflib` for tests only; Khronos `gltf-validator` (npm) and Blender as optional conformance checks.

**Spec:** `.knowledge/analysis/2026-10-08-gltf-export-design.md`

## Global Constraints

- No new runtime dependency. `pygltflib` goes in the `dev` dependency group only.
- glTF 2.0 binary only (`.glb`), one buffer, 4-byte aligned chunks and views.
- Axes, quaternion order (x, y, z, w) and UV origin pass through unchanged.
- Material colours are written linear (`material_rgb`); light colours as the engine uploads them (hex / 255); vertex colours unchanged.
- Nothing is dropped silently: every skip or approximation is a report entry.
- `strict=True` raises `ExportIncomplete` and writes no file.
- Tests that need a GPU skip through `get_offscreen_canvas`, like the rest of the suite; CI has no GPU.
- English for code, comments and messages; CHANGELOG entry under `[Unreleased]`.

## Review Focus

- An app whose only output is GUI and systems (no meshes): the export must still write a valid `.glb` (no empty arrays, which the glTF schema forbids) and a report listing everything. Test in Task 2 (empty scene) and Task 5.
- A mesh with 65,535 or more vertices: indices must switch to `uint32`, because glTF forbids the maximum value of the component type as an index. Test in Task 2.
- manifoldx's own internal handlers (`_GuiBridge` registers `pointer_down/move/up`) must not show up in the report of a plain app, or every report is noise. Test in Task 5 (fresh engine reports nothing).
- Non-JSON extras (a numpy float, a Path) from an app: the writer must fail with the node's name, not with a bare `json` traceback deep in the builder. Test in Task 2.
- The same geometry dict spawned many times must be written once (uh-twin's trees reuse geometry). Test in Task 2 (two nodes, one mesh).

---

### Task 1: `TextureHandle.source`

**Files:**
- Modify: `src/manifoldx/textures.py` (the `TextureHandle` dataclass and `load_texture`)
- Test: `tests/test_textures_source.py`

**Interfaces:**
- Produces: `TextureHandle.source: Path | None = None`, set by `load_texture` to `Path(path)`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from manifoldx.textures import TextureHandle


def test_handle_has_optional_source():
    h = TextureHandle(id=1, texture=None, view=None, sampler=None, size=(2, 2))
    assert h.source is None
    h2 = TextureHandle(id=2, texture=None, view=None, sampler=None, size=(2, 2), source=Path("a.png"))
    assert h2.source == Path("a.png")
```

Add a GPU-gated test in the same file that calls `load_texture` on a 2x2 PNG written with Pillow into `tmp_path` and asserts `handle.source == path` (skip like `tests/test_textures.py` does when no offscreen canvas).

- [ ] **Step 2: Run** `uv run pytest tests/test_textures_source.py -q`. Expected: FAIL, unexpected keyword `source`.
- [ ] **Step 3: Implement.** Add `source: Path | None = None` as the last field of `TextureHandle` (comment: `# the image file load_texture read; glTF export embeds it`) and pass `source=p` in `load_texture`'s `TextureHandle(...)` call.
- [ ] **Step 4: Run** the test file. Expected: PASS (the GPU test passes on zion).
- [ ] **Step 5: Commit** `feat(textures): remember the image file a TextureHandle came from`.

---

### Task 2: GLB builder and the geometry/material writer

**Files:**
- Create: `src/manifoldx/gltf/__init__.py`, `src/manifoldx/gltf/glb.py`, `src/manifoldx/gltf/report.py`, `src/manifoldx/gltf/writer.py`
- Modify: `pyproject.toml` (`pygltflib` in `[dependency-groups] dev`)
- Test: `tests/test_gltf_writer.py`, `tests/gltf_helpers.py`

**Interfaces:**
- Produces:
  - `GlbBuilder()` with `.add(key, item) -> int`, `.view(data: bytes, target=None) -> int`, `.accessor(array, *, target=None, bounds=False) -> int`, `.use_extension(name)`, `.to_bytes() -> bytes`.
  - `ExportReport` with `.add(kind, name, effect, detail, where=None, count=1)`, `.extend(other)`, `len()`, `bool()`, iteration over `Entry(kind, name, effect, detail, where, count)`, `.format() -> str`, `.print(file=None)`, `.to_dict()`, `.write_json(path)`; `ExportIncomplete(report)` with `.report`.
  - `Node(geometry, material=None, pos=(0,0,0), rot=(0,0,0,1), scale=(1,1,1), name=None, extras=None)`.
  - `build_glb(nodes, *, camera=None, lights=(), environment=None, scene_extras=None) -> tuple[bytes, ExportReport]`.
  - `export_gltf(path, nodes, **same) -> ExportReport` (writes the file).

- [ ] **Step 1: Write the test helpers** (`tests/gltf_helpers.py`):

```python
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
```

- [ ] **Step 2: Write the failing tests** (`tests/test_gltf_writer.py`):

```python
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from manifoldx.gltf import Node, export_gltf
from manifoldx.resources import BasicMaterial, FlatMaterial, PhongMaterial, StandardMaterial, cube
from manifoldx.textures import TextureHandle

from gltf_helpers import accessor, load, view_bytes


def tri(n=3):
    pos = np.zeros((n, 3), np.float32)
    pos[1] = (1, 0, 0)
    pos[2] = (0, 1, 0)
    return {"positions": pos, "normals": np.tile([0, 0, 1], (n, 1)).astype(np.float32),
            "uvs": np.zeros((n, 2), np.float32), "indices": np.array([0, 1, 2], np.uint32)}


def test_glb_header_and_alignment(tmp_path):
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(cube(1, 1, 1), StandardMaterial("#ffffff"))])
    data = out.read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_len, json_type = struct.unpack("<II", data[12:20])
    assert json_type == 0x4E4F534A and json_len % 4 == 0
    doc = json.loads(data[20:20 + json_len])
    assert all(v["byteOffset"] % 4 == 0 for v in doc["bufferViews"])


def test_geometry_and_transform_round_trip(tmp_path):
    geo = cube(2, 1, 1)
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff"), pos=(1, 2, 3), rot=(0, 0.7071068, 0, 0.7071068),
                           scale=(2, 2, 2), name="box", extras={"pick": "b1"})])
    g = load(out)
    node = g.nodes[0]
    assert node.name == "box" and node.extras == {"pick": "b1"}
    assert node.translation == [1, 2, 3] and node.scale == [2, 2, 2]
    assert node.rotation == pytest.approx([0, 0.7071068, 0, 0.7071068])
    prim = g.meshes[node.mesh].primitives[0]
    np.testing.assert_allclose(accessor(g, prim.attributes.POSITION), geo["positions"])
    np.testing.assert_allclose(accessor(g, prim.attributes.NORMAL), geo["normals"])
    np.testing.assert_array_equal(accessor(g, prim.indices), geo["indices"])
    assert g.accessors[prim.attributes.POSITION].min == pytest.approx(geo["positions"].min(axis=0).tolist())


def test_identity_transform_is_omitted(tmp_path):
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(cube(1, 1, 1), StandardMaterial("#ffffff"))])
    node = load(out).nodes[0]
    assert node.translation is None and node.rotation is None and node.scale is None
    assert node.name == "node_0"


def test_shared_geometry_is_written_once(tmp_path):
    geo, mat = cube(1, 1, 1), StandardMaterial("#ffffff")
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, mat, pos=(0, 0, 0)), Node(geo, mat, pos=(3, 0, 0))])
    g = load(out)
    assert len(g.meshes) == 1 and g.nodes[0].mesh == g.nodes[1].mesh == 0


def test_large_meshes_use_uint32_indices(tmp_path):
    n = 70_000
    geo = {"positions": np.random.default_rng(0).random((n, 3), dtype=np.float32),
           "indices": np.arange(n - n % 3, dtype=np.uint32)}
    small = tri()
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff")), Node(small, StandardMaterial("#ffffff"))])
    g = load(out)
    big_idx = g.meshes[g.nodes[0].mesh].primitives[0].indices
    small_idx = g.meshes[g.nodes[1].mesh].primitives[0].indices
    assert g.accessors[big_idx].componentType == 5125
    assert g.accessors[small_idx].componentType == 5123


def test_standard_material_is_linear_pbr(tmp_path):
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(tri(), StandardMaterial("#cc2222", roughness=0.4, metallic=0.6))])
    m = load(out).materials[0]
    r = (0.8 + 0.055) / 1.055
    assert m.pbrMetallicRoughness.baseColorFactor == pytest.approx([r ** 2.4, 0.016, 0.016, 1.0], abs=1e-3)
    assert m.pbrMetallicRoughness.roughnessFactor == pytest.approx(0.4)
    assert m.pbrMetallicRoughness.metallicFactor == pytest.approx(0.6)


def test_albedo_map_embeds_the_original_bytes(tmp_path):
    from PIL import Image

    src = tmp_path / "t.jpg"
    Image.new("RGB", (4, 4), (200, 100, 50)).save(src)
    tex = TextureHandle(id=7, texture=None, view=None, sampler=None, size=(4, 4), source=src)
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(tri(), StandardMaterial("#ffffff", albedo_map=tex)),
                      Node(tri(), StandardMaterial("#ffffff", albedo_map=tex))])
    g = load(out)
    assert len(g.images) == 1 and g.images[0].mimeType == "image/jpeg"
    assert view_bytes(g, g.images[0].bufferView) == src.read_bytes()
    s = g.samplers[0]
    assert (s.wrapS, s.wrapT, s.minFilter, s.magFilter) == (10497, 10497, 9987, 9729)
    assert g.materials[0].pbrMetallicRoughness.baseColorTexture.index == 0


def test_texture_without_source_is_reported(tmp_path):
    tex = TextureHandle(id=3, texture=None, view=None, sampler=None, size=(4, 4))
    report = export_gltf(tmp_path / "a.glb", [Node(tri(), StandardMaterial("#ffffff", albedo_map=tex))])
    assert [(e.kind, e.effect) for e in report] == [("texture", "dropped")]
    assert load(tmp_path / "a.glb").materials[0].pbrMetallicRoughness.baseColorTexture is None


def test_vertex_colors_only_with_a_vertex_color_material(tmp_path):
    geo = tri() | {"colors": np.full((3, 3), 0.5, np.float32)}
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff", vertex_colors=True)),
                      Node(geo, StandardMaterial("#ffffff"))])
    g = load(out)
    with_c = g.meshes[g.nodes[0].mesh].primitives[0].attributes
    without = g.meshes[g.nodes[1].mesh].primitives[0].attributes
    np.testing.assert_allclose(accessor(g, with_c.COLOR_0), geo["colors"])
    assert without.COLOR_0 is None


def test_flat_is_unlit_and_basic_is_approximated(tmp_path):
    out = tmp_path / "a.glb"
    report = export_gltf(out, [Node(tri(), FlatMaterial("#ff0000")), Node(tri(), BasicMaterial("#00ff00")),
                               Node(tri(), PhongMaterial("#0000ff")), Node(tri(), BasicMaterial((1, 1, 1, 0.5)))])
    g = load(out)
    assert "KHR_materials_unlit" in g.materials[0].extensions
    assert "KHR_materials_unlit" in g.extensionsUsed
    assert g.materials[1].pbrMetallicRoughness.roughnessFactor == 1.0
    assert g.materials[3].alphaMode == "BLEND"
    rows = {(e.name, e.effect, e.count) for e in report}
    assert ("BasicMaterial", "approximated", 2) in rows and ("PhongMaterial", "approximated", 1) in rows


def test_ao_is_reported(tmp_path):
    report = export_gltf(tmp_path / "a.glb", [Node(tri(), StandardMaterial("#ffffff", ao=0.5))])
    assert [(e.kind, e.name, e.effect) for e in report] == [("material", "StandardMaterial.ao", "dropped")]


def test_unknown_material_drops_the_node(tmp_path):
    class Glow:
        pass

    out = tmp_path / "a.glb"
    report = export_gltf(out, [Node(tri(), Glow()), Node(tri(), Glow()), Node(tri(), StandardMaterial("#fff"))])
    assert len(load(out).nodes) == 1
    assert [(e.kind, e.name, e.effect, e.count) for e in report] == [("material", "Glow", "dropped", 2)]


def test_bad_extras_name_the_node(tmp_path):
    with pytest.raises(TypeError, match="'wall'"):
        export_gltf(tmp_path / "a.glb", [Node(tri(), StandardMaterial("#fff"), name="wall", extras={"p": Path("x")})])


def test_empty_scene_is_valid(tmp_path):
    out = tmp_path / "a.glb"
    export_gltf(out, [])
    g = load(out)
    assert g.asset.version == "2.0" and not g.meshes and not g.accessors


def test_modeling_mesh_is_accepted(tmp_path):
    from manifoldx.modeling import Mesh as GeoMesh

    out = tmp_path / "a.glb"
    export_gltf(out, [Node(GeoMesh.plane(width=2, depth=2, segments=4), StandardMaterial("#ffffff"))])
    assert len(load(out).meshes) == 1
```

- [ ] **Step 3: Run** `uv run pytest tests/test_gltf_writer.py -q`. Expected: FAIL, `ModuleNotFoundError: manifoldx.gltf` (after adding `pygltflib` with `uv add --dev pygltflib`).

- [ ] **Step 4: Implement `glb.py`.**

```python
"""Binary glTF (GLB): a JSON document plus one binary buffer, both 4-byte aligned."""

import json
import struct

import numpy as np

FLOAT, UINT16, UINT32 = 5126, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
_COMPONENT = {np.dtype(np.float32): FLOAT, np.dtype(np.uint16): UINT16, np.dtype(np.uint32): UINT32}
_TYPE = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}


class GlbBuilder:
    """Accumulates glTF objects and binary data; to_bytes() returns the .glb."""

    def __init__(self):
        self.doc = {"asset": {"version": "2.0", "generator": "manifoldx"}}
        self.bin = bytearray()

    def add(self, key, item):
        """Append item to the document's top-level array `key`; return its index."""
        items = self.doc.setdefault(key, [])
        items.append(item)
        return len(items) - 1

    def view(self, data, target=None):
        self.bin.extend(b"\0" * (-len(self.bin) % 4))
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.bin.extend(data)
        return self.add("bufferViews", view)

    def accessor(self, array, *, target=None, bounds=False):
        a = np.ascontiguousarray(array)
        width = 1 if a.ndim == 1 else a.shape[1]
        acc = {"bufferView": self.view(a.tobytes(), target), "componentType": _COMPONENT[a.dtype],
               "count": int(a.shape[0]), "type": _TYPE[width]}
        if bounds:
            flat = a.reshape(len(a), width)
            acc["min"] = flat.min(axis=0).tolist()
            acc["max"] = flat.max(axis=0).tolist()
        return self.add("accessors", acc)

    def use_extension(self, name):
        used = self.doc.setdefault("extensionsUsed", [])
        if name not in used:
            used.append(name)

    def to_bytes(self):
        doc = dict(self.doc)
        if self.bin:
            doc["buffers"] = [{"byteLength": len(self.bin)}]
        text = json.dumps(doc, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        chunks = struct.pack("<II", len(text), 0x4E4F534A) + text
        if self.bin:
            binary = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
            chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
        return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks
```

- [ ] **Step 5: Implement `report.py`.**

```python
"""What an export left out or changed, by name, so nothing is dropped silently."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Entry:
    kind: str  # system, handler, gui, compute, task, material, texture, light
    name: str
    effect: str  # "dropped" or "approximated"
    detail: str
    where: str | None = None  # file:line for Python code
    count: int = 1


class ExportReport:
    def __init__(self):
        self.entries: list[Entry] = []

    def add(self, kind, name, effect, detail, where=None, count=1):
        for e in self.entries:
            if (e.kind, e.name, e.effect, e.detail, e.where) == (kind, name, effect, detail, where):
                e.count += count
                return
        self.entries.append(Entry(kind, name, effect, detail, where, count))

    def extend(self, other):
        for e in other.entries:
            self.add(e.kind, e.name, e.effect, e.detail, e.where, e.count)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def format(self):
        if not self.entries:
            return "glTF export: everything was exported."
        lines = [f"glTF export: {len(self.entries)} things not exported or approximated:"]
        for e in self.entries:
            what = e.name + (f" x{e.count}" if e.count > 1 else "")
            where = f" ({e.where})" if e.where else ""
            lines.append(f"  {e.kind:<8} {e.effect:<12} {what}{where}: {e.detail}")
        return "\n".join(lines)

    def print(self, file=None):
        print(self.format(), file=file or sys.stderr)

    def to_dict(self):
        return {"complete": not self.entries, "entries": [asdict(e) for e in self.entries]}

    def write_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


class ExportIncomplete(RuntimeError):
    """Raised by a strict export when the report is not empty."""

    def __init__(self, report):
        super().__init__(report.format())
        self.report = report
```

- [ ] **Step 6: Implement `writer.py`** (geometry, materials and textures; Task 3 adds the camera, lights and environment keywords to `build_glb` and `export_gltf`).

```python
"""Map manifoldx geometry and materials to glTF. Returns what it could not map."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldx.gltf.glb import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, GlbBuilder
from manifoldx.gltf.report import ExportReport

_LINEAR, _LINEAR_MIPMAP_LINEAR, _REPEAT = 9729, 9987, 10497


@dataclass
class Node:
    geometry: object  # geometry dict or anything with to_geometry() (modeling.Mesh)
    material: object = None
    pos: tuple = (0.0, 0.0, 0.0)
    rot: tuple = (0.0, 0.0, 0.0, 1.0)  # quaternion x, y, z, w
    scale: tuple = (1.0, 1.0, 1.0)
    name: str | None = None
    extras: dict | None = None


def build_glb(nodes):
    w = _Writer()
    for i, node in enumerate(nodes):
        w.node(node, i)
    return w.finish(), w.report


def export_gltf(path, nodes):
    data, report = build_glb(nodes)
    Path(path).write_bytes(data)
    return report


def _rgba(color):
    from manifoldx.resources import material_rgb

    alpha = 1.0 if isinstance(color, str) or len(color) == 3 else float(color[3])
    return [*map(float, material_rgb(color)), alpha]


def _image(path):
    data = Path(path).read_bytes()
    if data[:3] == b"\xff\xd8\xff":
        return data, "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data, "image/png"
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.open(path).save(buf, format="PNG")
    return buf.getvalue(), "image/png"


class _Writer:
    def __init__(self):
        self.b = GlbBuilder()
        self.report = ExportReport()
        self._meshes = {}  # (id(geometry), material index, colours) -> mesh index
        self._materials = {}  # id(material) -> (index or None, note or None)
        self._textures = {}  # TextureHandle.id -> texture index or None
        self._sampler = None
        self._roots = []

    def node(self, node, i):
        name = node.name or f"node_{i}"
        if node.extras is not None:
            try:
                json.dumps(node.extras)
            except TypeError as e:
                raise TypeError(f"extras of node {name!r} are not JSON-serialisable: {e}") from None
        mat, note = self.material(node.material)
        if note is not None:
            self.report.add("material", *note)
            if note[1] == "dropped" and note[0] != "StandardMaterial.ao":
                return
        colors = bool(getattr(node.material, "vertex_colors", False))
        key = (id(node.geometry), mat, colors)
        if key not in self._meshes:
            geo = node.geometry.to_geometry() if hasattr(node.geometry, "to_geometry") else node.geometry
            self._meshes[key] = self.b.add("meshes", {"primitives": [self.primitive(geo, mat, colors)]})
        entry = {"name": name, "mesh": self._meshes[key]}
        for key_, value, default in (("translation", node.pos, (0, 0, 0)), ("rotation", node.rot, (0, 0, 0, 1)),
                                     ("scale", node.scale, (1, 1, 1))):
            value = [float(v) for v in value]
            if value != list(default):
                entry[key_] = value
        if node.extras is not None:
            entry["extras"] = node.extras
        self._roots.append(self.b.add("nodes", entry))

    def primitive(self, geo, mat, colors):
        pos = np.asarray(geo["positions"] if "positions" in geo else geo["vertices"], np.float32)
        attrs = {"POSITION": self.b.accessor(pos, target=ARRAY_BUFFER, bounds=True)}
        if "normals" in geo:
            attrs["NORMAL"] = self.b.accessor(np.asarray(geo["normals"], np.float32), target=ARRAY_BUFFER)
        if "uvs" in geo:
            attrs["TEXCOORD_0"] = self.b.accessor(np.asarray(geo["uvs"], np.float32), target=ARRAY_BUFFER)
        if colors and "colors" in geo:
            attrs["COLOR_0"] = self.b.accessor(np.asarray(geo["colors"], np.float32), target=ARRAY_BUFFER)
        prim = {"attributes": attrs, "mode": 4}
        if "indices" in geo:
            # glTF forbids the component type's maximum value as an index.
            dtype = np.uint16 if len(pos) < 0xFFFF else np.uint32
            idx = np.asarray(geo["indices"]).reshape(-1).astype(dtype)
            prim["indices"] = self.b.accessor(idx, target=ELEMENT_ARRAY_BUFFER)
        if mat is not None:
            prim["material"] = mat
        return prim

    def material(self, mat):
        if mat is None:
            return None, None
        if id(mat) not in self._materials:
            self._materials[id(mat)] = self._material(mat)
        return self._materials[id(mat)]

    def _material(self, mat):
        from manifoldx.resources import BasicMaterial, FlatMaterial, PhongMaterial, StandardMaterial

        kind = type(mat)  # exact types: viz materials subclass BasicMaterial
        name = kind.__name__
        note = None
        if kind is StandardMaterial:
            pbr = {"baseColorFactor": _rgba(mat.color), "metallicFactor": float(mat.metallic),
                   "roughnessFactor": float(mat.roughness)}
            if mat.albedo_map is not None:
                tex = self.texture(mat.albedo_map)
                if tex is not None:
                    pbr["baseColorTexture"] = {"index": tex}
            if mat.ao != 1.0:
                note = ("StandardMaterial.ao", "dropped", "glTF has ambient occlusion only as a texture")
            out = {"pbrMetallicRoughness": pbr}
        elif kind is FlatMaterial:
            out = {"pbrMetallicRoughness": {"baseColorFactor": _rgba(mat.color), "metallicFactor": 0.0,
                                            "roughnessFactor": 1.0},
                   "extensions": {"KHR_materials_unlit": {}}}
            self.b.use_extension("KHR_materials_unlit")
        elif kind in (BasicMaterial, PhongMaterial):
            out = {"pbrMetallicRoughness": {"baseColorFactor": _rgba(mat.color), "metallicFactor": 0.0,
                                            "roughnessFactor": 1.0}}
            note = (name, "approximated", "written as rough PBR, lit by the scene instead of its fixed head-on light")
        else:
            return None, (name, "dropped", "no glTF equivalent; its entities are not exported")
        out["name"] = name
        if out["pbrMetallicRoughness"]["baseColorFactor"][3] < 1.0:
            out["alphaMode"] = "BLEND"
        return self.b.add("materials", out), note

    def texture(self, handle):
        if handle.id not in self._textures:
            source = getattr(handle, "source", None)
            if source is None:
                self.report.add("texture", f"texture {handle.id}", "dropped",
                                "made outside load_texture, so its pixels are unknown")
                self._textures[handle.id] = None
            else:
                data, mime = _image(source)
                if self._sampler is None:
                    self._sampler = self.b.add("samplers", {"magFilter": _LINEAR, "minFilter": _LINEAR_MIPMAP_LINEAR,
                                                            "wrapS": _REPEAT, "wrapT": _REPEAT})
                image = self.b.add("images", {"bufferView": self.b.view(data), "mimeType": mime})
                self._textures[handle.id] = self.b.add("textures", {"sampler": self._sampler, "source": image})
        return self._textures[handle.id]

    def finish(self):
        scene = {"nodes": self._roots} if self._roots else {}
        self.b.doc["scenes"] = [scene]
        self.b.doc["scene"] = 0
        return self.b.to_bytes()
```

`__init__.py`:

```python
"""glTF 2.0 export: write manifoldx scenes and meshes to .glb files."""

from manifoldx.gltf.report import Entry, ExportIncomplete, ExportReport
from manifoldx.gltf.writer import Node, build_glb, export_gltf

__all__ = ["Entry", "ExportIncomplete", "ExportReport", "Node", "build_glb", "export_gltf"]
```

- [ ] **Step 7: Run** `uv run pytest tests/test_gltf_writer.py -q`. Expected: all PASS. Then break it on purpose: swap the `rotation` list to `(w, x, y, z)` and confirm `test_geometry_and_transform_round_trip` fails; remove the `view()` padding and confirm `test_glb_header_and_alignment` fails. Restore.
- [ ] **Step 8: Commit** `feat(gltf): write geometry, PBR materials and textures to .glb with a report`.

---

### Task 3: Camera, lights, environment and scene extras

**Files:**
- Modify: `src/manifoldx/gltf/writer.py`
- Test: `tests/test_gltf_scene.py`

**Interfaces:**
- Consumes: `_Writer`, `GlbBuilder` from Task 2.
- Produces: `build_glb(nodes, *, camera=None, lights=(), environment=None, scene_extras=None)` and the same keywords on `export_gltf`. `scenes[0].extras.manifoldx` holds `scene_extras` plus `environment = {"bufferView", "width", "height", "intensity", "layout"}`.

- [ ] **Step 1: Write the failing tests**

```python
import math

import numpy as np
import pytest

from manifoldx.camera import Camera
from manifoldx.gltf import Node, export_gltf
from manifoldx.ibl import EnvironmentMap
from manifoldx.resources import DirectionalLight, PointLight, SpotLight, StandardMaterial, cube

from gltf_helpers import load, view_bytes


def rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def test_camera_looks_at_its_target(tmp_path):
    cam = Camera(position=(0, 5, 10), target=(0, 0, 0), fov=50, near=0.2, far=900)
    export_gltf(tmp_path / "a.glb", [], camera=cam)
    g = load(tmp_path / "a.glb")
    node = next(n for n in g.nodes if n.camera is not None)
    p = g.cameras[node.camera].perspective
    assert p.yfov == pytest.approx(math.radians(50)) and (p.znear, p.zfar) == (pytest.approx(0.2), 900)
    assert node.translation == [0, 5, 10]
    forward = rotate(node.rotation, (0, 0, -1))
    np.testing.assert_allclose(forward, np.array([0, -5, -10]) / np.linalg.norm([0, 5, 10]), atol=1e-6)
    assert rotate(node.rotation, (0, 1, 0))[1] > 0  # not upside down


def test_lights(tmp_path):
    sun = DirectionalLight(color="#ff8000", intensity=2.4, direction=(0, -1, 0))
    spot = SpotLight(color="#ffffff", intensity=3, position=(1, 2, 3), direction=(1, 0, 0),
                     inner_angle=0.2, outer_angle=0.4, distance=20)
    point = PointLight(color="#ffffff", intensity=5, position=(4, 5, 6))
    export_gltf(tmp_path / "a.glb", [], lights=[sun, spot, point])
    g = load(tmp_path / "a.glb")
    assert "KHR_lights_punctual" in g.extensionsUsed
    lights = g.extensions["KHR_lights_punctual"]["lights"]
    assert [l["type"] for l in lights] == ["directional", "spot", "point"]
    assert lights[0]["color"] == pytest.approx([1.0, 128 / 255, 0.0])
    assert lights[0]["intensity"] == 2.4 and lights[0]["extras"]["manifoldx_intensity"] == 2.4
    assert lights[1]["spot"] == {"innerConeAngle": 0.2, "outerConeAngle": 0.4} and lights[1]["range"] == 20
    assert "range" not in lights[2]
    by_light = {n.extensions["KHR_lights_punctual"]["light"]: n for n in g.nodes if n.extensions}
    np.testing.assert_allclose(rotate(by_light[0].rotation, (0, 0, -1)), (0, -1, 0), atol=1e-6)
    np.testing.assert_allclose(rotate(by_light[1].rotation, (0, 0, -1)), (1, 0, 0), atol=1e-6)
    assert by_light[2].translation == [4, 5, 6]


def test_environment_and_extras(tmp_path):
    env = EnvironmentMap.from_sky(zenith=(0.4, 0.5, 0.7), horizon=(0.6, 0.6, 0.7), ground=(0.3, 0.3, 0.2))
    env.intensity = 0.35
    export_gltf(tmp_path / "a.glb", [Node(cube(1, 1, 1), StandardMaterial("#fff"))], environment=env,
                scene_extras={"fog": {"start": 10.0, "end": 20.0, "color": [0.1, 0.2, 0.3]}})
    g = load(tmp_path / "a.glb")
    extras = g.scenes[0].extras["manifoldx"]
    assert extras["fog"]["end"] == 20.0
    e = extras["environment"]
    assert (e["width"], e["height"], e["intensity"]) == (128, 64, 0.35)
    data = np.frombuffer(view_bytes(g, e["bufferView"]), np.float32).reshape(64, 128, 3)
    np.testing.assert_allclose(data, env.data)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_gltf_scene.py -q`. Expected: FAIL, unexpected keyword `camera`.

- [ ] **Step 3: Implement.** Add to `writer.py`:

```python
def _look_rotation(forward, up=(0.0, 1.0, 0.0)):
    """Quaternion (x, y, z, w) that turns local -Z to `forward` with local +Y towards `up`."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    if abs(f @ u) > 0.999:  # looking straight along up: any perpendicular up will do
        u = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    z = -f
    x = np.cross(u, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.column_stack([x, y, z])
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = (s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s)
    return [float(c) for c in q]


def _light_rgb(color):
    """Light colours as the renderer uploads them: hex over 255, no sRGB decode."""
    if isinstance(color, str):
        h = color.lstrip("#")
        return [int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4)]
    return [float(c) for c in color[:3]]
```

and these `_Writer` methods:

```python
    def camera(self, cam):
        index = self.b.add("cameras", {"type": "perspective", "perspective": {
            "yfov": float(np.radians(cam.fov)), "znear": float(cam.near), "zfar": float(cam.far)}})
        pos = [float(c) for c in cam.position]
        rot = _look_rotation(np.asarray(cam.target, float) - pos, cam.up)
        self._roots.append(self.b.add("nodes", {"name": "camera", "camera": index, "translation": pos,
                                                "rotation": rot}))

    def light(self, light):
        from manifoldx.resources import DirectionalLight, PointLight, SpotLight

        entry = {"color": _light_rgb(light.color), "intensity": float(light.intensity),
                 "extras": {"manifoldx_intensity": float(light.intensity)}}
        node = {"name": type(light).__name__}
        if isinstance(light, DirectionalLight):
            entry["type"] = "directional"
            node["rotation"] = _look_rotation(light.direction)
        elif isinstance(light, SpotLight):
            entry["type"] = "spot"
            entry["spot"] = {"innerConeAngle": float(light.inner_angle), "outerConeAngle": float(light.outer_angle)}
            node["translation"] = [float(c) for c in light.position]
            node["rotation"] = _look_rotation(light.direction)
        elif isinstance(light, PointLight):
            entry["type"] = "point"
            node["translation"] = [float(c) for c in light.position]
        else:
            self.report.add("light", type(light).__name__, "dropped", "no glTF light type")
            return
        if getattr(light, "distance", 0):
            entry["range"] = float(light.distance)
        self.b.use_extension("KHR_lights_punctual")
        ext = self.b.doc.setdefault("extensions", {}).setdefault("KHR_lights_punctual", {"lights": []})
        ext["lights"].append(entry)
        node["extensions"] = {"KHR_lights_punctual": {"light": len(ext["lights"]) - 1}}
        self._roots.append(self.b.add("nodes", node))

    def environment(self, env):
        data = np.ascontiguousarray(env.data, dtype=np.float32)
        return {"bufferView": self.b.view(data.tobytes()), "width": int(data.shape[1]),
                "height": int(data.shape[0]), "intensity": float(env.intensity),
                "layout": "equirectangular, linear RGB float32, row 0 at the zenith"}
```

`build_glb` and `export_gltf` gain the keywords:

```python
def build_glb(nodes, *, camera=None, lights=(), environment=None, scene_extras=None):
    w = _Writer()
    for i, node in enumerate(nodes):
        w.node(node, i)
    if camera is not None:
        w.camera(camera)
    for light in lights:
        w.light(light)
    extras = dict(scene_extras or {})
    if environment is not None:
        extras["environment"] = w.environment(environment)
    return w.finish(extras), w.report


def export_gltf(path, nodes, **scene):
    data, report = build_glb(nodes, **scene)
    Path(path).write_bytes(data)
    return report
```

and `finish(self, extras)` puts `{"manifoldx": extras}` in the scene's `extras` when `extras` is not empty.

- [ ] **Step 4: Run** `uv run pytest tests/test_gltf_scene.py tests/test_gltf_writer.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(gltf): camera, punctual lights, environment and scene extras`.

---

### Task 4: `modeling.Mesh.to_gltf`

**Files:**
- Modify: `src/manifoldx/modeling/mesh.py` (method on `Mesh`)
- Test: `tests/modeling/test_mesh_to_gltf.py` (create `tests/modeling/` if missing; else put it at `tests/test_mesh_to_gltf.py`)

**Interfaces:**
- Consumes: `export_gltf`, `Node`.
- Produces: `Mesh.to_gltf(path, material=None) -> ExportReport`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from manifoldx.modeling import Mesh
from manifoldx.resources import StandardMaterial

from gltf_helpers import accessor, load


def test_plain_mesh_gets_a_white_rough_material(tmp_path):
    report = Mesh.plane(width=2, depth=2, segments=2).to_gltf(tmp_path / "p.glb")
    g = load(tmp_path / "p.glb")
    pbr = g.materials[0].pbrMetallicRoughness
    assert pbr.baseColorFactor == [1, 1, 1, 1] and pbr.roughnessFactor == 1.0
    assert len(report) == 0


def test_coloured_mesh_keeps_its_colours(tmp_path):
    m = Mesh.plane(width=2, depth=2, segments=2)
    m = m.with_colors(np.full((len(m.positions), 3), 0.25, np.float32))
    m.to_gltf(tmp_path / "c.glb")
    g = load(tmp_path / "c.glb")
    attrs = g.meshes[0].primitives[0].attributes
    np.testing.assert_allclose(accessor(g, attrs.COLOR_0), 0.25)


def test_explicit_material(tmp_path):
    Mesh.plane(width=2, depth=2, segments=2).to_gltf(tmp_path / "m.glb", material=StandardMaterial("#000000", metallic=1))
    assert load(tmp_path / "m.glb").materials[0].pbrMetallicRoughness.metallicFactor == 1.0
```

(`gltf_helpers` is importable from `tests/` because pytest puts the test's rootdir-relative directory on `sys.path`; if the test lives in `tests/modeling/`, add `sys.path.insert(0, str(Path(__file__).parents[1]))` at its top.)

- [ ] **Step 2: Run** it. Expected: FAIL, `AttributeError: to_gltf`.
- [ ] **Step 3: Implement** on `Mesh`:

```python
    def to_gltf(self, path, material=None):
        """Write this mesh to a .glb file at the origin; return the export report.

        Without a material it gets a white rough StandardMaterial, which shows
        the vertex colours when the mesh has them.
        """
        from manifoldx.gltf import Node, export_gltf
        from manifoldx.resources import StandardMaterial

        if material is None:
            material = StandardMaterial("#ffffff", roughness=1.0, vertex_colors=self.colors is not None)
        return export_gltf(path, [Node(self, material)])
```

- [ ] **Step 4: Run** it. Expected: PASS.
- [ ] **Step 5: Commit** `feat(modeling): Mesh.to_gltf writes a mesh straight to .glb`.

---

### Task 5: `Engine.export_gltf`

**Files:**
- Create: `src/manifoldx/gltf/engine_export.py`
- Modify: `src/manifoldx/engine.py` (one method next to `render_frame`)
- Test: `tests/test_gltf_engine.py`

**Interfaces:**
- Consumes: `build_glb`, `Node`, `ExportReport`, `ExportIncomplete`.
- Produces: `Engine.export_gltf(path, *, names=None, extras=None, strict=False) -> ExportReport`; writes `path` and `path.with_suffix(".export-report.json")`; prints the report to stderr.

- [ ] **Step 1: Write the failing tests**

```python
import json

import manifoldx as mx
import numpy as np
import pytest
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gltf import ExportIncomplete
from manifoldx.gui import Panel, Text, ValueDisplay
from manifoldx.resources import BasicMaterial, DirectionalLight, StandardMaterial, cube

from gltf_helpers import load


def scene():
    e = mx.Engine("t", width=64, height=48)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=2.0, direction=(0, -1, 0)))
    e.enable_fog(10, 20)
    e.enable_shadows(resolution=1024, bias=0.002, pcf_radius=2)
    geo = cube(1, 1, 1)
    a = e.spawn(Mesh(geo), Material(StandardMaterial("#ffffff")), Transform(pos=(1, 2, 3)))
    b = e.spawn(Mesh(geo), Material(StandardMaterial("#ff0000")), Transform(pos=(4, 0, 0)))
    return e, a, b


def test_fresh_engine_reports_nothing(tmp_path):
    e = mx.Engine("t", width=64, height=48)
    report = e.export_gltf(tmp_path / "s.glb")
    assert len(report) == 0
    assert json.loads((tmp_path / "s.export-report.json").read_text()) == {"complete": True, "entries": []}


def test_entities_names_extras_and_scene_state(tmp_path):
    e, a, b = scene()
    e.export_gltf(tmp_path / "s.glb", names={a.index: "a"}, extras={a.index: {"pick": "p1"}})
    g = load(tmp_path / "s.glb")
    named = {n.name: n for n in g.nodes}
    assert named["a"].translation == [1, 2, 3] and named["a"].extras == {"pick": "p1"}
    assert named[f"entity_{b.index}"].translation == [4, 0, 0]
    assert len(g.meshes) == 2  # one geometry, two materials
    extras = g.scenes[0].extras["manifoldx"]
    assert extras["fog"] == {"start": 10.0, "end": 20.0, "color": list(e.fog_color)}
    assert extras["shadows"]["resolution"] == 1024 and extras["background"] == list(e.background_color)
    assert any(n.camera is not None for n in g.nodes)
    assert g.extensions["KHR_lights_punctual"]["lights"][0]["type"] == "directional"


def test_code_and_unsupported_things_are_reported(tmp_path):
    e, _, _ = scene()

    @e.system
    def spin(query: mx.Query[Transform], dt: float):
        pass

    @e.on("key_down")
    def keys(payload):
        pass

    e.gui.append(Panel(children=[Text("hi"), ValueDisplay(getter=lambda: "x")]))
    e.spawn(Mesh(cube(1, 1, 1)), Material(BasicMaterial("#00ff00")), Transform())
    report = e.export_gltf(tmp_path / "s.glb")
    kinds = sorted((r.kind, r.effect) for r in report)
    assert kinds == sorted([("system", "dropped"), ("handler", "dropped"), ("gui", "dropped"),
                            ("gui", "dropped"), ("material", "approximated")])
    system = next(r for r in report if r.kind == "system")
    assert system.name.endswith("spin") and system.where.endswith(f"test_gltf_engine.py:{spin.__code__.co_firstlineno}")
    assert any(r.name.startswith("ValueDisplay.getter") for r in report)


def test_strict_raises_and_writes_nothing(tmp_path):
    e, _, _ = scene()
    e.spawn(Mesh(cube(1, 1, 1)), Material(BasicMaterial("#00ff00")), Transform())
    with pytest.raises(ExportIncomplete) as info:
        e.export_gltf(tmp_path / "s.glb", strict=True)
    assert len(info.value.report) == 1
    assert not (tmp_path / "s.glb").exists() and not (tmp_path / "s.export-report.json").exists()


def test_startup_spawns_are_exported(tmp_path):
    from manifoldx.backends import get_offscreen_canvas

    try:
        get_offscreen_canvas(width=8, height=8)
    except Exception:
        pytest.skip("no offscreen wgpu backend")
    e = mx.Engine("t", width=64, height=48)

    @e.on("startup")
    def spawn(_payload):
        e.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial("#ffffff")), Transform())

    report = e.export_gltf(tmp_path / "s.glb")
    assert len(load(tmp_path / "s.glb").meshes) == 1 and len(report) == 0
```

(Check how existing GPU tests skip, e.g. `grep -n "get_offscreen_canvas" tests/*.py | head -3`, and copy that exact guard.)

- [ ] **Step 2: Run** `uv run pytest tests/test_gltf_engine.py -q`. Expected: FAIL, `AttributeError: export_gltf`.

- [ ] **Step 3: Implement `engine_export.py`.**

```python
"""Engine.export_gltf: the engine's scene to a .glb, and a report of everything else."""

import inspect
from pathlib import Path

import numpy as np

from manifoldx.gltf.report import ExportIncomplete, ExportReport
from manifoldx.gltf.writer import Node, build_glb


def _where(obj):
    obj = getattr(obj, "__func__", obj)
    try:
        return f"{inspect.getsourcefile(obj)}:{inspect.getsourcelines(obj)[1]}"
    except (OSError, TypeError):
        return None


def _label(obj):
    return getattr(obj, "__qualname__", None) or type(obj).__name__


def _internal(func):
    """manifoldx's own handlers (the GUI bridge) are not the app's code."""
    module = getattr(getattr(func, "__func__", func), "__module__", "") or ""
    return module == "manifoldx" or module.startswith("manifoldx.")


def _widgets(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _widgets(child)


def code_report(engine):
    r = ExportReport()
    for system in engine.systems._systems:
        r.add("system", _label(system.func), "dropped", "custom Python that runs every frame", _where(system.func))
    for event, handlers in engine._event_bus._handlers.items():
        if event == "startup":  # the export runs it: it is how most apps spawn
            continue
        for h in handlers:
            if not _internal(h.func):
                r.add("handler", f'on("{event}") {_label(h.func)}', "dropped", "event handlers are not exported",
                      _where(h.func))
    for panel in engine.gui:
        r.add("gui", type(panel).__name__, "dropped", "glTF has no GUI")
        for widget in _widgets(panel):
            for attr, value in vars(widget).items():
                if callable(value) and not isinstance(value, type):
                    r.add("gui", f"{type(widget).__name__}.{attr} = {_label(value)}", "dropped",
                          "Python callbacks cannot run outside manifoldx", _where(value))
    for cls in engine._compute_runner._registered:
        r.add("compute", _label(cls), "dropped", "GPU compute kernels are not exported", _where(cls))
    for _task in engine._pending_tasks:
        r.add("task", "background task", "dropped", "pending background work is not exported")
    return r


def scene_extras(engine):
    extras = {"background": [float(c) for c in engine.background_color[:3]]}
    if engine.fog_enabled:
        extras["fog"] = {"start": engine.fog_start, "end": engine.fog_end,
                         "color": [float(c) for c in engine.fog_color[:3]]}
    if engine._shadow_config is not None:
        extras["shadows"] = {k: list(v) if isinstance(v, tuple) else v for k, v in engine._shadow_config.items()}
    return extras


def nodes(engine, names, extras):
    comps = engine.store._components
    out = []
    for i in np.where(engine.store._alive)[0]:
        i = int(i)
        gid, mid = int(comps["Mesh"][i, 0]), int(comps["Material"][i, 0])
        if gid == 0 or mid == 0:
            continue
        t = comps["Transform"][i]
        out.append(Node(engine._geometry_registry.get(gid), engine._material_registry.get(mid),
                        pos=tuple(t[0:3]), rot=tuple(t[3:7]), scale=tuple(t[7:10]),
                        name=names.get(i, f"entity_{i}"), extras=extras.get(i)))
    return out


def export_engine(engine, path, *, names=None, extras=None, strict=False):
    if engine._device is None and engine._event_bus._handlers.get("startup"):
        engine._ensure_offscreen(1)  # startup handlers may need the device (load_texture)
    report = code_report(engine)
    lights = [light for light in (engine._sun, engine._spot) if light is not None] + list(engine._lights)
    data, written = build_glb(nodes(engine, names or {}, extras or {}), camera=engine.camera, lights=lights,
                              environment=engine._environment, scene_extras=scene_extras(engine))
    report.extend(written)
    report.print()
    if strict and len(report):
        raise ExportIncomplete(report)
    path = Path(path)
    path.write_bytes(data)
    report.write_json(path.with_suffix(".export-report.json"))
    return report
```

Engine method (in `engine.py`, after `render_frame`):

```python
    def export_gltf(self, path, *, names=None, extras=None, strict=False):
        """Write the scene to a glTF binary (.glb) and report what it left out.

        Meshes, materials, textures, the camera, the lights, and fog, shadow,
        background and environment settings (in the scene's extras) are
        written. Systems, event handlers, GUI, compute and materials without a
        glTF equivalent are listed in the returned report, which is also
        printed and saved next to the file. `names` and `extras` map entity
        indices to node names and JSON extras. With strict=True an incomplete
        export raises ExportIncomplete and writes nothing.
        """
        from manifoldx.gltf.engine_export import export_engine

        return export_engine(self, path, names=names, extras=extras, strict=strict)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_gltf_engine.py -q`. Expected: PASS. If `test_fresh_engine_reports_nothing` fails on a manifoldx-internal handler other than the GUI bridge, that handler's module is not `manifoldx.*`; print `report.format()` and fix `_internal`, not the test.
- [ ] **Step 5: Commit** `feat(engine): export_gltf writes the scene and reports what it left out`.

---

### Task 6: The `manifoldx export` command

**Files:**
- Create: `src/manifoldx/cli.py`, `tests/fixtures/export_app.py`
- Modify: `pyproject.toml` (`[project.scripts] manifoldx = "manifoldx.cli:main"`)
- Test: `tests/test_gltf_cli.py`

**Interfaces:**
- Consumes: `Engine.export_gltf`, `ExportIncomplete`.
- Produces: `main(argv=None) -> int`.

- [ ] **Step 1: Write the fixture and the failing tests.** Fixture `tests/fixtures/export_app.py`:

```python
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
```

Tests:

```python
from pathlib import Path

from manifoldx.cli import main

from gltf_helpers import load

APP = str(Path(__file__).parent / "fixtures" / "export_app.py")


def test_exports_instead_of_running(tmp_path):
    out = tmp_path / "s.glb"
    assert main(["export", APP, str(out)]) == 0
    assert load(out).nodes[0].translation == [0, 1, 0]


def test_strict_exit_code(tmp_path):
    out = tmp_path / "s.glb"
    assert main(["export", APP, str(out), "--strict", "--", "--basic"]) == 1
    assert not out.exists()
    assert main(["export", APP, str(out), "--", "--basic"]) == 0 and out.exists()


def test_app_that_never_runs(tmp_path):
    assert main(["export", APP, str(tmp_path / "s.glb"), "--", "--no-run"]) == 2


def test_module_form(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(APP).parent))
    out = tmp_path / "s.glb"
    assert main(["export", "-m", "export_app", str(out)]) == 0 and out.exists()


def test_engine_run_is_restored(tmp_path):
    import manifoldx as mx

    before = mx.Engine.run
    main(["export", APP, str(tmp_path / "s.glb")])
    assert mx.Engine.run is before
```

- [ ] **Step 2: Run** `uv run pytest tests/test_gltf_cli.py -q`. Expected: FAIL, no module `manifoldx.cli`.

- [ ] **Step 3: Implement `cli.py`.**

```python
"""The manifoldx command. `manifoldx export` runs an app and writes its scene to glTF instead of a window.

    manifoldx export app.py scene.glb [--strict] [-- <app args>]
    manifoldx export -m package.module scene.glb [--strict] [-- <app args>]
"""

import argparse
import runpy
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, app_args = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(prog="manifoldx")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="run an app and write its scene to a .glb instead of opening a window")
    ex.add_argument("-m", dest="module", help="run this module as __main__, like python -m")
    ex.add_argument("paths", nargs="+", metavar="[app.py] out.glb")
    ex.add_argument("--strict", action="store_true", help="exit with code 1 and write nothing if anything is left out")
    args = parser.parse_args(argv)
    if len(args.paths) != (1 if args.module else 2):
        parser.error("give app.py and out.glb, or -m module and out.glb")
    return export(args.module, None if args.module else args.paths[0], args.paths[-1], args.strict, app_args)


def export(module, script, output, strict, app_args):
    from manifoldx.engine import Engine
    from manifoldx.gltf import ExportIncomplete

    exported = []

    def run(engine):
        exported.append(engine.export_gltf(output, strict=strict))

    original, saved_argv = Engine.run, sys.argv
    Engine.run = run
    sys.argv = [module or script, *app_args]
    try:
        if module:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(script, run_name="__main__")
    except ExportIncomplete:
        return 1
    finally:
        Engine.run, sys.argv = original, saved_argv
    if not exported:
        print(f"manifoldx export: {module or script} finished without calling engine.run(); nothing written",
              file=sys.stderr)
        return 2
    return 0
```

Add to `pyproject.toml`:

```toml
[project.scripts]
manifoldx = "manifoldx.cli:main"
```

- [ ] **Step 4: Run** the tests, then `uv sync --all-extras && uv run manifoldx export tests/fixtures/export_app.py /tmp/fx.glb; echo $?`. Expected: PASS, and `0` with the file written.
- [ ] **Step 5: Commit** `feat(cli): manifoldx export runs an app and writes its scene to glTF`.

---

### Task 7: Conformance, the example, and the changelog

**Files:**
- Create: `examples/gltf_export.py`, `tests/test_gltf_conformance.py`
- Modify: `CHANGELOG.md` (`[Unreleased]`), `examples/README.md` (one line if it lists examples)

- [ ] **Step 1: Write the example.**

```python
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
```

- [ ] **Step 2: Write the conformance tests.**

```python
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
VALIDATE = ("const v=require('gltf-validator');const fs=require('fs');"
            "v.validateBytes(new Uint8Array(fs.readFileSync(process.argv[1])))"
            ".then(r=>console.log(JSON.stringify(r.issues)))")


def sample(tmp_path):
    """Every writer feature in one file."""
    from PIL import Image

    from manifoldx.camera import Camera
    from manifoldx.gltf import Node, export_gltf
    from manifoldx.ibl import EnvironmentMap
    from manifoldx.resources import (BasicMaterial, DirectionalLight, FlatMaterial, PointLight, SpotLight,
                                     StandardMaterial, cube)
    from manifoldx.textures import TextureHandle

    img = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (10, 200, 30)).save(img)
    tex = TextureHandle(id=1, texture=None, view=None, sampler=None, size=(8, 8), source=img)
    geo = cube(1, 1, 1)
    geo = geo | {"uvs": geo["positions"][:, :2].copy()}
    out = tmp_path / "all.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff", albedo_map=tex), name="tex", extras={"pick": "a"}),
                      Node(geo, FlatMaterial("#ff0000"), pos=(2, 0, 0)),
                      Node(geo, BasicMaterial((1, 1, 1, 0.5)), pos=(4, 0, 0))],
                camera=Camera(position=(0, 3, 8)),
                lights=[DirectionalLight("#ffffff", 2.0, (0, -1, -1)), PointLight("#ffffff", 3, (0, 4, 0)),
                        SpotLight("#ffffff", 3, (0, 4, 4), (0, -1, -1), 0.2, 0.4, distance=30)],
                environment=EnvironmentMap.from_sky((0.4, 0.5, 0.7), (0.6, 0.6, 0.7), (0.3, 0.3, 0.2)),
                scene_extras={"fog": {"start": 1.0, "end": 9.0}})
    return out


@pytest.mark.skipif(shutil.which("npm") is None or shutil.which("node") is None, reason="needs node and npm")
def test_khronos_validator_finds_no_errors(tmp_path):
    out = sample(tmp_path)
    tools = tmp_path / "validator"
    install = subprocess.run(["npm", "install", "--silent", "--prefix", str(tools), "gltf-validator"],
                             capture_output=True, text=True, timeout=300)
    if install.returncode != 0:
        pytest.skip(f"npm install gltf-validator failed: {install.stderr[-300:]}")
    res = subprocess.run(["node", "-e", VALIDATE, str(out)], cwd=tools, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    issues = json.loads(res.stdout)
    errors = [m for m in issues["messages"] if m["severity"] == 0]
    assert issues["numErrors"] == 0, errors


@pytest.mark.skipif(shutil.which("blender") is None, reason="needs blender")
def test_blender_imports_the_example(tmp_path):
    out = tmp_path / "terrain.glb"
    subprocess.run(["uv", "run", "python", str(ROOT / "examples" / "gltf_export.py"), str(out)], check=True,
                   cwd=ROOT, capture_output=True, timeout=300)
    script = ("import bpy;bpy.ops.wm.read_factory_settings(use_empty=True);"
              f"bpy.ops.import_scene.gltf(filepath={str(out)!r});"
              "print('COUNTS', len(bpy.data.meshes), len(bpy.data.materials))")
    res = subprocess.run(["blender", "-b", "--factory-startup", "--python-expr", script],
                         capture_output=True, text=True, timeout=300)
    line = next(l for l in res.stdout.splitlines() if l.startswith("COUNTS"))
    assert line == "COUNTS 1 1", res.stdout[-2000:]
```

- [ ] **Step 3: Run** `uv run pytest tests/test_gltf_conformance.py -q -rs`. Expected on zion: 2 passed. Break the writer once (write `"mode": 9`, an invalid primitive mode) and confirm the validator test fails; restore.
- [ ] **Step 4: Changelog** under `[Unreleased]`, `### Added`:

```markdown
- glTF export. `manifoldx.gltf.export_gltf(path, nodes)`, `modeling.Mesh.to_gltf(path)` and `Engine.export_gltf(path)` write `.glb` files that Blender, Godot, Unity and three.js open: geometry, PBR materials, embedded textures, punctual lights, the camera, and fog, shadow, background and environment settings in the scene's extras. Everything that cannot go into glTF (systems, event handlers, GUI, compute, volumes, point clouds, labels, materials without an equivalent) is reported by name, printed and saved as `<name>.export-report.json`; `strict=True` raises `ExportIncomplete` instead. `manifoldx export app.py out.glb` exports any app that ends in `engine.run()` without changing it. Example: `examples/gltf_export.py`.
- `TextureHandle.source`: the image file `load_texture` read.
```

- [ ] **Step 5: Run** `make lint` and `make test` (the first full run of the branch, since it touches `engine.py` and `textures.py`), reading the exit code directly. Expected: both pass.
- [ ] **Step 6: Commit** `docs(gltf): example, conformance tests and changelog`, then open the PR (`Fixes #23`) with the validator and Blender results in the body.
