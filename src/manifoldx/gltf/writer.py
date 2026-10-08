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
