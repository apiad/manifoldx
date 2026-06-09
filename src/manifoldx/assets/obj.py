"""Minimal Wavefront OBJ parser.

Supports the four face-line forms (1-indexed):
    f v ...
    f v/vt ...
    f v/vt/vn ...
    f v//vn ...

All face lines in a file must use the same form. Polygon faces with >3
vertices are fan-triangulated. Materials (`mtllib`, `usemtl`) and
grouping directives (`o`, `g`, `s`) are silently ignored — v1 carries
material info through Python kwargs, not MTL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ObjParseError(ValueError):
    """The OBJ file is malformed or uses a feature v1 doesn't support."""


def load_obj(path: str | Path) -> dict:
    """Parse a Wavefront .obj file into a manifoldx geometry dict.

    Returns:
        {"name": str,
         "positions": (N, 3) float32,
         "normals":   (N, 3) float32,   # present if file has normals
         "uvs":       (N, 2) float32,   # present if file has UVs
         "indices":   (M,)   uint32}
    """
    p = Path(path)
    text = p.read_text()

    raw_positions: list[list[float]] = []
    raw_normals: list[list[float]] = []
    raw_uvs: list[list[float]] = []
    face_triples: list[tuple] = []
    face_form: Optional[str] = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, *rest = line.split()
        if head == "v":
            raw_positions.append([float(x) for x in rest[:3]])
        elif head == "vn":
            raw_normals.append([float(x) for x in rest[:3]])
        elif head == "vt":
            raw_uvs.append([float(x) for x in rest[:2]])
        elif head == "f":
            parsed, this_form = _parse_face(lineno, rest)
            if face_form is None:
                face_form = this_form
            elif face_form != this_form:
                raise ObjParseError(
                    f"line {lineno}: face-line form changed from "
                    f"'{face_form}' to '{this_form}'; pick one"
                )
            # Fan-triangulate.
            for i in range(1, len(parsed) - 1):
                face_triples.append(parsed[0])
                face_triples.append(parsed[i])
                face_triples.append(parsed[i + 1])
        # Silently ignored: o, g, s, mtllib, usemtl.

    return _build_geometry(p.stem, raw_positions, raw_normals, raw_uvs,
                           face_triples, face_form)


def _parse_face(lineno: int, tokens: list[str]) -> tuple[list[tuple], str]:
    """Parse one face line's tokens into a list of (pi, ti|None, ni|None)
    triples and detect the face-line form."""
    if len(tokens) < 3:
        raise ObjParseError(
            f"line {lineno}: face needs at least 3 vertices, got {len(tokens)}"
        )

    parsed = []
    forms = set()
    for tok in tokens:
        if "//" in tok:
            pi_s, ni_s = tok.split("//")
            pi = _idx(lineno, pi_s)
            ni = _idx(lineno, ni_s)
            parsed.append((pi, None, ni))
            forms.add("v//vn")
        elif "/" in tok:
            parts = tok.split("/")
            if len(parts) == 2:
                pi = _idx(lineno, parts[0])
                ti = _idx(lineno, parts[1])
                parsed.append((pi, ti, None))
                forms.add("v/vt")
            elif len(parts) == 3:
                pi = _idx(lineno, parts[0])
                ti = _idx(lineno, parts[1])
                ni = _idx(lineno, parts[2])
                parsed.append((pi, ti, ni))
                forms.add("v/vt/vn")
            else:
                raise ObjParseError(
                    f"line {lineno}: malformed face token '{tok}'"
                )
        else:
            pi = _idx(lineno, tok)
            parsed.append((pi, None, None))
            forms.add("v")

    if len(forms) > 1:
        raise ObjParseError(
            f"line {lineno}: mixed face-line forms within a single face: "
            f"{sorted(forms)}"
        )
    (this_form,) = forms
    return parsed, this_form


def _idx(lineno: int, s: str) -> int:
    """Parse a 1-indexed OBJ index. Negative (relative) indices unsupported."""
    n = int(s)
    if n < 0:
        raise ObjParseError(
            f"line {lineno}: negative face indices not supported in v1; "
            f"re-export with absolute indices"
        )
    return n - 1


def _build_geometry(name, raw_positions, raw_normals, raw_uvs,
                    face_triples, face_form):
    has_uv = face_form in ("v/vt", "v/vt/vn")
    has_normal = face_form in ("v/vt/vn", "v//vn")

    dedup: dict[tuple, int] = {}
    positions_out: list[list[float]] = []
    normals_out: list[list[float]] = []
    uvs_out: list[list[float]] = []
    indices_out: list[int] = []

    for (pi, ti, ni) in face_triples:
        key = (pi, ti, ni)
        if key in dedup:
            indices_out.append(dedup[key])
            continue
        vi = len(positions_out)
        dedup[key] = vi
        positions_out.append(raw_positions[pi])
        if has_normal and ni is not None:
            normals_out.append(raw_normals[ni])
        if has_uv and ti is not None:
            uvs_out.append(raw_uvs[ti])
        indices_out.append(vi)

    geo = {
        "name": name,
        "positions": np.asarray(positions_out, dtype=np.float32),
        "indices": np.asarray(indices_out, dtype=np.uint32),
    }
    if has_normal:
        geo["normals"] = np.asarray(normals_out, dtype=np.float32)
    if has_uv:
        geo["uvs"] = np.asarray(uvs_out, dtype=np.float32)
    return geo
