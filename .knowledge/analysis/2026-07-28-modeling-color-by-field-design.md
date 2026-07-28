# Modeling Color-by-Field v1 — Design (SP-2)

**Date:** 2026-07-28
**Scope:** Per-vertex color for `manifoldx.modeling` meshes, plus a composable `Gradient` primitive and `Mesh.color_by(field, gradient)`, so a developer shades their own geometry by any field (height, slope, noise, distance…). The one renderer touch in the PCG primitive series. **No opinionated biome/terrain coloring — just the tools.**

## Motivation

SP-1 gives composable *shape* (fields → `displace`); meshes still render as flat clay. Coloring a terrain by height (water→grass→rock→snow) or a tree by height (bark→leaf) needs per-vertex color reaching the fragment shader. This slice adds that, keeping the composable philosophy: the developer supplies a `Field` (what varies) and a `Gradient` (how it maps to color); nothing prescribes what the color *means*.

## Design (additive, mirrors the existing `"textured"` subtype)

### 1. `Gradient` primitive — `src/manifoldx/modeling/gradient.py`

```python
Gradient(stops)   # stops: [(pos: float, color), ...], color a "#rrggbb" hex or (r,g,b) in [0,1]
gradient(values)  # (K,) -> (K, 3) float32, piecewise-linear between sorted stops, clamped at the ends
```

Pure numpy, no GPU. The developer composes their own ramp (`Gradient([(0,"#20407a"),(0.4,"#4a7a3a"),(0.7,"#7a6a55"),(1,"#ffffff")])`).

### 2. `Mesh` per-vertex color (host side)

- `Mesh.colors: (N,3) float32 | None` — new frozen-dataclass field (alongside `normals`, `uvs`).
- `Mesh.with_colors(colors) -> Mesh`.
- `Mesh.color_by(field, gradient) -> Mesh` — `colors = gradient(field(positions))`; returns a colored copy.
- `to_geometry()` includes `"colors"` (N,3 float32) when present.
- `with_positions`/deformers carry `colors` through unchanged (per-vertex, topology-invariant); topology ops (`subdivide`/`extrude`/`decimate`/booleans) drop colors (they change vertex sets) — colors are meant to be applied *after* shaping, which is the natural pipeline order anyway.

### 3. GPU path — a `"vcolor"` StandardMaterial subtype

- `GeometryRegistry.create_buffers`: new branch when the geometry dict has `"colors"` (and normals, no uvs) → interleave `[pos(3), normal(3), color(3)]`, `stride = 9*4 = 36`, `buffers["has_colors"] = True`.
- `StandardMaterial(vertex_colors=True)` → `pipeline_subtype == "vcolor"`. (Mutually exclusive with `albedo_map`/`"textured"` in v1.)
- New WGSL variant `_STANDARDMATERIAL_VCOLOR_SHADER`: `VertexInput` gains `@location(2) color: vec3<f32>`, passed through to the fragment stage, where `albedo = in.color * material.albedo` (material color acts as a tint; default white tint = pure vertex color). Full PBR/lighting/shadows otherwise unchanged.
- Renderer mesh pass: `_compile(vertex_colors=True)` selects the variant; the vertex-attribute list appends `{format: float32x3, offset: 24, shader_location: 2}` when `material_subtype == "vcolor"` (exactly parallel to the `"textured"` uv attribute).

All changes are additive behind the new subtype; the scalar and textured paths are untouched (low regression risk).

### 4. `color_by` ergonomics

```python
colored = terrain.color_by(
    fields.coord("y").remap(0.0, 4.0, 0.0, 1.0),      # normalize height to [0,1]
    Gradient([(0,"#20407a"),(0.15,"#c2b280"),(0.4,"#4a7a3a"),(0.7,"#7a6a55"),(0.95,"#ffffff")]),
)
engine.spawn(Mesh(colored.to_geometry()), Material(StandardMaterial(color="#ffffff", vertex_colors=True)), ...)
```

## Testing

- **Gradient:** exact color at each stop, linear midpoint between two stops, clamp below first / above last stop, output shape `(K,3)` float32.
- **Mesh color (host):** `with_colors`/`color_by` set `colors` `(N,3)`; `color_by(constant_field, grad)` → uniform color; `color_by(coord("y"), grad)` varies with height; `to_geometry()` carries `"colors"`; a deformer preserves colors; `to_geometry` still valid without colors (back-compat).
- **GPU (gated on `get_offscreen_canvas`):** a vcolor mesh renders without error; interleave stride is 36. Plus a visual render check of the colored-terrain demo.
- **Back-compat:** existing mesh/PBR/textured tests unchanged (new subtype only).
- **Demo:** `examples/modeling_colored_terrain.py` — the SP-1 terrain, `color_by(height, gradient)`, lit + shadowed, slow camera orbit; rendered to MP4.

## Non-goals (this slice)

- Slope/normal-based coloring helper (developer composes it: a field from the mesh's own normals is a later convenience; for v1 color by any spatial field).
- Combining vertex color *and* an albedo texture (separate subtypes for now).
- Vertex alpha / transparency.
- Carrying colors through topology/boolean ops (apply color last).
- A named-colormap library — `Gradient` + hex stops is the composable primitive; presets are the developer's to define.
