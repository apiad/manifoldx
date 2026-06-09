# Textured PBR v1 — Design

**Date:** 2026-06-09
**Scope:** First slice of texture support in the core mesh-PBR path. Albedo map only. Loader for OBJ meshes. Demo: a textured Utah teapot rendered with the existing `StandardMaterial` PBR shader.

## Motivation

`StandardMaterial` is already a full GGX-PBR shader, but every material parameter is a scalar uniform — there's no way to vary `albedo` across a mesh's surface. The engine also has no way to load mesh geometry from a file; only `cube()`, `sphere()`, and `plane()` exist as procedural built-ins.

This plan adds the minimum end-to-end path to render an arbitrary `.obj` mesh with a 2D albedo texture sampled in the fragment shader. Once that path lights up, follow-up plans can layer normal / metallic-roughness / AO maps onto the same `StandardMaterial` class.

Non-goals for v1 (each its own follow-up):
- Normal maps (TBN basis, tangent attribute, MikkT).
- Metallic-roughness maps (glTF G/B-channel convention).
- AO maps.
- Environment / IBL.
- Mipmaps.
- HDR / EXR textures.
- Texture hot-reload.

## Architecture

Three new units; one extended unit; one demo + asset.

### 1. `manifoldx.assets.obj` — OBJ parser

Pure-Python, no new dependency. Public surface:

```python
def load_obj(path: str | Path) -> dict:
    """Parse a Wavefront .obj file into a manifoldx geometry dict.

    Returns:
        {"name": str,
         "positions": (N, 3) float32,
         "normals":   (N, 3) float32,
         "uvs":       (N, 2) float32,
         "indices":   (M,)   uint32}

    Where N is the number of unique (pos, normal, uv) tuples
    after de-duplication and M is divisible by 3.
    """
```

Behavior:
- Handles `v`, `vn`, `vt`, and the four OBJ face-line forms (1-indexed): `f v ...`, `f v/vt ...`, `f v/vt/vn ...`, `f v//vn ...`.
- All face lines in a single file must use the same form. Mixed forms within one file → `ObjParseError`.
- If face lines use the `v/vt` or `v/vt/vn` form, the returned dict includes a `"uvs"` key. If face lines use the `v` or `v//vn` form, the returned dict omits `"uvs"`. The textured `StandardMaterial` path requires a geometry with `"uvs"`; mismatch is caught at mesh-pass pipeline-cache time (see Error handling).
- De-duplicates the (pos, normal, uv) tuples so each unique combination becomes one GPU vertex. OBJ stores positions, normals, and UVs as parallel index streams; the GPU vertex buffer needs one stream.
- Fan-triangulates polygon faces with >3 vertices.
- Silently ignores `o`, `g`, `s`, `mtllib`, `usemtl` directives — v1 carries material info through Python kwargs, not MTL.
- Raises `ObjParseError(message_with_line_number)` for negative (relative) indices, malformed face lines, and mixed face-line forms.
- Returns a dict that is shape-compatible with `cube()` / `sphere()` output, with `"uvs"` present only when the source file carries UVs.

Module sits at `src/manifoldx/assets/obj.py` with a small `assets/__init__.py` that re-exports `load_obj`. A new top-level export `manifoldx.load_obj` mirrors `manifoldx.cube`.

### 2. `manifoldx.textures` — texture loading + registry

New module. Public surface:

```python
def load_texture(engine: Engine, path: str | Path) -> TextureHandle:
    """Decode an image file and upload it to the GPU.

    Returns a small handle the engine owns. Pass the handle to
    StandardMaterial(albedo_map=...).
    """
```

Internals:
- Decodes via Pillow (`PIL.Image.open(path).convert("RGBA")` → `(H, W, 4) uint8`).
- Pillow is already a `[viz]` optional extra; v1 promotes it to a required dependency (it's small, transitive deps are minimal, and it's needed for any texture work going forward).
- Creates a `wgpu.Texture` with format `Rgba8UnormSrgb` (so sampling auto-decodes sRGB to linear in the shader), `usage=TEXTURE_BINDING|COPY_DST`, mip level count 1.
- Creates a default `wgpu.Sampler` with `mag_filter=linear`, `min_filter=linear`, `address_mode_u/v=repeat`.
- Returns a `TextureHandle` dataclass holding the texture view + sampler + a stable integer id.

Lifecycle:
- A new `TextureRegistry` on the engine (mirrors `VolumeRegistry`) owns the strong references. v1 has no `unload` and no eviction. Textures live for the engine lifetime.
- Two materials sharing the same `TextureHandle` share the underlying GPU resource.

Errors:
- Missing file → `FileNotFoundError` (propagated).
- Image larger than `device.limits.max_texture_dimension_2d` → `TextureSizeError(image_dims, device_limit)`.
- Non-RGBA modes (RGB, L, P) → silently converted to RGBA.

### 3. Extended `StandardMaterial`

One new optional kwarg:

```python
StandardMaterial(color, roughness=0.5, metallic=0.0, ao=1.0,
                 albedo_map: TextureHandle | None = None)
```

When `albedo_map` is set:
- `pipeline_subtype` returns `"textured"` (instead of `None`), so the pipeline cache builds a distinct entry for the textured variant.
- `_compile()` emits a shader variant that adds two extra bind-group entries — `@binding(4) var albedo_sampler: sampler;` and `@binding(5) var albedo_tex: texture_2d<f32>;` — adds `@location(2) uv: vec2<f32>;` to the vertex input, passes UV through to the fragment shader, and replaces the fragment-shader line `let albedo = material.albedo;` with `let albedo = textureSample(albedo_tex, albedo_sampler, in.uv).rgb;`.
- The scalar `material.albedo` field stays in the uniform struct. It is unused in the textured variant; keeping the layout identical means `uniform_type()` and `get_data()` have one path, not two. (When we later add an `albedo_tint` multiplier, the field is already there.)
- A new `Material` method `get_texture_bindings(self) -> dict[int, TextureHandle]` returns `{4: handle}` for textured variants and `{}` by default. The mesh-pass bind-group builder reads this to attach texture resources.

`StandardMaterial` is the *only* class touched. `BasicMaterial`, `PhongMaterial`, `Transform`, `Mesh`, `Material` ECS components are unchanged.

### 4. Plumbing changes (small, surgical)

**`GeometryRegistry.create_buffers`** (in `src/manifoldx/resources.py`):
- When the geometry dict has `"uvs"`, build the interleaved vertex buffer as `[pos.xyz, normal.xyz, uv.xy]` (stride 32 B, 8 floats). Record `has_uvs=True` and `stride=32` in the buffer record.
- When the geometry dict has no `"uvs"`, behavior is unchanged: stride 24 B, layout `[pos, normal]`.
- The pipeline keyed on a textured material gets a vertex layout with three attributes; the pipeline keyed on a scalar material gets the existing two-attribute layout. Both layouts can target a stride-32 buffer (extra bytes are ignored) — but in v1 we keep textured-geom = stride-32 buffer, scalar-geom = stride-24 buffer, no cross-mixing.

**Mesh pass bind-group builder** (in `src/manifoldx/render/passes/mesh.py`):
- After the existing globals / transforms / material / lights entries, call `mat_obj.get_texture_bindings()`. For each `(binding, handle)` entry, append `{binding: N, resource: handle.sampler}` and `{binding: N+1, resource: handle.texture_view}` to `bind_group_entries`.
- Replace the existing `needs_lights = "@binding(3)" in type(mat_obj)._compile()` string-search with a cleaner predicate (`mat_obj.uses_lights()`) on `Material`. Default `True` for materials with the lights binding, override per-material. This is a small surgical cleanup; not a refactor.

**Pipeline cache** (in `src/manifoldx/renderer.py`'s `_get_or_create_pipeline`):
- Cache key already includes `pipeline_subtype` per AGENTS.md and the viz precedent. Verify against the source during implementation; if missing, add it as part of this plan (one-line fix).
- Vertex layout for the textured subtype adds `@location(2) uv: vec2<f32>` at offset 24, stride 32.
- Bind-group layout for the textured subtype adds entries 4 (sampler) and 5 (texture, `float`, `d2`, non-multisampled).

### 5. Demo — `examples/teapot_demo.py`

- Loads `examples/assets/teapot/teapot.obj`.
- Loads `examples/assets/teapot/teapot_albedo.png` via `load_texture`.
- Constructs `StandardMaterial(color=(1,1,1), roughness=0.4, metallic=0.0, albedo_map=albedo)`.
- Spawns the entity, configures the existing 4-point-light rig (reuse the layout from `pbr_demo.py`), sets a slow camera orbit via the input/event layer.
- Supports the standard `--render --duration N --fps 30 --output path.mp4` flags like other examples for offline smoke renders.

### 6. Asset — `examples/assets/teapot/`

- `teapot.obj` — the classic Newell Utah teapot. Public domain. The vintage distributions of the Newell teapot ship *without* texture coordinates; for v1 we use a version that has UVs (the McGuire computer-graphics archive ships one with auto-generated UVs from Blender, ~100 KB tessellated). Checked in. The asset README documents the provenance.
- `teapot_albedo.png` — generated by a small `scripts/gen_teapot_albedo.py`. Checked in (regenerable). A "blue-and-white china" style procedural pattern: cobalt-blue floral motifs on a soft cream background, ~512×512, ~50 KB. Generator script is also checked in so the asset is reproducible.
- `README.md` — provenance for both files (license, source URL, regen instructions for the PNG).

**Asset-prep contingency.** If a suitable UV-mapped teapot OBJ under a clearly compatible license isn't readily available, the fallback is a small `scripts/prep_teapot_uvs.py` that takes the bare-Newell OBJ and assigns spherical-projection UVs. The output is committed as `teapot.obj`; the script is committed alongside for reproducibility. Either path produces the same `teapot.obj` artifact in the assets directory.

Both assets sit under `examples/assets/teapot/`. This is the first bundled binary asset in the repo; the precedent is fine (other PBR engines bundle a small teapot or sphere asset for the same reason). It will be noted in CHANGELOG.

## Data flow

**Init (one-shot, before the frame loop):**

```
teapot_geo = mx.load_obj("examples/assets/teapot/teapot.obj")
  → {positions: (N,3), normals: (N,3), uvs: (N,2), indices: (M,), name: "teapot"}

albedo = mx.load_texture(engine, "examples/assets/teapot/teapot_albedo.png")
  → Pillow decodes → (H,W,4) uint8 → device.create_texture(Rgba8UnormSrgb)
    → queue.write_texture(...) → TextureHandle(view, sampler, id=7)

mat = mx.StandardMaterial(color=(1,1,1), roughness=0.4, metallic=0.0,
                          albedo_map=albedo)

engine.spawn(Transform(...), Mesh(teapot_geo), Material(mat))
  → GeometryRegistry.register(teapot_geo) lazily allocates the stride-32
    interleaved vertex buffer + index buffer on first frame.
```

**Pipeline-cache lookup (lazy, on first frame this batch appears):**

- `_get_or_create_pipeline` keys on `(geom_id, type(mat_obj), mat_obj.pipeline_subtype)`.
- For the textured teapot batch the key is `(<teapot_id>, StandardMaterial, "textured")`.
- Cache miss → compile the textured WGSL variant → build vertex layout `(pos@0, normal@12, uv@24, stride=32)` → build bind-group layout with entries 0–5 → cache.

**Per-frame mesh pass (existing path, two additive changes):**

```
for (geom_id, mat_type) batch:
  upload material uniforms                       # unchanged
  bind_group_entries = [globals, transforms,
                        material_uniform, lights] # unchanged
  texture_bindings = mat_obj.get_texture_bindings()   # NEW
  for (binding, handle) in texture_bindings:          # NEW
      bind_group_entries += [{binding, sampler},
                             {binding+1, view}]
  build bind group from cached layout
  set_pipeline, set_vertex_buffer, set_index_buffer
  draw_indexed(index_count, instance_count, ..., first_instance)
```

The "scalar `StandardMaterial`" path is byte-identical to today — `get_texture_bindings()` returns `{}` and the loop body is a no-op.

## Error handling

Validation happens at the boundaries; internal calls trust their inputs.

**`load_obj`:**
- Missing file → `FileNotFoundError` (propagated).
- Malformed `f` line → `ObjParseError("line N: malformed face line: ...")`.
- Mixed face-line forms in one file (e.g. some `f v/vt/vn`, some `f v//vn`) → `ObjParseError("line N: face-line form changed from '{prior}' to '{this}'; pick one")`.
- Polygon face with >3 verts → fan-triangulate silently.
- Negative (relative) face indices → `ObjParseError("line N: negative face indices not supported in v1; re-export with absolute indices")`.
- File whose face lines have no UV component (`v` or `v//vn` form) → load succeeds, returned dict has no `"uvs"` key. Mismatch with a textured material is caught downstream by `MaterialGeometryMismatchError`.
- `o` / `g` / `s` / `mtllib` / `usemtl` → silently ignored.

**`load_texture`:**
- Missing file → `FileNotFoundError`.
- Pillow `UnidentifiedImageError` → propagated as-is.
- Non-RGBA mode (RGB, L, P, RGBA-with-alpha-1.0) → converted to RGBA via `img.convert("RGBA")`. No error.
- Image larger than `device.limits.max_texture_dimension_2d` → `TextureSizeError(image=(w,h), device_max=N)`.

**`StandardMaterial`:**
- `albedo_map=` passed a non-`TextureHandle` (e.g. a path string) → `TypeError("albedo_map expects a TextureHandle from load_texture(...); got {type}. Did you forget to call load_texture(engine, path) first?")`.

**Mesh-pass runtime:**
- Textured `StandardMaterial` attached to a geometry without `"uvs"` → pipeline-cache miss raises `MaterialGeometryMismatchError("StandardMaterial(albedo_map=...) requires geometry with UVs; geometry '{geom.name}' has none")`. Surfaces on the first frame the entity appears, not at construction time — but loud, with a specific fix-it.

**Deliberately not validated in v1:**
- UV ranges (out-of-[0,1] UVs are fine; the sampler `repeat` wrap mode handles them).
- sRGB vs linear PNGs — v1 assumes sRGB-encoded source, decoded as `Rgba8UnormSrgb` so the GPU auto-converts to linear on sample. No user-facing knob; documented in the demo comments.
- Mesh winding order — wrong winding makes faces vanish under the existing CW-cull policy. Don't auto-detect; document.

## Testing

Following the repo convention: TDD plan in `.knowledge/plans/`, GPU tests gated on `get_offscreen_canvas`.

**Unit tests (no GPU):**

- `tests/test_obj_loader.py`
  - Inline single-triangle OBJ string → assert array shapes + index dedupe.
  - Inline quad face → assert fan-triangulation produces 2 triangles, 6 indices, 4 vertices.
  - Malformed `f` line → assert `ObjParseError` with line number in message.
  - Negative-index `f` line → assert `ObjParseError`.
  - File with mixed face-line forms (e.g. `f v/vt/vn` then `f v//vn`) → assert `ObjParseError`.
  - File with only `f v//vn` faces → assert load succeeds, returned dict has no `"uvs"` key.
  - `o` / `g` / `s` / `mtllib` / `usemtl` lines → assert silently ignored, parse succeeds.
  - Real `examples/assets/teapot/teapot.obj` → assert loads without error, produces non-empty arrays of expected shapes (positions / normals / uvs all aligned to N, indices divisible by 3).

- `tests/test_textures.py`
  - `StandardMaterial(albedo_map="path.png")` → assert `TypeError` with the fix-it message.
  - `load_texture` with a tiny PNG fixture → assert returns `TextureHandle` (GPU-upload step gated on device availability).
  - Oversized image → assert `TextureSizeError` mentioning both the image dims and the device limit.

**Integration tests (GPU, gated on `get_offscreen_canvas`):**

- `tests/test_textured_material.py`
  - Build a 2×2 PNG with four distinct corner colors → `load_texture` → spawn a sphere with `StandardMaterial(color=(1,1,1), albedo_map=tex)`. Render one frame. Assert the center pixel color falls within the expected range for a UV near (0.5, 0.5). Allow a generous tolerance — this is a "the texture path lit up" test, not a numerical-correctness test.
  - Spawn one scalar `StandardMaterial` cube and one textured `StandardMaterial` sphere in the same scene. Assert both render without error (no pipeline collision, no bind-group mismatch).
  - Spawn `cube()` (no UVs) with a textured material → assert `MaterialGeometryMismatchError` on first frame.

**Smoke / visual regression (manual):**

- `make test` runs the suite headless on the offscreen wgpu backend.
- `examples/teapot_demo.py --render --duration 2 --fps 30 --output /tmp/teapot.mp4` is the human-eyeball test. Not asserted in CI. Reviewer plays the MP4, checks the teapot looks like a teapot with china-pattern albedo and PBR specular highlights.
- No golden-image diffing — that's the Sci-viz Plan 5 work, separate.

**Deliberately not tested in v1:**
- Mipmap generation (mip-level count is 1).
- HDR / EXR textures.
- Texture hot-reload.
- Multiple texture sets per material.

## File-level change summary

New:
- `src/manifoldx/assets/__init__.py`
- `src/manifoldx/assets/obj.py`
- `src/manifoldx/textures.py`
- `examples/teapot_demo.py`
- `examples/assets/teapot/teapot.obj`
- `examples/assets/teapot/teapot_albedo.png`
- `examples/assets/teapot/README.md`
- `scripts/gen_teapot_albedo.py`
- `scripts/prep_teapot_uvs.py` (contingency, only if needed)
- `tests/test_obj_loader.py`
- `tests/test_textures.py`
- `tests/test_textured_material.py`

Modified:
- `src/manifoldx/resources.py` — extend `StandardMaterial` (`albedo_map`, `pipeline_subtype`, `_compile()` variant, `get_texture_bindings`), extend `GeometryRegistry.create_buffers` (UV-aware interleave), add `Material.get_texture_bindings` default, add `Material.uses_lights` predicate.
- `src/manifoldx/renderer.py` — verify pipeline cache keys on `pipeline_subtype`; vertex layout + bind-group layout branches for textured subtype.
- `src/manifoldx/render/passes/mesh.py` — append texture bindings to bind-group entries when material provides them; switch lights detection from string-search to `mat_obj.uses_lights()`.
- `src/manifoldx/__init__.py` — re-export `load_obj`, `load_texture`.
- `pyproject.toml` — promote Pillow from `[viz]` extra to required.
- `CHANGELOG.md` — `[Unreleased]` entry under `### Features`.

Net Python LOC estimate: ~600 added, ~10 modified.
