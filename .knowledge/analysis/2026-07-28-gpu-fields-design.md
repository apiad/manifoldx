# GPU Fields — transpiling the Field algebra to WGSL

**Status:** v1 shipped (transpiler + `examples/gpu_field.py`). CPU path unchanged.

## Goal

The composable `Field` algebra (`manifoldx.modeling.fields`) lets a developer
build terrain/deformation fields from primitives (`ridged`, `fbm`, `warp`,
`remap`, …). Until now a Field only evaluated on the CPU (numpy) and got *baked*
into mesh vertices via `Mesh.displace`. GPU fields let the **same composed
field** run in a shader — for live displacement, detail-normals, or colouring —
so the mesh stays cheap and detail is evaluated per-vertex/per-fragment.

## Design

Additive AST + a transpiler; the CPU path is untouched.

1. **`Field._ast`** — every Field optionally carries a small tuple IR describing
   how it was built (`('mul', ('fbm', seed, freq, oct, lac, gain), ('const', 0.5))`).
   Sources and combinators populate it; hand-written `Field(fn)` callables leave
   it `None`. The numpy `_fn` is still the source of truth for CPU evaluation —
   the AST is a *parallel* description, so existing behaviour and tests are
   unaffected (699 pass).

2. **`gpu_fields.field_to_wgsl(field, name)`** — walks the AST and emits
   `fn <name>(P: vec3<f32>) -> f32`, plus a shared value-noise prelude
   (`WGSL_NOISE_PRELUDE`: hash → trilinear value noise → octave loop with a
   `shape` switch for fbm/ridged/billow). `warp`/`shift` rebind the sample point
   with a `let`; everything else is a pure expression. Unsupported nodes
   (`worley`) or AST-less fields raise `FieldNotTranspilable`.

3. **Usage** — a material bakes the emitted WGSL into its shader and calls the
   field. `examples/gpu_field.py`: a flat 256² plane displaced in the vertex
   shader by a transpiled terrain field, with normals from central differences
   of the same field. No CPU displacement, no baked heightmap.

## Deliberate limitation: noise is decorative, not a CPU match

The CPU noise is permutation-table gradient (Perlin); the WGSL prelude is
hash-based value noise. They do **not** match bit-for-bit — only the
*composition* (octaves, warp, remap, ridged/billow shaping, arithmetic) is
faithful. For displacement/detail this is what matters. If an exact match is
ever needed, upload the permutation table as a texture and port the gradient
noise; deferred as unneeded.

## Next steps (not yet built)

- **Planet near-surface detail.** Wire a transpiled high-frequency detail field
  into the PBR terrain shader as a fragment detail-normal. Needs an isolated
  `vcolor_detail` pipeline subtype (don't touch the shared `vcolor` path). Marginal
  at orbit distance in `planet_dive`; worth it for a low surface-skim pass.
- **Field uniforms.** Let transpiled fields read live params (amplitude, time,
  scroll offset) from the material uniform in the vertex stage — requires the
  material bind group at binding 2 to include `VERTEX` visibility (currently
  fragment-only), a small shared-renderer change. `examples/gpu_field.py` bakes
  amplitude/epsilon as shader consts to avoid this for now.
- **worley → WGSL** (cellular) for cracks/Voronoi on GPU.
