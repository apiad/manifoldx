# Geometric Modeling v1 — Design

**Date:** 2026-07-28
**Scope:** A host-side, numpy-first procedural geometry subsystem — `manifoldx.modeling` — that lets you build and transform meshes in code and hand the result to the existing render path. Programmatic warping, sculpting, topology, and (later) booleans, composed as PCG-style pipelines. Everything runs on CPU, is deterministic, and bakes to the geometry dict the `GeometryRegistry` already consumes.

## Motivation

Today, mesh geometry in manifoldx is either a hand-authored procedural built-in (`cube()`, `sphere()`, `plane()` in `resources.py` / `viz.geometry`) or a loaded `.obj`. There is no way to *transform* geometry in code — no deformation, no procedural sculpting, no CSG. Yet the underlying representation is already ideal for it: geometry is plain numpy (`positions (N,3)`, `normals (N,3)`, `uvs (N,2)`, `indices`), which is exactly the substrate a procedural-content-generation (PCG) layer wants.

This subsystem adds a composable, deterministic modeling layer on top of that representation. The design goal is that a researcher (or the engine's own demos) can write a pipeline like:

```python
from manifoldx import modeling as mdl

mesh = (
    mdl.Mesh.icosphere(subdivisions=4)
        .displace(mdl.noise.fbm(seed=7, freq=1.5, octaves=4), amount=0.3)
        .twist(angle=1.2, axis="y")
        .smooth(iterations=2)
)
engine.spawn(Transform(), Mesh(mesh.to_geometry()), Material(...))
```

The whole layer is CPU/numpy, decoupled from the render loop, and testable without a GPU.

## Design principles

1. **Immutable value type.** `modeling.Mesh` is an immutable value; every operator returns a new `Mesh`. Operators are `Mesh -> Mesh` functions. Pipelines are just method chains (or plain function composition).
2. **Bake, then render.** Modeling produces a `Mesh`; `.to_geometry()` emits the existing geometry dict. **No renderer or ECS changes.** The GPU path is untouched.
3. **Numpy-first, dependency-free.** Every operator is vectorized numpy. The *only* candidate for an external dependency is robust booleans (Batch 4), and the decision there is to **reimplement** rather than vendor (see Batch 4).
4. **Deterministic.** Any operator with randomness (noise, scatter) seeds off the existing `manifoldx.random` module, so a pipeline is reproducible from its seeds.
5. **Testable per operator.** Because each operator is `Mesh -> Mesh` over numpy arrays, tests assert on vertex/topology invariants (bounds, vertex counts, watertightness, normal unit-length) with no GPU device required.

## Naming

The ECS already has a GPU-handle component named `Mesh` (`manifoldx.components` / built-in). The host-side value type is **`Mesh` inside the `manifoldx.modeling` namespace** — accessed as `modeling.Mesh` / `mdl.Mesh` — and is **never re-exported at top level**. There is no `manifoldx.Mesh` alias for the modeling type. This keeps the two `Mesh` names unambiguous at every call site: the component is `manifoldx.Mesh`, the value type is `manifoldx.modeling.Mesh`.

## Architecture

New subpackage `src/manifoldx/modeling/`, parallel to `viz/`, `gui/`, `compute/`.

```
src/manifoldx/modeling/
    __init__.py        # re-exports Mesh, noise, Falloff, primitives
    mesh.py            # the Mesh value type + core (normals, adjacency, to_geometry)
    primitives.py      # box/plane/icosphere/cylinder/torus generators
    deform.py          # Batch 1
    sculpt.py          # Batch 2
    topology.py        # Batch 3
    boolean.py         # Batch 4 (own design cycle)
    noise.py           # Perlin / fbm value-noise fields, seeded
```

### The `Mesh` value type (`mesh.py`)

```python
@dataclass(frozen=True)
class Mesh:
    positions: np.ndarray   # (N, 3) float32
    faces:     np.ndarray   # (M, 3) uint32  — triangles
    normals:   np.ndarray | None = None   # (N, 3) float32, lazily computed
    uvs:       np.ndarray | None = None   # (N, 2) float32, optional
```

Core methods (Batch 0):

- `with_positions(new_positions) -> Mesh` — the single internal primitive every deformer/sculpt op uses; returns a copy with positions replaced and normals invalidated (set to `None`).
- `recompute_normals() -> Mesh` — area-weighted vertex normals from faces; returns a `Mesh` with `normals` populated.
- `adjacency() -> VertexAdjacency` — lazily-built, cached one-ring vertex neighborhood (CSR-style: `neighbors`, `offsets`) used by `smooth` and any Laplacian op. Cached on the instance (frozen dataclass → cache via `object.__setattr__` or a `functools.cached_property`-equivalent that respects immutability).
- `to_geometry() -> dict` — emits `{"positions", "normals", "uvs"?, "indices"}` with dtypes/shape matching `cube()` output, so `GeometryRegistry.create_buffers` consumes it unchanged. Triangulated faces flatten to the `indices` uint32 stream. Auto-computes normals if absent.
- `from_geometry(dict) -> Mesh` — inverse, so a loaded `.obj` (via existing `load_obj`) can enter the modeling pipeline.

Design note: faces are stored as `(M, 3)` triangles internally (topology ops are far cleaner on a 2-D face array than a flat index stream); `to_geometry()` flattens at the boundary.

### Determinism (`noise.py`)

`noise.perlin(...)`, `noise.fbm(...)` return **field callables** `(points (K,3)) -> values (K,)` seeded via `manifoldx.random`. A deformer like `displace(field, amount)` samples the field at each vertex position. Field-as-callable keeps noise decoupled from meshes and independently testable.

### Integration surface

The entire subsystem touches the rest of the engine through exactly one seam: `Mesh.to_geometry()` → the dict passed to the ECS `Mesh` component / `GeometryRegistry`. No changes to `renderer.py`, `resources.py` internals, materials, or the ECS. `from_geometry` + existing `load_obj` let external meshes enter the pipeline.

## Batch roadmap (value-per-buck order)

The subsystem ships across multiple implementation plans, one per batch (mirroring how sci-viz-v1 spanned plans 1–4). This design doc is the umbrella; each batch gets its own plan. Batch 4 additionally gets its own design doc.

### Batch 0 — Foundation *(vertical slice)*

The `Mesh` value type, `primitives.py` (box/plane/icosphere/cylinder/torus — lifting and reusing the existing `resources.py`/`viz.geometry` generators where they exist, not duplicating), `recompute_normals`, `adjacency`, `noise.py`, and `to_geometry`/`from_geometry`. **End-to-end deliverable:** `examples/modeling_asteroid.py` spawns a bare `Mesh.icosphere(4)` via `.to_geometry()`, PBR-lit with `StandardMaterial`, slowly rotating — viewable but plain, proving the `modeling → GPU` seam before any operator exists. This is the smoke-test demo the whole subsystem is validated against; it grows one operator at a time through Batch 1.

### Batch 1 — Deformers *(highest value/buck)*

Pure per-vertex numpy on `positions`, zero topology change, normals recomputed after each:

- `twist(angle, axis)` — rotate each vertex about `axis` proportional to its coordinate along `axis`.
- `bend(angle, axis)` — bend along an axis.
- `taper(factor, axis)` — scale cross-section proportional to position along `axis`.
- `displace(field, amount, along="normal"|vector)` — offset each vertex by a sampled scalar field (the PCG workhorse: terrain, blobs, greebles).

Deferred to the tail of this batch (pricier, need extra machinery): `ffd(lattice)` free-form deformation, and `bend_along_curve(curve)`.

**Demo payoff:** `examples/modeling_asteroid.py` (the Batch 0 seam demo) upgrades in place — `.displace(noise.fbm(seed=7, octaves=5), amount=0.35)` turns the icosphere into a lumpy rock, then `.twist(...)`/`.smooth(1)` refine it into a recognizable procedural asteroid. Renders to MP4 via the existing `--render` smoke path for visual regression.

### Batch 2 — Sculpt brushes *(high value, moderate)*

A `Falloff` region model (center + radius + smoothstep profile) selecting a weighted vertex set, then brushes applied programmatically (no viewport):

- `draw(center, radius, strength)` / `inflate(...)` — displace selected vertices along their normals.
- `pinch(center, radius, strength)` — pull selected vertices toward the center.
- `flatten(center, radius, strength, plane?)` — project selected vertices toward a plane.
- `smooth(center?, radius?, iterations)` — Laplacian smoothing over the one-ring `adjacency` (global if no region given). This is the operator that *requires* Batch 0's adjacency.

### Batch 3 — Topology *(mixed)*

- `subdivide(iterations, scheme="midpoint"|"loop")` — increases resolution; also an *enabler* for high-res inputs to Batches 1–2. Midpoint first, Loop second.
- `extrude(faces_selection, distance)` — extrude a face selection along normals.
- `decimate(grid)` — **v1: grid vertex-clustering** (snap vertices to a lattice over the bounding box, drop collapsed faces, compact orphans). Robust and always valid. Quadric-error-metric edge collapse is the quality refinement, **deferred** — clustering ships first because it is safe to build unattended and demonstrably reduces triangle count.

### Batch 4 — Booleans *(highest cost — own design cycle)*

`union` / `difference` / `intersection`. This is the one genuinely deep algorithm in the subsystem — **not** in the "no novel algorithm" register of the other batches.

- **Reimplement, do not vendor.** No `manifold3d` (or other compiled CSG) dependency. Read `manifold3d` and the classic `csg.js`/BSP-tree lineage as references.
- **Staged inside the batch:** a **BSP-tree first cut** (pure, tractable, good enough for most PCG unions/diffs) → **Manifold-style robustness** (halfedge representation, careful coplanar/degenerate/numerical handling) only if the first cut's fragility bites in practice.
- **Own design doc + plan** when we reach it. This umbrella spec only names booleans as reimplemented-and-deferred; it does not specify the algorithm.

## Testing strategy

- Per-operator invariant tests over numpy arrays, no GPU: bounds after a deformer, vertex count preserved by deformers, `smooth` reduces a roughness metric, `subdivide` multiplies face count by the expected factor, normals are unit-length after `recompute_normals`, `to_geometry()` round-trips through `from_geometry()`.
- Determinism tests: same seed → identical arrays; different seed → different arrays.
- One offscreen render smoke test per batch's demo (gated on `get_offscreen_canvas`, `pytest.skip` if no backend), consistent with the repo's existing GPU-optional test convention.

## Non-goals (v1)

- Interactive / viewport sculpting (mouse brushes). This layer is programmatic only.
- GPU/compute-kernel offload of operators. The API is designed so operators *could* later be offloaded to the Python→WGSL compute path, but v1 is CPU/numpy. (This is the "C" evolution from brainstorming.)
- Quad meshes / n-gon topology internally. Triangles only; polygon inputs are triangulated at the boundary.
- Node-graph / lazy-DAG pipeline evaluation. The fluent immutable surface composes into a `Pipeline([...])` value cheaply later if provenance/serialization is wanted; not built now.
- UV generation / re-parameterization for procedurally created geometry beyond what primitives carry.
- Animation of deformation over time (deformers are baked once, not re-evaluated per frame).
