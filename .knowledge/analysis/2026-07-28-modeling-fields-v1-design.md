# Modeling Fields v1 — Design (composable field algebra)

**Date:** 2026-07-28
**Scope:** A composable scalar-field algebra for `manifoldx.modeling` — a fluent `Field` type plus noise/pattern sources and combinators — so a developer builds their own terrain (and any field-driven deformation or, later, coloring) out of small reusable primitives. **No terrain generator, no opinionated presets.**

## Motivation

The modeling subsystem already lets you `plane.displace(fbm)` for a rolling heightfield. But `fbm` alone gives only soft hills, and composing richer terrain means hand-writing numpy field functions. This sub-project makes *fields themselves* first-class composable values: sources you mix, mask, warp, and remap into whatever shape you want, then feed to `displace` (today) or `color_by` (SP-2). It is the first of four composable-primitive sub-projects for PCG (fields → color → curves/sweep → scatter); it prescribes nothing about what "terrain" is.

## The `Field` type

A `Field` wraps a callable `(points: (K, 3) float) -> (K,) float32`. A `Field` is itself callable, so every existing/ future consumer that samples a field (`displace`, later `color_by`) accepts a `Field` unchanged — no downstream API change.

```python
class Field:
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]): ...
    def __call__(self, points: np.ndarray) -> np.ndarray: ...   # -> (K,) float32
```

Scalars are coerced to constant fields inside operators, so `field + 0.5` and `2.0 * field` work.

### Operators (Field ⊕ Field or Field ⊕ scalar)

`+`, `-`, `*`, `/`, and unary `-`, each returning a new `Field` that samples its operands and combines element-wise. Reflected forms (`__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`) so `scalar ⊕ field` works.

### Combinator methods (each returns a new `Field`)

| method | meaning |
|---|---|
| `mix(other, t)` | linear blend `self*(1-t) + other*t`; `t` scalar or `Field` |
| `minimum(other)` / `maximum(other)` | element-wise min / max (unions & intersections of shapes) |
| `clamp(lo, hi)` | clip values to `[lo, hi]` |
| `remap(a, b, c, d)` | linear remap `(a,b) -> (c,d)` |
| `abs()` | `|value|` |
| `power(n)` | `value ** n` (sharpen/soften after remap to ≥0) |
| `scale(s)` / `bias(b)` | sugar for `* s` / `+ b` |
| `warp(amount, fx=None, fy=None, fz=None)` | **domain warp**: sample `self` at `points + amount * (fx, fy, fz)(points)`; missing axis fields = 0 |

`warp` is the key terrain primitive: offsetting sample coordinates by other fields turns concentric noise into meandering ridges and rivers.

## Sources (module functions returning `Field`)

- `perlin(seed=None, freq=1.0)` — **moved here from `noise.py`**, now returns a `Field`. Range ~[-1, 1].
- `fbm(seed=None, freq=1.0, octaves=4, lacunarity=2.0, gain=0.5)` — **moved here**, now a `Field`. Range ~[-1, 1].
- `ridged(seed=None, freq=1.0, octaves=4, lacunarity=2.0, gain=0.5)` — ridged multifractal: per octave `(1 - |perlin|)^2`, weighted-summed. Range ~[0, 1]. Mountain ridges / cliffs.
- `billow(seed=None, freq=1.0, octaves=4, lacunarity=2.0, gain=0.5)` — per octave `|perlin|`, summed. Range ~[0, 1]. Dunes / puffy hills.
- `worley(seed=None, freq=1.0, feature="f1", metric="euclidean")` — cellular/Voronoi: scatter feature points on a seeded integer lattice (one per cell), return distance to nearest (`"f1"`) or `F2 - F1` (`"f2f1"`, for cracks/borders). Range ≥ 0.
- `constant(value)` — a field that returns `value` everywhere.
- `coord(axis)` — `axis` in `{"x","y","z"}`: returns that world coordinate as a field (linear ramps, height/position masks).
- `distance(center=(0,0,0))` — Euclidean distance from `center` (islands, craters, radial masks).

## Module layout & backward compatibility

- New `src/manifoldx/modeling/fields.py` — `Field` class + all sources above.
- `src/manifoldx/modeling/noise.py` — becomes a thin back-compat shim: `from manifoldx.modeling.fields import perlin, fbm` (and keeps its `_resolve_rng` used elsewhere if needed). Existing `from manifoldx.modeling import noise; noise.fbm(...)` keeps working, now returning a `Field` (still callable, so `displace`/tests are unaffected).
- `src/manifoldx/modeling/__init__.py` — export `fields` (module) and `Field`.
- The gradient/value-noise core (`perlin`) is implemented once in `fields.py`; `ridged`/`billow`/`fbm` layer it; the previous `noise.py` implementation moves, not duplicates.

**No renderer or ECS changes.** Fields feed `displace` (already `field(points)`). Coloring is SP-2.

## Testing

- **Algebra:** on constant/known fields, verify `+ - * /`, `mix`, `minimum/maximum`, `clamp`, `remap`, `power`, `scale/bias` produce the exact expected values; `warp` shifts sampled output; scalar⊕field and field⊕scalar both work.
- **Sources:** determinism (same seed → identical arrays; different seed → different), range bounds (`ridged`/`billow`/`worley` ≥ 0; `perlin` in ~[-1,1]), shape `(K,)`; `worley` `"f2f1"` is small near cell centers and large near borders; `coord("x")` returns the x column; `distance` matches `np.linalg.norm`.
- **Back-compat:** `noise.fbm(...)` still callable and equal to `fields.fbm(...)` for the same seed; existing deformer/demo tests still pass unchanged.
- **Demo:** `examples/modeling_fields.py` — a developer-composed field (e.g. `ridged*0.7 + fbm*0.15`, domain-warped, remapped) displacing a subdivided `plane` into mountains, lit with a raking `DirectionalLight` + shadows; monochrome clay render (color arrives in SP-2). Rendered to MP4 for visual regression.

## Non-goals (this slice)

- Per-vertex color / any renderer change (SP-2).
- Curves/sweep (SP-3), scatter (SP-4).
- Erosion, hydrology, biome systems — those are *things a developer composes*, not primitives we ship.
- A node graph for fields — the callable + operators are the composition mechanism.
- 2-D-only optimization: fields are sampled at 3-D points (works for heightfields on any plane and for volumetric use).
