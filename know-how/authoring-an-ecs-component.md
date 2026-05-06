---
triggers: ["new component", "add a component", "design a component", "ECS component", "new field per entity", "per-entity scalar", "per-entity vector", "marker component"]
---

# Authoring an ECS component

## When to use this

Reach for this when adding a new ECS component to ManifoldX — anything that holds per-entity data (scalar, vector, atlas-slice index, etc.) or tags entities for a render path (marker, no per-entity data). Covers both `manifoldx.viz`-style sci-viz components and core engine components written by hand.

Does NOT cover: built-in components like `Transform`, `Mesh`, `Material`. Those pre-date the `Component` base class and have richer `get_data` semantics (composite shapes, registry-aware id lookup) that don't fit the simple-field pattern. Don't migrate them without a separate design pass.

## Inputs

- **Name.** PascalCase. The class name becomes the storage key in `engine.store._components`.
- **Field shape(s).** Each field is one of:
  - `Float` (1 column per entity)
  - `Vector3` (3 columns)
  - `Vector4` (4 columns)
  All from `manifoldx.types`. Multiple fields are packed in declaration order.
- **Default value (optional).** If you want a non-zero default for a field, give it a class-level value. Missing → 0.
- **Optional render-path integration.** If the component participates in routing (sprite, label, etc.), the renderer's `RenderPipeline.run` needs to know about it.

## Procedure

### 1. Declare the class

```python
from manifoldx.components import Component
from manifoldx.types import Float, Vector3

class Velocity(Component):
    """Per-entity velocity in world space."""
    velocity: Vector3                    # 3 cols, default (0, 0, 0)

class Lifetime(Component):
    """Per-entity remaining lifetime in seconds."""
    seconds: Float = 1.0                 # 1 col, default 1.0

class Highlighted(Component):
    """Marker — no per-entity data."""
    # no annotations → zero-width
```

`__init_subclass__` reads the annotations at class-creation time and computes:
- `_dtype` (defaults to `np.dtype("f4")`; override with a class attribute if you need otherwise — rare)
- `_shape` (total_cols summed across all annotated fields; `(0,)` for marker components)
- `_field_specs` (ordered list of `(name, shape)` tuples)
- `_field_defaults` (dict of field → default value)

The class is also added to a module-level registry (`_COMPONENT_CLASSES` in `manifoldx.components`).

### 2. Use it in spawn — no manual registration needed

```python
engine.spawn(
    Mesh(cube_geo),
    Material(StandardMaterial(...)),
    Transform(pos=(0, 0, 0)),
    Velocity(velocity=(1, 0, 0)),        # broadcast same velocity to all
    Lifetime(seconds=np.array([...])),    # per-entity lifetimes
    n=N,
)
```

`engine.spawn` detects the `Component` instance, calls `type(value).register(self.store)` (idempotent — no-op if already registered), then calls `value.get_data(n, registry)` to materialize the per-entity numpy array.

### 3. (Only if you need it) override `__init__` or `get_data`

The base class's defaults handle:
- `MyComp()` → fields fall back to class-level defaults
- `MyComp(field=scalar)` → broadcast to all rows
- `MyComp(field=array_shape_size)` → broadcast same vector to all rows
- `MyComp(field=array_shape_n)` → one value per entity (size==1 fields only; reshapes to `(n, 1)`)
- `MyComp(field=array_shape_n_size)` → verbatim per-entity assignment
- Anything else → `ValueError`

If you need richer semantics — e.g. per-instance registry lookup like `Mesh` does — override `__init__` and `get_data` in the subclass. Keep the dtype/shape derivation working by leaving the annotations intact.

### 4. Wire render-path routing if applicable

If the component changes how the renderer batches entities (like `PointCloud` flips entities into the sprite path, or `TextLabel + LabelMaterial` flips entities into the label path), edit `RenderPipeline.run` in `src/manifoldx/renderer.py`. Look for the `has_mesh`, `has_point_cloud`, `has_text_label` early-return guards and the per-entity `is_label`, `is_sprite` routing block. Add a parallel guard + branch for the new component, place it in the priority order you want.

Pure-data components (no render-path routing — Velocity, Lifetime, etc.) need nothing here. They're read by user-defined `@engine.system` functions only.

### 5. TDD it

The repo's plans use a TDD rhythm — write a failing test first, confirm fail, implement, confirm pass, commit. For a new component, the canonical test shape is:

```python
def test_default_zero():
    data = MyComp().get_data(n=10)
    assert data.shape == (10, total_cols)
    assert np.all(data == 0.0)  # or default

def test_scalar_broadcast():
    data = MyComp(field=42).get_data(n=5)
    assert np.all(data == 42.0)

def test_per_entity_array():
    arr = np.array([...], dtype=np.float32)
    data = MyComp(field=arr).get_data(n=4)
    np.testing.assert_array_equal(data[:, col_slice], arr)

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        MyComp(field=np.array([1, 2, 3])).get_data(n=4)
```

See `tests/viz/test_components_viz.py` and `tests/viz/test_label_components.py` for working examples.

### 6. Re-export from the public surface

If the component is part of a subpackage (e.g. `manifoldx.viz`), add it to that subpackage's `__init__.py` `__all__` list. Pin the public surface with a smoke test like `tests/viz/test_public_surface.py`.

## Outputs

- Class definition appended to the right module (`src/manifoldx/viz/components.py` for sci-viz, `src/manifoldx/components.py` only if it's a core component — rare).
- Optional renderer wiring in `src/manifoldx/renderer.py`.
- Tests in the matching `tests/` subdirectory.
- Re-export in the relevant `__init__.py` if part of a public subpackage.
- One commit per coherent step (declaration + tests, then renderer wiring if applicable, then re-export — three commits is usually about right).

## Gotchas

- **Don't `engine.store.register_component(...)` by hand.** The Component base auto-registers on first spawn. If you write the manual call anyway, it works (the auto-register is idempotent), but it's noise — remove it.
- **Marker components are zero-width, not zero-instances.** A class with no annotations gets `_shape = (0,)` — the store array has shape `(max_entities, 0)`. That's fine; it's just a tag. Don't try to make it `_shape = ()` — the store expects a tuple of column counts.
- **Don't use `Float` for integer-valued fields.** ManifoldX stores everything as `f4` for SoA homogeneity (the same `_FieldView` path serves all numeric fields). For an integer-semantic field like `TextLabel.index`, declare `index: Float = 0.0` and cast to `u32` inside the shader. The atlas slice indices live as floats in CPU memory and uint32s in WGSL — an intentional asymmetry.
- **Class-level defaults can't be `None`.** A class attribute set to `None` is indistinguishable from "no default" in the current `__init_subclass__` logic. If you want a None-meaning-default, leave the attribute unset entirely; the field's default becomes "all zeros".
- **Order of fields = order in storage.** When packing multiple fields into one component, the column layout is the order the annotations appear in the class body. Don't reorder annotations after the fact without checking that no callers index columns directly.
