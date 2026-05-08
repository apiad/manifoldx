# Compute systems Phase 2 — Python → WGSL shader compiler

**Status:** Design — approved through brainstorming on 2026-05-08.

**Goal:** Add a Python-as-shader DSL on top of Phase-1's `Compute` base class. Users override `def main(self, i: int)` with a typed-Python body; the base class's default `compile()` traces it to WGSL. Phase-1's API shape (class layout, marker types, bind-group layout, residency model, frame ordering, dispatch resolution, `engine.compute(cls)`) is preserved byte-for-byte. The only thing changing is what lives inside `compile()` by default.

**Non-goal:** A general-purpose Python-to-GPU JIT. The recognized surface is exactly what's needed to express ECS-flavored numeric kernels, and not one feature more. Module-level helper functions, atomics, matrices, transcendentals, and multi-pass kernels are explicit non-goals for v1.

---

## Locked decisions (from brainstorming)

1. **Strategy.** Static AST walk over `inspect.getsource(type(self).main)`. No runtime tracing, no proxy objects. Native Python `for/while/if/return/continue/break` map literally to WGSL control flow. The full body is one pass — AST in, WGSL string out, no IR.

2. **Field syntax.** `self.<binding>[entity_idx].<field>` for indexed component fields; `self.<uniform>` for uniforms. The transpiler hides stride/offset arithmetic, deriving each Component subclass's per-field offset table at class-creation time.

3. **Local types.** PEP-526 annotations on every local. Type-name set: `{int, float, bool, vec3, vec4}`, mapped to WGSL `{i32, f32, bool, vec3<f32>, vec4<f32>}`. `vec3` and `vec4` are imported from `manifoldx.compute.shader`. Reassigned name → emitted as `var`; single-assign → emitted as `let`.

4. **Numeric promotion.** Strict — mixed scalar arithmetic requires explicit `f32(...) / i32(...) / u32(...)` casts. `vec * scalar` and `scalar * vec` are implicit (WGSL native broadcast). Implicit `int + float` raises `ComputeShaderCompileError` with line/col.

5. **Builtins.** Exactly the existing `manifoldx.compute.shader.BUILTINS` set: `vec3, vec4, length, dot, cross, normalize, sqrt, pow, floor, ceil, abs, min, max, clamp`.

6. **Subroutines.** Class methods other than `main` are recognized as helper subroutines. Each method requires full PEP-526 annotations on every parameter and the return type. Recursion is forbidden (WGSL constraint) and detected by call-graph cycle check. Nested functions, lambdas, `@staticmethod`, and `@classmethod` are unsupported in v1. Module-level `@kernel_fn` helpers are explicitly deferred — they need different machinery (import resolution, namespace separation from numpy helpers) and no current kernel demands them.

7. **Validation.** Generated WGSL compiles synchronously at `engine.compute(cls)` time via `device.create_shader_module(...)`. Failures surface with both the Python source line and the wgpu validation message before any frame runs. (Phase-1's first-frame compile is removed.)

8. **Success criterion.** `examples/nbody_compute.py` works with a `def main(self, i)` body factored over a `def pair_accel(...) -> vec3` helper method, produces the same n-body cluster collapse as the Phase-1 hand-written WGSL version (within `np.allclose(rtol=1e-5)`), and the full test suite stays green.

---

## User-facing surface

### Imports

```python
from manifoldx.compute import Compute, Reads, Writes, ReadsWrites, Uniform
from manifoldx.compute.shader import vec3, vec4, length, dot, cross, normalize, sqrt
```

### Reference kernel — the v1 success bar

```python
class GravityKernel(Compute):
    transforms: ReadsWrites[Transform]
    masses:     Reads[Mass]
    velocities: ReadsWrites[Velocity]

    G:         Uniform[float] = 20.0
    softening: Uniform[float] = 0.05
    dt:        Uniform[float] = "frame_dt"
    n:         Uniform[float] = "entity_count"

    workgroup_size = 64
    dispatch       = "entity_count"

    def pair_accel(self, pos_i: vec3, pos_j: vec3, m_j: float) -> vec3:
        diff:   vec3  = pos_j - pos_i
        r2:     float = dot(diff, diff) + self.softening * self.softening
        inv_r3: float = 1.0 / (r2 * sqrt(r2))
        return self.G * m_j * diff * inv_r3

    def main(self, i: int):
        if i >= u32(self.n):
            return
        pos_i: vec3 = self.transforms[i].pos
        accel: vec3 = vec3(0.0, 0.0, 0.0)
        for j in range(u32(self.n)):
            if i == j:
                continue
            accel += self.pair_accel(pos_i, self.transforms[j].pos, self.masses[j].value)
        self.velocities[i].vector += accel * self.dt
        self.transforms[i].pos    += self.velocities[i].vector * self.dt
```

### Recognized surface

| Construct | WGSL output |
|---|---|
| `def main(self, i: int)` | `@compute @workgroup_size(W) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i: u32 = gid.x; … }` |
| `def helper(self, p: T, …) -> R` | `fn _<ClassName>_helper(p: T, …) -> R { … }` |
| `x: T = expr` (single-assign) | `let x: T = …` |
| `x: T = expr` (later mutated) | `var x: T = …` |
| `if/elif/else`, `while`, `continue`, `break`, bare `return` | WGSL equivalents, literally |
| `for j in range(stop)` / `range(start, stop)` | `for (var j: u32 = <start>; j < <stop>; j = j + 1u) { … }` |
| `+ - * / %`, comparisons, `and`/`or`/`not` | WGSL native (with strict-cast rule) |
| `**` (power) | `pow(x, y)` |
| `+= -= *= /=` on locals | WGSL augmented-assign |
| `+= -= *= /=` on `self.<binding>[i].<field>` | Desugar: load → op → store, component-wise for vector fields |
| `self.<binding>[i].<field>` | Read or write, computing offset from the Component's per-field layout table |
| `self.<uniform>` | `uniforms.<uniform>`, type from class annotation |
| `self.helper(args)` | `_<ClassName>_helper(args)` |
| `vec3, vec4, length, dot, cross, normalize, sqrt, pow, floor, ceil, abs, min, max, clamp` | WGSL builtins |
| `f32(x), i32(x), u32(x), bool(x)` | WGSL casts |

### Out of bounds (raise `ComputeShaderCompileError` with source line + hint)

- Nested functions or lambdas inside any method body.
- `@staticmethod`, `@classmethod`, decorators of any kind.
- Module-level helper calls.
- Any name that isn't a parameter, an annotated local, a `self.<binding/uniform>` access, a sibling method, or a recognized builtin/cast.
- `list`, `dict`, `tuple`, set, comprehensions, generator expressions, `yield`.
- Implicit numeric promotion (`int + float` without a cast).
- Multi-target assignment (`a = b = …`), tuple unpacking (`a, b = …`).
- `return <value>` in `main` (`main` returns void; only bare `return` allowed). Helper methods must return their declared return type.
- Recursion (direct or via the call graph).
- Plain `Assign` to an unannotated name (must be `AnnAssign` for first introduction).
- Use of `print`, exceptions, `assert`.

---

## Architecture

### New module: `manifoldx/compute/transpile.py`

Single public entry point:

```python
def transpile_compute(cls: type[Compute]) -> str:
    """Walk the class's methods and emit a complete WGSL shader source."""
```

Phase-1's `Compute.compile()` default body becomes:

```python
def compile(self) -> str:
    from manifoldx.compute.transpile import transpile_compute
    return transpile_compute(type(self))
```

User-overridden `compile()` keeps full control — escape hatch for hand-tuned kernels.

### Pipeline (single pass, no IR)

1. **Source extraction.** `inspect.getsource(method)` for every method on `cls`. `textwrap.dedent` so indentation is column-zero. `ast.parse(source)` → `FunctionDef` nodes. Confirm exactly one `main` with signature `(self, i: int)`. Reject decorators and `*args`/`**kwargs`.

2. **Method registry.** Build `{name: FunctionDef}` for every method on the class. Walk call sites in each body (`Call(Attribute(self, name))`) to build a call graph. Detect cycles (recursion) → raise with offending method name + source line.

3. **Symbol environment.** Three nested scopes:
   - **Class-level**: `_reads`, `_writes`, `_uniforms` lists (already populated by Phase-1's `__init_subclass__`).
   - **Method-param scope**: parameter names + their annotated WGSL types.
   - **Local scope**: every `AnnAssign` introduces a name with the annotated type. Plain `Assign` to an unknown name raises `missing-annotation`. Reassignment to a known name with a matching type is fine.

4. **Mutability detection.** Second AST walk per method, counting writes per local. >1 assignment OR any `AugAssign` → `var`; otherwise `let`. Augmented-assign on a `self.<binding>[i].<field>` LHS is its own pattern (load → op → store, component-wise).

5. **Expression codegen.** Recursive AST → WGSL emitter, returning `(wgsl_text, wgsl_type)`:
   - `Constant(int|float|bool)` → typed literal.
   - `Name(x)` → look up in scope, emit `x` with bound type.
   - `Attribute(self, "<uniform>")` → `uniforms.<uniform>`.
   - `Subscript(Attribute(self, "<binding>"), idx)` followed by `Attribute("<field>")` → resolve via the Component's `_layout` table; emit scalar load, or `vec3<f32>(buf[off], buf[off+1], buf[off+2])` reconstruction for vector fields.
   - `Call(Name("vec3"|...))` → recognized builtin → WGSL function call; type from builtin signature table.
   - `Call(Name("f32"|"i32"|"u32"|"bool"), arg)` → WGSL cast.
   - `Call(Attribute(self, "method"), args)` → `_<ClassName>_<method>(args)`, return type from method registry.
   - `BinOp` → strict-cast rule. `vec * scalar` allowed; `int + float` raises.
   - `Compare`, `BoolOp`, `UnaryOp` → straightforward.

6. **Statement codegen.** `If/While/For/Return/Continue/Break` → WGSL equivalents. `For(target, range_call, body)` is special-cased to a c-style `for` with a `u32` counter. `AnnAssign` → `let`/`var` based on the mutability table. `AugAssign` on a storage-buffer LHS desugars per field; on a local emits the WGSL `+=`.

7. **Kernel wrapper & method emission.** Helpers emit first as free `fn _<ClassName>_<name>(...) -> R { … }`. `main` emits last, wrapped:

   ```wgsl
   @compute @workgroup_size(W) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
       let i: u32 = gid.x;
       <body>
   }
   ```

8. **Header.** Reuse Phase-1's existing header generator that emits the `Uniforms` struct and `@group(0) @binding(K)` declarations from `_uniforms`/`_reads`/`_writes`. Same source of truth, no duplication.

9. **Validation.** `engine.compute(cls)` calls `transpile_compute(cls)` → `device.create_shader_module(code=...)`. If wgpu raises, the error is wrapped to include the originating Python source location via a line table built during emission.

### Component layout table

A small new piece of metadata per Component subclass — derived once in `Component.__init_subclass__`:

```python
Component._layout: dict[str, tuple[int, int]]
# { field_name: (offset_in_floats, length_in_floats) }
```

For `Transform`, `_layout = {"pos": (0, 3), "rot": (3, 4), "scale": (7, 3)}`. For `Mass`, `{"value": (0, 1)}`. For `Velocity`, `{"vector": (0, 3)}`. The transpiler reads this table to emit offsets — no hardcoded knowledge of any specific component.

---

## What v1 ships

### New code

- **`src/manifoldx/compute/transpile.py`** (~700 lines): AST walker, builtin signature table, type rules, error formatter, line-table tracker.
- **`src/manifoldx/compute/_core.py`** (+10 lines): default `Compute.compile()` calls the transpiler.
- **`src/manifoldx/engine.py`** (+15 lines): `engine.compute(cls)` validates synchronously at registration time.
- **`src/manifoldx/components.py`** (+30 lines): `Component.__init_subclass__` builds the per-field `_layout` table from existing annotations.

### New tests

- **`tests/test_compute_transpile.py`** (~400 lines): one focused test per language feature; `.wgsl.expected` snapshot files alongside; one full-kernel snapshot for the Gravity demo. Plus error-message tests covering each unsupported construct.
- **`tests/test_compute_transpile_integration.py`** (~150 lines): runs the transpiled `GravityKernel` on a 16-entity engine for one frame, asserts `np.allclose(rtol=1e-5)` against the Phase-1 hand-written WGSL on the same inputs.

### Demo rewrite

- **`examples/nbody_compute.py`**: replace the WGSL string + `compile()` override with the `def main` + `def pair_accel` body shown above. Net diff ≈ 60 lines.

### File budget

~700 lines new code + ~550 lines tests + ~60 lines demo diff. Reasonable for a 10–14 task plan.

---

## Testing strategy

- **Snapshot codegen tests.** Each language feature gets a focused test: input Python AST snippet → expected WGSL substring. Single-feature tests stay readable; full-kernel snapshots only on the Gravity demo. Snapshots live next to the test as `.wgsl.expected` files; updated via `pytest --snapshot-update`.

- **Error-message tests.** Each unsupported construct gets a test that confirms the error includes the source line, column, category, and a hint string. Categories tested: `unsupported-construct`, `missing-annotation`, `unknown-name`, `type-mismatch`, `implicit-promotion`, `recursion`, `wgpu-validation`.

- **Numeric integration tests.** Spawn 16 entities with deterministic positions/masses, run one frame of the transpiled `GravityKernel`, compare the resulting `Velocity.vector` and `Transform.pos` arrays to those produced by the Phase-1 hand-written WGSL on the same inputs. Tolerance: `np.allclose(rtol=1e-5)`. Fail-loud if they diverge.

- **Demo regression.** Smoke-render `examples/nbody_compute.py` for 2 seconds at 30fps, compare frame-100 transform array against a stored fixture from the Phase-1 hand-written version.

---

## Error UX

- Single exception type: `ComputeShaderCompileError`, raised by `transpile_compute(cls)` and wrapped around any wgpu validation error from `device.create_shader_module(...)`.

- Format:

  ```
  <file>:<line>:<col>: <category>: <message>
    <source line>
    <caret>
  ```

- Categories: `unsupported-construct`, `missing-annotation`, `unknown-name`, `type-mismatch`, `implicit-promotion`, `recursion`, `wgpu-validation`.

- The `wgpu-validation` case maps the WGSL line back to the originating AST node via the line table built during emission. Falls back to `(WGSL line N — source mapping unavailable)` if the table can't resolve, with the raw WGSL source attached.

---

## Out of scope for v1

- Module-level `@kernel_fn` helpers (resolved when a real second kernel demands cross-class reuse).
- `mat4`, `mat3`, `atomicAdd/Min/Max`, group-shared memory, barriers.
- Transcendentals beyond `sqrt` and `pow` (`sin`, `cos`, `tan`, `exp`, `log`, etc.).
- Multi-pass kernels (still one shader per Compute).
- Compute → CPU readback for same-frame CPU systems (still next-frame).
- Ping-pong buffers (Phase-1 deferred this; still deferred).
- User-extensible auto-bound uniforms or dispatch symbols.
- Type unification beyond first-assignment annotation (no Hindley–Milner).
- Tuple unpacking, multi-target assignment, augmented-assign on multi-attribute LHS (`self.foo.bar += …` requires explicit load/op/store in user code).

---

## Self-review

- **Placeholder scan:** none.
- **Internal consistency:** Phase-1 and Phase-2 share API shape, marker types, bind-group layout, frame ordering, residency model. The only thing that differs is `compile()`'s default body (now calls `transpile_compute`) and the timing of WGSL validation (moved from first-frame to `engine.compute(cls)`). The `Component._layout` table is the only new piece of metadata; it's derived from existing per-field annotations and is also useful outside the transpiler (introspection, debugging).
- **Scope check:** ~700 lines transpiler + 550 lines tests + 60 lines demo diff. Single sub-project, single plan. No decomposition needed.
- **Ambiguity check:** "PEP-526 annotations on every local" is locked — plain `Assign` to a name not previously introduced raises `missing-annotation`. "Strict numeric promotion" is locked — mixed scalar arithmetic always requires an explicit cast. Both documented above.
- **Subroutine semantics:** class methods only; module-level helpers explicitly deferred; recursion explicitly forbidden with cycle-check.
