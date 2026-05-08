# Compute Phase 2 — Python → WGSL shader compiler — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Compute.compile()`'s default `NotImplementedError` body with a single-pass AST walker that traces a typed-Python `def main(self, i)` body to WGSL. Phase-1's API shape is preserved byte-for-byte; only `compile()`'s default body and the timing of WGSL validation change.

**Architecture:** New module `manifoldx.compute.transpile` exposing `transpile_compute(cls) -> str`. Each method on the Compute class becomes a free WGSL function (helpers first, `main` last with `@compute @workgroup_size` wrapper). A per-Component `_layout` table (offset, length per field) is the only new piece of metadata. Validation happens at `engine.compute(cls)` registration time via `device.create_shader_module(...)`.

**Tech Stack:** Python 3.13, `ast` + `inspect` from stdlib, `wgpu` for WGSL validation, pytest for tests. No new third-party deps.

**Spec:** `.knowledge/analysis/2026-05-08-compute-phase-2-shader-compiler-design.md`.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `src/manifoldx/components.py` | Add per-Component `_layout: dict[str, tuple[int, int]]` table | Modify |
| `src/manifoldx/compute/shader.py` | Add type-tag classes `vec3`/`vec4`/`bool` usable in PEP-526 annotations | Modify |
| `src/manifoldx/compute/transpile.py` | The transpiler — AST walker, builtin signature table, type rules, error formatter | **Create** |
| `src/manifoldx/compute/_core.py` | `Compute.compile()` default delegates to `transpile_compute(type(self))` | Modify |
| `src/manifoldx/compute/__init__.py` | Re-export `ComputeShaderCompileError` and `transpile_compute` | Modify |
| `src/manifoldx/engine.py` | `engine.compute(cls)` validates synchronously at registration time | Modify |
| `tests/test_compute_transpile.py` | Codegen unit tests + error-message tests | **Create** |
| `tests/test_compute_transpile_integration.py` | Numeric integration test against Phase-1 hand-written WGSL | **Create** |
| `examples/nbody_compute.py` | Rewrite using `def main` + `def pair_accel` helper | Modify |
| `CHANGELOG.md` | `[Unreleased]` entry under "Features" | Modify |

---

## Task 1: Per-Component `_layout` table

**Files:**
- Modify: `src/manifoldx/components.py:64-83` (`Component.__init_subclass__`)
- Modify: `src/manifoldx/components.py` (Transform, Mesh, Material — add explicit `_layout` class attrs since they predate the Component base)
- Test: `tests/test_compute_transpile.py` (new)

- [ ] **Step 1: Write failing test for derived `_layout` on Component subclasses**

```python
# tests/test_compute_transpile.py
"""Unit tests for the Phase-2 Python→WGSL transpiler."""
import pytest


def test_component_layout_table_derived_from_field_specs():
    """Component subclasses get a `_layout` dict mapping field → (offset, length)."""
    from manifoldx.components import Component
    from manifoldx.types import Float, Vector3

    class MyVel(Component):
        vector: Vector3
        spin: Float

    assert MyVel._layout == {"vector": (0, 3), "spin": (3, 1)}


def test_component_layout_handles_marker_only_components():
    """A marker component (no fields) gets an empty `_layout`."""
    from manifoldx.components import Component

    class Marker(Component):
        pass

    assert Marker._layout == {}


def test_builtin_transform_layout():
    """Transform is pre-`Component` base — explicit `_layout` matches its 10-float layout."""
    from manifoldx.components import Transform

    assert Transform._layout == {"pos": (0, 3), "rot": (3, 4), "scale": (7, 3)}


def test_builtin_mesh_and_material_layouts():
    """Mesh and Material are scalar u32 references — `_layout` reflects that."""
    from manifoldx.components import Material, Mesh

    assert Mesh._layout == {"geometry_id": (0, 1)}
    assert Material._layout == {"material_id": (0, 1)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_compute_transpile.py -v`
Expected: FAIL with `AttributeError: type object 'MyVel' has no attribute '_layout'` and three other AttributeErrors.

- [ ] **Step 3: Add `_layout` derivation to `Component.__init_subclass__`**

Edit `src/manifoldx/components.py`. After the existing `cls._shape = (total,)` line in `__init_subclass__`, add:

```python
        # Derive per-field offset table for the Phase-2 transpiler. Each
        # field maps to (offset_in_floats, length_in_floats) within the
        # interleaved per-entity row.
        layout: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, shape in fields:
            length = int(np.prod(shape))
            layout[name] = (offset, length)
            offset += length
        cls._layout = layout
```

Also add `_layout: dict = {}` to the class-level attribute block alongside `_field_specs` and `_field_defaults`.

- [ ] **Step 4: Add explicit `_layout` to the three pre-`Component` built-ins**

In `src/manifoldx/components.py`, add to the `Transform` class body (anywhere before its methods):

```python
    _layout = {"pos": (0, 3), "rot": (3, 4), "scale": (7, 3)}
```

To the `Mesh` class:

```python
    _layout = {"geometry_id": (0, 1)}
```

To the `Material` class:

```python
    _layout = {"material_id": (0, 1)}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v`
Expected: 4 passed.

Run: `uv run pytest tests/test_compute.py tests/test_compute_integration.py -v --tb=short`
Expected: all Phase-1 tests still green (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/components.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): per-Component _layout table for Phase-2 transpiler

Each Component subclass gets a `_layout` dict mapping field name to
(offset_in_floats, length_in_floats), derived in __init_subclass__ from
the existing _field_specs. The pre-Component built-ins (Transform, Mesh,
Material) get explicit _layout class attrs since they don't go through
the base class's annotation walk.

The transpiler reads this table to emit storage-buffer offset arithmetic
for `self.<binding>[i].<field>` accesses without hardcoding component-
specific knowledge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: PEP-526 type-tag classes for kernel-body annotations

**Files:**
- Modify: `src/manifoldx/compute/shader.py` (add `vec3`/`vec4`/`bool` classes alongside existing builtin functions)
- Test: `tests/test_compute_transpile.py`

The existing `shader.vec3` / `shader.vec4` are *callable* sentinels (`@_shader_only` decorated). The transpiler also needs them as *type tags* in PEP-526 annotations like `accel: vec3 = vec3(0.0, 0.0, 0.0)`. The simplest fix: make them dual-role — class-as-type-tag *and* callable factory.

- [ ] **Step 1: Write failing test for `vec3`/`vec4` as type tags**

Append to `tests/test_compute_transpile.py`:

```python
def test_shader_type_tags_usable_as_annotations():
    """vec3 and vec4 work as PEP-526 annotation tags inside kernel bodies."""
    from manifoldx.compute.shader import vec3, vec4

    # The transpiler will look up each annotation as a name reference; the
    # tag's identity is what matters.
    def example(x: vec3, y: vec4) -> vec3:
        return x

    assert example.__annotations__ == {"x": vec3, "y": vec4, "return": vec3}


def test_shader_type_tags_still_raise_when_called_at_runtime():
    """The tags remain shader-only — calling outside a kernel still raises."""
    from manifoldx.compute.shader import vec3, vec4

    with pytest.raises(NotImplementedError, match="shader primitive"):
        vec3(1.0, 2.0, 3.0)
    with pytest.raises(NotImplementedError, match="shader primitive"):
        vec4(1.0, 2.0, 3.0, 4.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py::test_shader_type_tags_usable_as_annotations -v`
Expected: PASS (the existing decorated functions ARE objects, so they work as annotation values without changes).

Run: `uv run pytest tests/test_compute_transpile.py::test_shader_type_tags_still_raise_when_called_at_runtime -v`
Expected: PASS (the existing `@_shader_only` decorator already does this).

- [ ] **Step 3: If both pass, no code change needed for this task — proceed to Step 4**

If either test fails, edit `src/manifoldx/compute/shader.py` so `vec3`/`vec4` are concrete classes whose `__init__` raises `NotImplementedError` matching `"shader primitive"`, and confirm both annotation use and runtime-call-raises still work.

- [ ] **Step 4: Commit (if anything changed)**

If no edits, skip. Otherwise:

```bash
git add src/manifoldx/compute/shader.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): shader type tags usable in PEP-526 annotations

Confirmed via test that the existing shader.vec3 / shader.vec4 sentinels
work as kernel-body annotation tags (the transpiler reads them by
identity from __annotations__) while still raising at call time outside
a compiled kernel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ComputeShaderCompileError` exception

**Files:**
- Create: `src/manifoldx/compute/transpile.py` (with just the exception for now)
- Test: `tests/test_compute_transpile.py`

- [ ] **Step 1: Write failing test for the exception's structure**

Append to `tests/test_compute_transpile.py`:

```python
def test_compute_shader_compile_error_formats_with_source_line():
    """ComputeShaderCompileError carries file/line/col/category and renders cleanly."""
    from manifoldx.compute.transpile import ComputeShaderCompileError

    err = ComputeShaderCompileError(
        category="missing-annotation",
        message="local 'x' used without annotation",
        filename="<kernel>",
        line=12,
        col=8,
        source_line="    x = 1.0",
    )
    text = str(err)
    assert "<kernel>:12:8" in text
    assert "missing-annotation" in text
    assert "local 'x' used without annotation" in text
    assert "    x = 1.0" in text
    # Caret line indicates the column.
    assert "^" in text
    assert err.category == "missing-annotation"


def test_compute_shader_compile_error_without_source_line():
    """ComputeShaderCompileError tolerates missing source_line (e.g. wgpu-validation)."""
    from manifoldx.compute.transpile import ComputeShaderCompileError

    err = ComputeShaderCompileError(
        category="wgpu-validation",
        message="invalid storage binding",
        filename="kernel.py",
        line=0,
        col=0,
        source_line=None,
    )
    text = str(err)
    assert "kernel.py" in text
    assert "wgpu-validation" in text
    assert "invalid storage binding" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py::test_compute_shader_compile_error_formats_with_source_line -v`
Expected: FAIL with `ModuleNotFoundError: manifoldx.compute.transpile`.

- [ ] **Step 3: Create `transpile.py` with the exception class**

Create `src/manifoldx/compute/transpile.py`:

```python
"""Phase-2 Python → WGSL transpiler for Compute kernels.

See `.knowledge/analysis/2026-05-08-compute-phase-2-shader-compiler-design.md`.
"""
from __future__ import annotations


class ComputeShaderCompileError(Exception):
    """Raised when a Compute kernel cannot be transpiled to valid WGSL.

    Carries structured fields for the IDE/REPL to surface clearly:
    file, line, column, error category, the offending source line, and
    a human-readable message.
    """

    def __init__(
        self,
        *,
        category: str,
        message: str,
        filename: str,
        line: int,
        col: int,
        source_line: str | None = None,
    ):
        self.category = category
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        self.source_line = source_line
        super().__init__(self._render())

    def _render(self) -> str:
        head = f"{self.filename}:{self.line}:{self.col}: {self.category}: {self.message}"
        if self.source_line is None:
            return head
        caret = " " * self.col + "^"
        return f"{head}\n  {self.source_line}\n  {caret}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v`
Expected: 6 passed (Tasks 1+3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): ComputeShaderCompileError with source-line caret rendering

First slice of the Phase-2 transpiler module: a structured exception
type that carries category, file, line, col, and the offending source
line. Renders as <file>:<line>:<col>: <category>: <message> followed by
the source line and a caret pointing at the column.

Categories the transpiler will use: unsupported-construct,
missing-annotation, unknown-name, type-mismatch, implicit-promotion,
recursion, wgpu-validation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `transpile_compute` skeleton + source extraction + method registry + recursion check

**Files:**
- Modify: `src/manifoldx/compute/transpile.py` (add the entry point, source extraction, recursion check)
- Test: `tests/test_compute_transpile.py`

- [ ] **Step 1: Write failing tests for source extraction & recursion detection**

Append to `tests/test_compute_transpile.py`:

```python
def test_transpile_extracts_main_function_def():
    """Given a Compute subclass with a `main` method, transpile parses the AST."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import _collect_method_asts
    from manifoldx.components import Transform

    class K(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def main(self, i: int):
            return

    methods = _collect_method_asts(K)
    assert "main" in methods
    fn = methods["main"]
    assert fn.name == "main"
    assert [a.arg for a in fn.args.args] == ["self", "i"]


def test_transpile_rejects_class_without_main():
    """A Compute class with no `main` method raises ComputeShaderCompileError."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import (
        ComputeShaderCompileError,
        _collect_method_asts,
    )
    from manifoldx.components import Transform

    class NoMain(Compute):
        transforms: ReadsWrites[Transform]

    with pytest.raises(ComputeShaderCompileError, match="must define a `main`"):
        _collect_method_asts(NoMain)


def test_transpile_rejects_recursion_self_call():
    """A `main` that calls itself (or a helper that calls itself) is rejected."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import (
        ComputeShaderCompileError,
        _check_no_recursion,
        _collect_method_asts,
    )
    from manifoldx.components import Transform

    class Recursive(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def helper(self, n: int) -> int:
            return self.helper(n)
        def main(self, i: int):
            self.helper(i)

    methods = _collect_method_asts(Recursive)
    with pytest.raises(ComputeShaderCompileError, match="recursion"):
        _check_no_recursion(methods)


def test_transpile_rejects_mutual_recursion():
    """A → B → A is detected as recursion."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import (
        ComputeShaderCompileError,
        _check_no_recursion,
        _collect_method_asts,
    )
    from manifoldx.components import Transform

    class Mutual(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def a(self, n: int) -> int:
            return self.b(n)
        def b(self, n: int) -> int:
            return self.a(n)
        def main(self, i: int):
            self.a(i)

    methods = _collect_method_asts(Mutual)
    with pytest.raises(ComputeShaderCompileError, match="recursion"):
        _check_no_recursion(methods)


def test_transpile_compute_top_level_entry_point_exists():
    """transpile_compute(cls) is the public entry; for now any output is fine."""
    from manifoldx.compute.transpile import transpile_compute
    assert callable(transpile_compute)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "transpile"`
Expected: 5 FAILs with `ImportError: cannot import name '_collect_method_asts'` etc.

- [ ] **Step 3: Implement source extraction, method registry, recursion check, and `transpile_compute` stub**

Append to `src/manifoldx/compute/transpile.py`:

```python
import ast
import inspect
import textwrap
from typing import Dict, List


def _collect_method_asts(cls: type) -> Dict[str, ast.FunctionDef]:
    """Parse every method defined on `cls` into a {name: FunctionDef} dict.

    Methods inherited from object/Compute are ignored. Decorators are
    rejected — Phase-2 supports only plain methods.
    """
    methods: Dict[str, ast.FunctionDef] = {}
    for name, member in cls.__dict__.items():
        if not inspect.isfunction(member):
            continue
        try:
            src = textwrap.dedent(inspect.getsource(member))
        except OSError as e:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"cannot read source for method {name!r}: {e}",
                filename=getattr(member, "__code__").co_filename,
                line=0, col=0, source_line=None,
            )
        module = ast.parse(src)
        fn = module.body[0]
        if not isinstance(fn, ast.FunctionDef):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"{name!r} is not a plain function (lambdas / async / decorators rejected)",
                filename=member.__code__.co_filename,
                line=getattr(fn, "lineno", 0),
                col=getattr(fn, "col_offset", 0),
                source_line=None,
            )
        if fn.decorator_list:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"decorators are not supported on Compute methods (found on {name!r})",
                filename=member.__code__.co_filename,
                line=fn.lineno, col=fn.col_offset, source_line=None,
            )
        methods[name] = fn

    if "main" not in methods:
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"Compute class {cls.__name__!r} must define a `main(self, i: int)` method",
            filename=getattr(cls, "__module__", "<class>"),
            line=0, col=0, source_line=None,
        )

    main = methods["main"]
    arg_names = [a.arg for a in main.args.args]
    if arg_names != ["self", "i"]:
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"`main` must have signature (self, i: int); got {arg_names!r}",
            filename="<class>", line=main.lineno, col=main.col_offset, source_line=None,
        )
    return methods


def _method_calls_in(fn: ast.FunctionDef) -> List[str]:
    """Return the names of sibling-method calls (`self.<name>(...)`) in fn."""
    names: List[str] = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            names.append(node.func.attr)
    return names


def _check_no_recursion(methods: Dict[str, ast.FunctionDef]) -> None:
    """DFS the call graph; raise ComputeShaderCompileError if any cycle is found."""
    graph = {name: _method_calls_in(fn) for name, fn in methods.items()}

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}

    def dfs(node: str, fn: ast.FunctionDef) -> None:
        color[node] = GRAY
        for callee in graph.get(node, []):
            if callee not in graph:
                continue  # ignored — bound binding/uniform attrs aren't methods
            if color[callee] == GRAY:
                raise ComputeShaderCompileError(
                    category="recursion",
                    message=f"recursion detected: {node!r} (in)directly calls itself via {callee!r}",
                    filename="<class>", line=fn.lineno, col=fn.col_offset, source_line=None,
                )
            if color[callee] == WHITE:
                dfs(callee, methods[callee])
        color[node] = BLACK

    for name, fn in methods.items():
        if color[name] == WHITE:
            dfs(name, fn)


def transpile_compute(cls: type) -> str:
    """Walk the class's methods and emit a complete WGSL shader source.

    Single entry point. The Phase-1 Compute base class's default
    `compile()` calls this. Raises `ComputeShaderCompileError` for any
    Python-source-level issue.
    """
    methods = _collect_method_asts(cls)
    _check_no_recursion(methods)
    raise NotImplementedError(
        "transpile_compute body filled in by subsequent tasks"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "transpile"`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): transpile_compute skeleton — source extraction + recursion check

Skeleton of the Phase-2 transpiler:
- _collect_method_asts(cls): inspect.getsource + ast.parse for every
  method defined on the class. Confirms a `main(self, i)` exists and
  rejects decorated methods, lambdas, and missing-main classes.
- _check_no_recursion(methods): DFS the call graph built from
  `self.<name>(...)` call sites; raises with category 'recursion' on
  any cycle.
- transpile_compute(cls): public entry. Currently runs the two checks
  and raises NotImplementedError. Filled in by subsequent tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Type environment & symbol resolution

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

The expression-codegen tasks need a `TypeEnv` object to look up the WGSL type of each name (parameter, local, uniform, binding). This task introduces the env and a `_TYPE_MAP` from Python-side type tags to WGSL type strings.

- [ ] **Step 1: Write failing tests for `TypeEnv` and `_python_type_to_wgsl`**

Append to `tests/test_compute_transpile.py`:

```python
def test_python_type_to_wgsl_basic_scalars_and_vectors():
    from manifoldx.compute.shader import vec3, vec4
    from manifoldx.compute.transpile import _python_type_to_wgsl

    assert _python_type_to_wgsl(int) == "i32"
    assert _python_type_to_wgsl(float) == "f32"
    assert _python_type_to_wgsl(bool) == "bool"
    assert _python_type_to_wgsl(vec3) == "vec3<f32>"
    assert _python_type_to_wgsl(vec4) == "vec4<f32>"


def test_python_type_to_wgsl_unknown_raises():
    from manifoldx.compute.transpile import (
        ComputeShaderCompileError,
        _python_type_to_wgsl,
    )

    with pytest.raises(ComputeShaderCompileError, match="unsupported-construct"):
        _python_type_to_wgsl(list)


def test_type_env_lookup_param_local_uniform_and_binding():
    """TypeEnv knows the WGSL type of each name the transpiler emits."""
    from manifoldx.compute.transpile import TypeEnv

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_local("accel", "vec3<f32>")
    env.set_uniform("G", "f32")
    env.set_binding("transforms", component_name="Transform")

    assert env.lookup("i") == "u32"
    assert env.lookup("accel") == "vec3<f32>"
    assert env.lookup_uniform("G") == "f32"
    assert env.lookup_binding("transforms") == "Transform"


def test_type_env_unknown_name_raises():
    """Looking up an unknown name surfaces a clear error category."""
    from manifoldx.compute.transpile import ComputeShaderCompileError, TypeEnv

    env = TypeEnv()
    with pytest.raises(ComputeShaderCompileError, match="unknown-name"):
        env.lookup("ghost")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "type_env or python_type"`
Expected: 4 FAILs.

- [ ] **Step 3: Implement `_python_type_to_wgsl` and `TypeEnv`**

Append to `src/manifoldx/compute/transpile.py`:

```python
from manifoldx.compute import shader as _shader


_TYPE_MAP = {
    int:           "i32",
    float:         "f32",
    bool:          "bool",
    _shader.vec3:  "vec3<f32>",
    _shader.vec4:  "vec4<f32>",
}


def _python_type_to_wgsl(tp) -> str:
    """Map a Python annotation tag to a WGSL type string."""
    if tp in _TYPE_MAP:
        return _TYPE_MAP[tp]
    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported annotation type: {getattr(tp, '__name__', tp)!r}; "
                f"use int, float, bool, vec3, or vec4",
        filename="<annotation>", line=0, col=0, source_line=None,
    )


class TypeEnv:
    """Lookup table for the WGSL type of each name in scope."""

    def __init__(self) -> None:
        self._params: Dict[str, str] = {}
        self._locals: Dict[str, str] = {}
        self._uniforms: Dict[str, str] = {}
        self._bindings: Dict[str, str] = {}  # binding name → Component class name

    def set_param(self, name: str, wgsl: str) -> None:
        self._params[name] = wgsl

    def set_local(self, name: str, wgsl: str) -> None:
        self._locals[name] = wgsl

    def set_uniform(self, name: str, wgsl: str) -> None:
        self._uniforms[name] = wgsl

    def set_binding(self, name: str, component_name: str) -> None:
        self._bindings[name] = component_name

    def lookup(self, name: str) -> str:
        if name in self._locals:
            return self._locals[name]
        if name in self._params:
            return self._params[name]
        raise ComputeShaderCompileError(
            category="unknown-name",
            message=f"name {name!r} not in scope",
            filename="<expr>", line=0, col=0, source_line=None,
        )

    def lookup_uniform(self, name: str) -> str:
        if name not in self._uniforms:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"uniform {name!r} not declared on Compute class",
                filename="<expr>", line=0, col=0, source_line=None,
            )
        return self._uniforms[name]

    def lookup_binding(self, name: str) -> str:
        if name not in self._bindings:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"binding {name!r} not declared on Compute class",
                filename="<expr>", line=0, col=0, source_line=None,
            )
        return self._bindings[name]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "type_env or python_type"`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): TypeEnv + _python_type_to_wgsl for the transpiler

Lookup table for the WGSL type of each name in scope (parameter, local,
uniform, binding). Plus _python_type_to_wgsl() mapping the supported
PEP-526 annotation set {int, float, bool, vec3, vec4} to WGSL types.
Unsupported annotations and unknown name lookups raise
ComputeShaderCompileError with the right category.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Expression codegen — Constant, Name, `self.<uniform>`, builtins, casts

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

This task introduces `_emit_expr(node, env, src) -> (wgsl_text, wgsl_type)`. It handles the leaves and call-shaped expressions. BinOp and `self.<binding>[i].<field>` come in subsequent tasks.

- [ ] **Step 1: Write failing tests for the leaf cases**

Append to `tests/test_compute_transpile.py`:

```python
def _parse_expr(src: str):
    """Helper — parse a Python expression and return its AST node."""
    import ast
    return ast.parse(src, mode="eval").body


def test_emit_expr_constants():
    """Constant int/float/bool emit as their typed WGSL literals."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    assert _emit_expr(_parse_expr("0"), env, "0") == ("0", "i32")
    assert _emit_expr(_parse_expr("42"), env, "42") == ("42", "i32")
    assert _emit_expr(_parse_expr("0.0"), env, "0.0") == ("0.0", "f32")
    assert _emit_expr(_parse_expr("3.14"), env, "3.14") == ("3.14", "f32")
    assert _emit_expr(_parse_expr("True"), env, "True") == ("true", "bool")
    assert _emit_expr(_parse_expr("False"), env, "False") == ("false", "bool")


def test_emit_expr_name_lookup():
    """Bare names look up in the TypeEnv."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_local("accel", "vec3<f32>")
    env.set_param("i", "u32")
    assert _emit_expr(_parse_expr("accel"), env, "accel") == ("accel", "vec3<f32>")
    assert _emit_expr(_parse_expr("i"), env, "i") == ("i", "u32")


def test_emit_expr_self_uniform():
    """self.G → uniforms.G with the declared type."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_uniform("G", "f32")
    assert _emit_expr(_parse_expr("self.G"), env, "self.G") == ("uniforms.G", "f32")


def test_emit_expr_casts():
    """f32(x) / i32(x) / u32(x) / bool(x) emit as WGSL casts."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_uniform("n", "f32")
    text, typ = _emit_expr(_parse_expr("u32(self.n)"), env, "u32(self.n)")
    assert text == "u32(uniforms.n)"
    assert typ == "u32"

    env.set_local("a", "i32")
    text, typ = _emit_expr(_parse_expr("f32(a)"), env, "f32(a)")
    assert text == "f32(a)" and typ == "f32"


def test_emit_expr_builtin_calls():
    """vec3, dot, length, sqrt, etc. emit as WGSL built-in calls with the right return type."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_local("d", "vec3<f32>")
    env.set_local("r2", "f32")

    text, typ = _emit_expr(_parse_expr("vec3(0.0, 0.0, 0.0)"), env, "vec3(0.0, 0.0, 0.0)")
    assert text == "vec3<f32>(0.0, 0.0, 0.0)"
    assert typ == "vec3<f32>"

    text, typ = _emit_expr(_parse_expr("dot(d, d)"), env, "dot(d, d)")
    assert text == "dot(d, d)"
    assert typ == "f32"

    text, typ = _emit_expr(_parse_expr("sqrt(r2)"), env, "sqrt(r2)")
    assert text == "sqrt(r2)"
    assert typ == "f32"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "emit_expr"`
Expected: 5 FAILs (`_emit_expr` not defined).

- [ ] **Step 3: Implement `_emit_expr` for the leaf cases**

Append to `src/manifoldx/compute/transpile.py`:

```python
# Builtin signature table. Each entry: name -> (arg_count or None, return_type).
# return_type may be a string (concrete WGSL type) or a callable
# `arg_types -> wgsl_type` for polymorphic builtins.
_BUILTIN_RETURNS = {
    "vec3":      (3, "vec3<f32>"),
    "vec4":      (4, "vec4<f32>"),
    "length":    (1, "f32"),
    "dot":       (2, "f32"),
    "cross":     (2, "vec3<f32>"),
    "normalize": (1, lambda ts: ts[0]),
    "sqrt":      (1, "f32"),
    "pow":       (2, "f32"),
    "floor":     (1, lambda ts: ts[0]),
    "ceil":      (1, lambda ts: ts[0]),
    "abs":       (1, lambda ts: ts[0]),
    "min":       (2, lambda ts: ts[0]),
    "max":       (2, lambda ts: ts[0]),
    "clamp":     (3, lambda ts: ts[0]),
}

_CASTS = {"f32", "i32", "u32", "bool"}


def _emit_expr(node, env: TypeEnv, src: str) -> tuple[str, str]:
    """Emit (wgsl_text, wgsl_type) for an expression AST node."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return ("true" if v else "false"), "bool"
        if isinstance(v, int):
            return str(v), "i32"
        if isinstance(v, float):
            return repr(v), "f32"
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported literal {v!r}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    if isinstance(node, ast.Name):
        return node.id, env.lookup(node.id)

    if isinstance(node, ast.Attribute):
        # self.<uniform> only — self.<binding>[i].<field> is handled by
        # the Subscript case (added in Task 7).
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"uniforms.{node.attr}", env.lookup_uniform(node.attr)
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported attribute access: {ast.unparse(node)!r}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    if isinstance(node, ast.Call):
        # Pure-name calls only; method calls (self.helper()) come in Task 9.
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            arg_emits = [_emit_expr(a, env, src) for a in node.args]
            arg_texts = [t for t, _ in arg_emits]
            arg_types = [ty for _, ty in arg_emits]

            if fname in _CASTS:
                if len(node.args) != 1:
                    raise ComputeShaderCompileError(
                        category="unsupported-construct",
                        message=f"{fname}() cast takes exactly one argument",
                        filename="<expr>", line=node.lineno, col=node.col_offset,
                        source_line=None,
                    )
                return f"{fname}({arg_texts[0]})", fname

            if fname in _BUILTIN_RETURNS:
                expected_arity, ret = _BUILTIN_RETURNS[fname]
                if expected_arity is not None and len(node.args) != expected_arity:
                    raise ComputeShaderCompileError(
                        category="unsupported-construct",
                        message=f"{fname} expects {expected_arity} args, got {len(node.args)}",
                        filename="<expr>", line=node.lineno, col=node.col_offset,
                        source_line=None,
                    )
                ret_type = ret(arg_types) if callable(ret) else ret
                return f"{fname}({', '.join(arg_texts)})", ret_type

            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"unknown function {fname!r}; not a recognized builtin or cast",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )

        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported call form: {ast.unparse(node)!r}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported expression: {ast.unparse(node)!r}",
        filename="<expr>", line=node.lineno, col=node.col_offset,
        source_line=None,
    )
```

Note: `vec3<f32>` emission — the sentinel function `vec3` is `_BUILTIN_RETURNS["vec3"]` with return type `"vec3<f32>"`, but the WGSL constructor literal needs to be emitted as `vec3<f32>(args)`, not `vec3(args)`. Adjust the call-emit to special-case the constructors:

```python
            if fname in _BUILTIN_RETURNS:
                expected_arity, ret = _BUILTIN_RETURNS[fname]
                if expected_arity is not None and len(node.args) != expected_arity:
                    raise ComputeShaderCompileError(
                        category="unsupported-construct",
                        message=f"{fname} expects {expected_arity} args, got {len(node.args)}",
                        filename="<expr>", line=node.lineno, col=node.col_offset,
                        source_line=None,
                    )
                ret_type = ret(arg_types) if callable(ret) else ret
                emit_name = ret_type if fname in {"vec3", "vec4"} else fname
                return f"{emit_name}({', '.join(arg_texts)})", ret_type
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "emit_expr"`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): expression codegen — Constants, Name, self.uniform, casts, builtins

_emit_expr(node, env, src) -> (wgsl_text, wgsl_type) for the leaf
expressions and call-shaped nodes:

- Constant int/float/bool → typed WGSL literal.
- Name lookup via TypeEnv.
- Attribute(self, uniform) → uniforms.<name> with declared type.
- Call(Name in {f32,i32,u32,bool}) → WGSL cast.
- Call(Name in BUILTINS) → WGSL builtin call. vec3/vec4 emit as
  vec3<f32>(args) / vec4<f32>(args).

BinOp, self.<binding>[i].<field>, and self.helper(...) come in subsequent tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `self.<binding>[i].<field>` reads

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

The most distinctive Phase-2 feature: index a binding by entity, attribute-access the field, get the right scalar or `vec3<f32>(...)` reconstruction emitted with the per-Component layout offsets.

- [ ] **Step 1: Write failing tests for vector and scalar field reads**

Append to `tests/test_compute_transpile.py`:

```python
def test_emit_expr_binding_vector_field_read():
    """self.transforms[i].pos → vec3<f32>(transforms[i*10u + 0u], …, …)."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_binding("transforms", component_name="Transform")

    text, typ = _emit_expr(
        _parse_expr("self.transforms[i].pos"), env, "self.transforms[i].pos"
    )
    assert text == "vec3<f32>(transforms[i * 10u + 0u], transforms[i * 10u + 1u], transforms[i * 10u + 2u])"
    assert typ == "vec3<f32>"


def test_emit_expr_binding_scalar_field_read():
    """self.masses[j].value → masses[j * 1u + 0u] with type f32."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_param("j", "u32")
    env.set_binding("masses", component_name="Mass")

    text, typ = _emit_expr(
        _parse_expr("self.masses[j].value"), env, "self.masses[j].value"
    )
    assert text == "masses[j * 1u + 0u]"
    assert typ == "f32"


def test_emit_expr_binding_unknown_field_raises():
    """Accessing a field not in the Component's _layout raises a clear error."""
    from manifoldx.compute.transpile import ComputeShaderCompileError, TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_binding("transforms", component_name="Transform")

    with pytest.raises(ComputeShaderCompileError, match="unknown-name"):
        _emit_expr(_parse_expr("self.transforms[i].ghost"), env, "self.transforms[i].ghost")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "binding_vector_field or binding_scalar_field or binding_unknown_field"`
Expected: 3 FAILs.

- [ ] **Step 3: Add field-read handling to `_emit_expr`**

Find the `if isinstance(node, ast.Attribute):` block in `_emit_expr` and add a sibling branch BEFORE the `Attribute(self, uniform)` case for the indexed-binding pattern:

```python
    if isinstance(node, ast.Attribute):
        # self.<binding>[idx].<field> — indexed component field access.
        if (
            isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Attribute)
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id == "self"
        ):
            binding_name = node.value.value.attr
            component_name = env.lookup_binding(binding_name)
            from manifoldx.components import _COMPONENT_CLASSES
            from manifoldx.components import Material, Mesh, Transform
            cls_lookup = {
                "Transform": Transform, "Mesh": Mesh, "Material": Material,
                **_COMPONENT_CLASSES,
            }
            comp_cls = cls_lookup[component_name]
            layout = comp_cls._layout
            if node.attr not in layout:
                raise ComputeShaderCompileError(
                    category="unknown-name",
                    message=f"field {node.attr!r} not declared on component {component_name!r}",
                    filename="<expr>", line=node.lineno, col=node.col_offset,
                    source_line=None,
                )
            offset, length = layout[node.attr]
            stride = sum(L for _, L in layout.values())
            idx_text, _ = _emit_expr(node.value.slice, env, src)
            if length == 1:
                wgsl = f"{binding_name}[{idx_text} * {stride}u + {offset}u]"
                return wgsl, "f32"
            if length == 3:
                parts = ", ".join(
                    f"{binding_name}[{idx_text} * {stride}u + {offset + k}u]"
                    for k in range(3)
                )
                return f"vec3<f32>({parts})", "vec3<f32>"
            if length == 4:
                parts = ", ".join(
                    f"{binding_name}[{idx_text} * {stride}u + {offset + k}u]"
                    for k in range(4)
                )
                return f"vec4<f32>({parts})", "vec4<f32>"
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"field {node.attr!r} has length {length}; only 1/3/4 supported",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )

        # self.<uniform> — fallthrough to the existing uniform branch.
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"uniforms.{node.attr}", env.lookup_uniform(node.attr)

        raise ComputeShaderCompileError(...)  # existing error
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "binding"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): self.<binding>[i].<field> reads in expression codegen

Adds the indexed-component-field read pattern to _emit_expr. Resolves
the binding to a Component class, looks up its _layout table for the
offset/length of the requested field, and emits either a scalar load
or a vec3<f32>/vec4<f32> reconstruction with the right per-element
offsets.

Stride is the sum of all field lengths in the component's _layout —
matches Phase-1's per-entity row layout exactly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: BinOp with strict numeric promotion

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

- [ ] **Step 1: Write failing tests for BinOp scalar/vector and strict-promotion errors**

Append to `tests/test_compute_transpile.py`:

```python
def test_emit_expr_binop_same_scalar_type():
    """f32 + f32 emits cleanly with the shared type."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_local("a", "f32")
    env.set_local("b", "f32")
    text, typ = _emit_expr(_parse_expr("a + b"), env, "a + b")
    assert text == "(a + b)" and typ == "f32"

    text, typ = _emit_expr(_parse_expr("a * b - a / b"), env, "a * b - a / b")
    assert text == "((a * b) - (a / b))" and typ == "f32"


def test_emit_expr_binop_vec3_arithmetic():
    """vec3 + vec3 → vec3, vec3 * f32 → vec3, f32 * vec3 → vec3 (WGSL native broadcast)."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_local("v", "vec3<f32>")
    env.set_local("w", "vec3<f32>")
    env.set_local("s", "f32")

    text, typ = _emit_expr(_parse_expr("v + w"), env, "v + w")
    assert text == "(v + w)" and typ == "vec3<f32>"

    text, typ = _emit_expr(_parse_expr("v * s"), env, "v * s")
    assert text == "(v * s)" and typ == "vec3<f32>"

    text, typ = _emit_expr(_parse_expr("s * v"), env, "s * v")
    assert text == "(s * v)" and typ == "vec3<f32>"


def test_emit_expr_binop_implicit_int_float_raises():
    """int + float without explicit cast raises implicit-promotion."""
    from manifoldx.compute.transpile import ComputeShaderCompileError, TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_local("a", "i32")
    env.set_local("b", "f32")
    with pytest.raises(ComputeShaderCompileError, match="implicit-promotion"):
        _emit_expr(_parse_expr("a + b"), env, "a + b")


def test_emit_expr_pow_lowers_to_pow_call():
    """`x ** y` emits as `pow(x, y)`."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_local("x", "f32")
    env.set_local("y", "f32")
    text, typ = _emit_expr(_parse_expr("x ** y"), env, "x ** y")
    assert text == "pow(x, y)" and typ == "f32"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "binop or pow_lowers"`
Expected: 4 FAILs.

- [ ] **Step 3: Add `BinOp` handling to `_emit_expr`**

Add a new top-level case to `_emit_expr`, BEFORE the final raise:

```python
    if isinstance(node, ast.BinOp):
        l_text, l_type = _emit_expr(node.left, env, src)
        r_text, r_type = _emit_expr(node.right, env, src)
        op = node.op

        if isinstance(op, ast.Pow):
            return f"pow({l_text}, {r_text})", l_type

        op_str = {
            ast.Add:  "+",
            ast.Sub:  "-",
            ast.Mult: "*",
            ast.Div:  "/",
            ast.Mod:  "%",
        }.get(type(op))
        if op_str is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"unsupported binary op {type(op).__name__}",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )

        # Strict promotion. Same-type → fine.
        if l_type == r_type:
            return f"({l_text} {op_str} {r_text})", l_type

        # vec * scalar / scalar * vec native broadcast (WGSL).
        VEC = {"vec3<f32>", "vec4<f32>"}
        if l_type in VEC and r_type == "f32":
            return f"({l_text} {op_str} {r_text})", l_type
        if r_type in VEC and l_type == "f32":
            return f"({l_text} {op_str} {r_text})", r_type

        raise ComputeShaderCompileError(
            category="implicit-promotion",
            message=f"mixed types in binary op: {l_type} {op_str} {r_type}; "
                    f"insert an explicit cast (e.g. f32(...))",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "binop or pow_lowers"`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): BinOp codegen with strict numeric promotion

- + / - / * / / / % between same-type scalars emit cleanly.
- vec3/vec4 * f32 and f32 * vec3/vec4 stay implicit (WGSL native).
- Mixed scalar types (i32 + f32) raise implicit-promotion with a hint.
- ** lowers to pow(x, y).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Method calls (helper subroutines)

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

A `self.helper(args)` call resolves to a free WGSL function `_<ClassName>_<helper>(args)`. The transpiler needs the helper's annotated return type to know the call's value type. Method registry built in Task 4 carries the FunctionDef; return type comes from `fn.returns`.

- [ ] **Step 1: Write failing tests for self-method calls**

Append to `tests/test_compute_transpile.py`:

```python
def test_emit_expr_self_method_call():
    """self.helper(x) → _<ClassName>_helper(x), return type from helper's annotation."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import TypeEnv, _build_method_signatures, _emit_expr
    from manifoldx.compute.shader import vec3
    from manifoldx.components import Transform

    class K(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def helper(self, x: vec3, k: float) -> vec3:
            return x
        def main(self, i: int):
            return

    sigs = _build_method_signatures(K)
    env = TypeEnv()
    env.set_local("x", "vec3<f32>")
    env.set_local("k", "f32")
    env.set_method_sigs(sigs, class_name="K")

    text, typ = _emit_expr(_parse_expr("self.helper(x, k)"), env, "self.helper(x, k)")
    assert text == "_K_helper(x, k)"
    assert typ == "vec3<f32>"


def test_emit_expr_self_method_unknown_raises():
    """self.bogus(...) when bogus isn't a method raises unknown-name."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import (
        ComputeShaderCompileError, TypeEnv,
        _build_method_signatures, _emit_expr,
    )
    from manifoldx.components import Transform

    class K(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def main(self, i: int):
            return

    sigs = _build_method_signatures(K)
    env = TypeEnv()
    env.set_method_sigs(sigs, class_name="K")
    with pytest.raises(ComputeShaderCompileError, match="unknown-name"):
        _emit_expr(_parse_expr("self.bogus()"), env, "self.bogus()")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "self_method"`
Expected: 2 FAILs.

- [ ] **Step 3: Implement `_build_method_signatures`, env.set_method_sigs, and the Call branch for `self.method(...)`**

Append to `src/manifoldx/compute/transpile.py`:

```python
def _build_method_signatures(cls: type) -> Dict[str, Dict[str, str]]:
    """For each method on `cls`, derive {param_name → wgsl_type} + 'return'.

    Used by the call-site emitter (return type) and by Task 12's helper
    emission (parameter declarations + body env).
    """
    methods = _collect_method_asts(cls)
    sigs: Dict[str, Dict[str, str]] = {}
    for name, fn in methods.items():
        sig: Dict[str, str] = {}
        for arg in fn.args.args:
            if arg.arg == "self":
                continue
            if arg.annotation is None:
                raise ComputeShaderCompileError(
                    category="missing-annotation",
                    message=f"parameter {arg.arg!r} of method {name!r} requires a type annotation",
                    filename="<class>", line=arg.lineno, col=arg.col_offset,
                    source_line=None,
                )
            tp = _resolve_annotation(arg.annotation, cls)
            sig[arg.arg] = _python_type_to_wgsl(tp)
        if name != "main":
            if fn.returns is None:
                raise ComputeShaderCompileError(
                    category="missing-annotation",
                    message=f"method {name!r} requires a return-type annotation",
                    filename="<class>", line=fn.lineno, col=fn.col_offset,
                    source_line=None,
                )
            sig["return"] = _python_type_to_wgsl(_resolve_annotation(fn.returns, cls))
        sigs[name] = sig
    return sigs


def _resolve_annotation(node, cls: type):
    """Look up a Name/Attribute annotation in the class's defining module."""
    import sys
    mod = sys.modules.get(cls.__module__)
    src = ast.unparse(node)
    # Names: int, float, bool, vec3, vec4 — try builtins first, then module globals.
    builtins = {"int": int, "float": float, "bool": bool}
    if src in builtins:
        return builtins[src]
    if mod is not None and hasattr(mod, src):
        return getattr(mod, src)
    raise ComputeShaderCompileError(
        category="unknown-name",
        message=f"could not resolve annotation {src!r}; not a builtin or module-level name",
        filename="<class>", line=getattr(node, "lineno", 0),
        col=getattr(node, "col_offset", 0), source_line=None,
    )
```

Extend `TypeEnv` with:

```python
    def __init__(self) -> None:
        ...  # existing
        self._method_sigs: Dict[str, Dict[str, str]] = {}
        self._class_name: str = ""

    def set_method_sigs(self, sigs: Dict[str, Dict[str, str]], *, class_name: str) -> None:
        self._method_sigs = sigs
        self._class_name = class_name

    def lookup_method(self, name: str) -> Dict[str, str]:
        if name not in self._method_sigs:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"method {name!r} not declared on Compute class {self._class_name!r}",
                filename="<expr>", line=0, col=0, source_line=None,
            )
        return self._method_sigs[name]

    @property
    def class_name(self) -> str:
        return self._class_name
```

In `_emit_expr`, add a Call sub-branch BEFORE the existing `Call(Name)` branch:

```python
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            method = node.func.attr
            sig = env.lookup_method(method)
            arg_emits = [_emit_expr(a, env, src) for a in node.args]
            arg_texts = [t for t, _ in arg_emits]
            return f"_{env.class_name}_{method}({', '.join(arg_texts)})", sig["return"]
        # ... existing Name-call branch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "self_method"`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): self.<method>(...) calls resolve to free WGSL functions

- _build_method_signatures(cls) → {method: {param: wgsl_type, return: wgsl_type}}
- TypeEnv.set_method_sigs / lookup_method.
- _emit_expr handles Call(Attribute(self, method)) by emitting
  _<ClassName>_<method>(args). Return type comes from the helper's
  Python return annotation.

Missing parameter or return annotation raises missing-annotation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Statement codegen — AnnAssign, AugAssign, Compare, BoolOp, UnaryOp

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

This task introduces `_emit_stmt(node, env, mut, src) -> str` and the mutability scan helper. We also extend `_emit_expr` with `Compare`, `BoolOp`, `UnaryOp` since they're trivial.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_compute_transpile.py`:

```python
def test_emit_stmt_annassign_let():
    """Single-assign annotated local emits as `let`."""
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_local("a", "f32")  # known so RHS resolves
    env.set_local("b", "f32")
    body = _ast.parse("x: float = a + b").body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "x: float = a + b")
    assert out == "let x: f32 = (a + b);"
    assert env.lookup("x") == "f32"


def test_emit_stmt_annassign_var_when_reassigned():
    """A name reassigned later in the body emits as `var` on first introduction."""
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    body = _ast.parse(
        "x: float = 0.0\n"
        "x = 1.0\n"
    ).body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "x: float = 0.0")
    assert out == "var x: f32 = 0.0;"


def test_emit_stmt_augassign_local_vector():
    """vec3 += vec3 emits straight WGSL augmented assignment."""
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_local("accel", "vec3<f32>")
    env.set_local("d", "vec3<f32>")
    body = _ast.parse("accel += d").body
    mut = _scan_mutability(body)
    assert _emit_stmt(body[0], env, mut, "accel += d") == "accel = (accel + d);"


def test_emit_stmt_augassign_storage_buffer_vector_field():
    """self.transforms[i].pos += dv desugars to per-component load/op/store."""
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_binding("transforms", component_name="Transform")
    env.set_local("dv", "vec3<f32>")
    body = _ast.parse("self.transforms[i].pos += dv").body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "self.transforms[i].pos += dv")
    expected = (
        "transforms[i * 10u + 0u] = (transforms[i * 10u + 0u] + dv.x);\n"
        "transforms[i * 10u + 1u] = (transforms[i * 10u + 1u] + dv.y);\n"
        "transforms[i * 10u + 2u] = (transforms[i * 10u + 2u] + dv.z);"
    )
    assert out == expected


def test_emit_stmt_assign_storage_buffer_scalar_field():
    """self.masses[i].value = x emits a single store."""
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_binding("masses", component_name="Mass")
    env.set_local("x", "f32")
    body = _ast.parse("self.masses[i].value = x").body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "self.masses[i].value = x")
    assert out == "masses[i * 1u + 0u] = x;"


def test_emit_expr_compare_and_boolop_and_unary():
    """Comparisons, and/or/not work."""
    from manifoldx.compute.transpile import TypeEnv, _emit_expr

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_uniform("n", "f32")

    text, typ = _emit_expr(_parse_expr("i >= u32(self.n)"), env, "i >= u32(self.n)")
    assert text == "(i >= u32(uniforms.n))" and typ == "bool"

    env.set_local("a", "bool")
    env.set_local("b", "bool")
    text, typ = _emit_expr(_parse_expr("a and not b"), env, "a and not b")
    assert text == "(a && (!b))" and typ == "bool"

    env.set_local("x", "f32")
    text, typ = _emit_expr(_parse_expr("-x"), env, "-x")
    assert text == "(-x)" and typ == "f32"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "annassign or augassign_local or augassign_storage or assign_storage or compare_and_boolop"`
Expected: 6 FAILs.

- [ ] **Step 3: Implement statement codegen and the missing expr cases**

Add to `_emit_expr` (before the final raise):

```python
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="chained comparisons not supported; use explicit and/or",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        l_text, _ = _emit_expr(node.left, env, src)
        r_text, _ = _emit_expr(node.comparators[0], env, src)
        op_str = {
            ast.Eq: "==", ast.NotEq: "!=",
            ast.Lt: "<",  ast.LtE:   "<=",
            ast.Gt: ">",  ast.GtE:   ">=",
        }.get(type(node.ops[0]))
        if op_str is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"unsupported comparison {type(node.ops[0]).__name__}",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        return f"({l_text} {op_str} {r_text})", "bool"

    if isinstance(node, ast.BoolOp):
        op_str = "&&" if isinstance(node.op, ast.And) else "||"
        emits = [_emit_expr(v, env, src)[0] for v in node.values]
        joined = f" {op_str} ".join(emits)
        return f"({joined})", "bool"

    if isinstance(node, ast.UnaryOp):
        v_text, v_type = _emit_expr(node.operand, env, src)
        if isinstance(node.op, ast.USub):
            return f"(-{v_text})", v_type
        if isinstance(node.op, ast.Not):
            return f"(!{v_text})", "bool"
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported unary op {type(node.op).__name__}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )
```

Add the statement emitter:

```python
def _scan_mutability(body: List[ast.stmt]) -> Dict[str, int]:
    """Count assignments per local name. >1 OR any AugAssign → 'var' later."""
    counts: Dict[str, int] = {}
    aug_targets: set[str] = set()

    def visit(stmts):
        for s in stmts:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                counts[s.target.id] = counts.get(s.target.id, 0) + 1
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        counts[t.id] = counts.get(t.id, 0) + 1
            elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
                aug_targets.add(s.target.id)
                counts[s.target.id] = counts.get(s.target.id, 0) + 1
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.stmt):
                    visit([child])
                elif hasattr(child, "body") and isinstance(getattr(child, "body"), list):
                    visit(child.body)

    visit(body)
    return {**counts, **{k: 999 for k in aug_targets}}


def _emit_stmt(node, env: TypeEnv, mut: Dict[str, int], src: str) -> str:
    """Emit a WGSL statement string for a single AST stmt."""
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="annotated assignment target must be a plain name",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        target_name = node.target.id
        wgsl_type = _python_type_to_wgsl(_resolve_annotation_simple(node.annotation))
        if node.value is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="annotated declarations require an initializer",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        rhs_text, _ = _emit_expr(node.value, env, src)
        env.set_local(target_name, wgsl_type)
        keyword = "var" if mut.get(target_name, 0) > 1 else "let"
        return f"{keyword} {target_name}: {wgsl_type} = {rhs_text};"

    if isinstance(node, ast.Assign):
        # Plain `name = expr` reassignment OR `self.binding[i].field = expr`.
        if len(node.targets) != 1:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="multi-target assignment not supported",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        return _emit_assign(node.targets[0], node.value, op=None, env=env, src=src,
                             line=node.lineno, col=node.col_offset)

    if isinstance(node, ast.AugAssign):
        op_str = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
        }.get(type(node.op))
        if op_str is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"unsupported augmented op {type(node.op).__name__}",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        return _emit_assign(node.target, node.value, op=op_str, env=env, src=src,
                             line=node.lineno, col=node.col_offset)

    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported statement: {ast.unparse(node)!r}",
        filename="<stmt>", line=node.lineno, col=node.col_offset,
        source_line=None,
    )


def _resolve_annotation_simple(node) -> type:
    """Annotation→type for a Name annotation. Vec types resolve via shader module."""
    if isinstance(node, ast.Name):
        builtins = {"int": int, "float": float, "bool": bool}
        if node.id in builtins:
            return builtins[node.id]
        if node.id == "vec3":
            return _shader.vec3
        if node.id == "vec4":
            return _shader.vec4
    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported annotation: {ast.unparse(node)!r}",
        filename="<stmt>", line=getattr(node, "lineno", 0),
        col=getattr(node, "col_offset", 0), source_line=None,
    )


def _emit_assign(target, value, *, op: str | None, env: TypeEnv, src: str,
                 line: int, col: int) -> str:
    """Emit a single assignment or augmented-assignment statement."""
    rhs_text, rhs_type = _emit_expr(value, env, src)

    # Storage-buffer field LHS: self.<binding>[idx].<field>.
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Subscript)
        and isinstance(target.value.value, ast.Attribute)
        and isinstance(target.value.value.value, ast.Name)
        and target.value.value.value.id == "self"
    ):
        binding_name = target.value.value.attr
        component_name = env.lookup_binding(binding_name)
        from manifoldx.components import _COMPONENT_CLASSES, Material, Mesh, Transform
        cls_lookup = {
            "Transform": Transform, "Mesh": Mesh, "Material": Material,
            **_COMPONENT_CLASSES,
        }
        comp_cls = cls_lookup[component_name]
        layout = comp_cls._layout
        if target.attr not in layout:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"field {target.attr!r} not declared on component {component_name!r}",
                filename="<stmt>", line=line, col=col, source_line=None,
            )
        offset, length = layout[target.attr]
        stride = sum(L for _, L in layout.values())
        idx_text, _ = _emit_expr(target.value.slice, env, src)

        def slot(k): return f"{binding_name}[{idx_text} * {stride}u + {offset + k}u]"
        components = ["x", "y", "z", "w"]

        lines = []
        for k in range(length):
            lhs = slot(k)
            rhs_part = rhs_text if length == 1 else f"{rhs_text}.{components[k]}"
            if op is None:
                lines.append(f"{lhs} = {rhs_part};")
            else:
                lines.append(f"{lhs} = ({lhs} {op} {rhs_part});")
        return "\n".join(lines)

    # Plain name target (reassignment). Lookup raises unknown-name if the
    # name was never introduced via AnnAssign — `x = 1.0` without a prior
    # `x: float = ...` is rejected.
    if isinstance(target, ast.Name):
        env.lookup(target.id)
        if op is None:
            return f"{target.id} = {rhs_text};"
        return f"{target.id} = ({target.id} {op} {rhs_text});"

    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported assignment target: {ast.unparse(target)!r}",
        filename="<stmt>", line=line, col=col, source_line=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "annassign or augassign or assign_storage or compare_and_boolop"`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): statement codegen — AnnAssign, AugAssign, Compare/BoolOp/UnaryOp

- _scan_mutability(body) → {name: assignment-count} so AnnAssign emits
  `var` when a name is later reassigned or augmented, `let` otherwise.
- AnnAssign with int/float/bool/vec3/vec4 annotations.
- Assign and AugAssign on locals + on self.<binding>[i].<field> with
  per-component desugaring for vector fields.
- Compare → bool, BoolOp/and/or → bool, UnaryOp/not/usub.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Control flow — If, While, For-range, Return, Continue, Break

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_compute_transpile.py`:

```python
def test_emit_stmt_if_simple():
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_uniform("n", "f32")
    body = _ast.parse("if i >= u32(self.n):\n    return\n").body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "if i >= u32(self.n): return")
    assert out == "if ((i >= u32(uniforms.n))) {\n  return;\n}"


def test_emit_stmt_if_else():
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_local("x", "f32")
    body = _ast.parse(
        "if x > 0.0:\n"
        "    x = 1.0\n"
        "else:\n"
        "    x = -1.0\n"
    ).body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "if x > 0.0: ...")
    assert "if ((x > 0.0)) {" in out
    assert "} else {" in out
    assert "x = 1.0;" in out
    assert "x = (-1.0);" in out


def test_emit_stmt_for_range_stop_only():
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_param("i", "u32")
    env.set_uniform("n", "f32")
    body = _ast.parse(
        "for j in range(u32(self.n)):\n"
        "    continue\n"
    ).body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "for j in range(u32(self.n)): continue")
    assert out == (
        "for (var j: u32 = 0u; j < u32(uniforms.n); j = j + 1u) {\n"
        "  continue;\n"
        "}"
    )


def test_emit_stmt_while():
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    env.set_local("x", "f32")
    body = _ast.parse("while x > 0.0:\n    x = x - 1.0\n").body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "while …")
    assert out.startswith("while ((x > 0.0)) {")
    assert "x = (x - 1.0);" in out


def test_emit_stmt_break_and_bare_return():
    from manifoldx.compute.transpile import TypeEnv, _emit_stmt, _scan_mutability
    import ast as _ast

    env = TypeEnv()
    body = _ast.parse("if True:\n    break\n").body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "...")
    assert "break;" in out

    body = _ast.parse("if True:\n    return\n").body
    mut = _scan_mutability(body)
    out = _emit_stmt(body[0], env, mut, "...")
    assert "return;" in out


def test_emit_stmt_for_non_range_raises():
    from manifoldx.compute.transpile import (
        ComputeShaderCompileError, TypeEnv,
        _emit_stmt, _scan_mutability,
    )
    import ast as _ast

    env = TypeEnv()
    env.set_local("xs", "vec3<f32>")
    body = _ast.parse("for j in xs:\n    continue\n").body
    mut = _scan_mutability(body)
    with pytest.raises(ComputeShaderCompileError, match="unsupported-construct"):
        _emit_stmt(body[0], env, mut, "for j in xs: ...")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "stmt_if or stmt_for or stmt_while or stmt_break or stmt_for_non_range"`
Expected: 6 FAILs.

- [ ] **Step 3: Add control-flow handlers to `_emit_stmt`**

Extend `_emit_stmt` with branches BEFORE the final raise:

```python
    if isinstance(node, ast.If):
        cond_text, _ = _emit_expr(node.test, env, src)
        body_text = _emit_block(node.body, env, mut, src)
        out = f"if ({cond_text}) {{\n{body_text}\n}}"
        if node.orelse:
            else_text = _emit_block(node.orelse, env, mut, src)
            out += f" else {{\n{else_text}\n}}"
        return out

    if isinstance(node, ast.While):
        cond_text, _ = _emit_expr(node.test, env, src)
        body_text = _emit_block(node.body, env, mut, src)
        return f"while ({cond_text}) {{\n{body_text}\n}}"

    if isinstance(node, ast.For):
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="for-loops must iterate over range(...) only",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        if not isinstance(node.target, ast.Name):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="for-loop target must be a plain name",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        args = node.iter.args
        if len(args) == 1:
            start_text, stop_text = "0u", _emit_expr(args[0], env, src)[0]
        elif len(args) == 2:
            start_text = _emit_expr(args[0], env, src)[0]
            stop_text  = _emit_expr(args[1], env, src)[0]
        else:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="range(start, stop, step) — step not supported in v1",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        env.set_local(node.target.id, "u32")
        body_text = _emit_block(node.body, env, mut, src)
        return (
            f"for (var {node.target.id}: u32 = {start_text}; "
            f"{node.target.id} < {stop_text}; "
            f"{node.target.id} = {node.target.id} + 1u) {{\n"
            f"{body_text}\n}}"
        )

    if isinstance(node, ast.Return):
        if node.value is not None:
            # Helper-function return; the caller validates the type.
            v_text, _ = _emit_expr(node.value, env, src)
            return f"return {v_text};"
        return "return;"

    if isinstance(node, ast.Continue):
        return "continue;"

    if isinstance(node, ast.Break):
        return "break;"
```

Add the block emitter:

```python
def _emit_block(stmts: List[ast.stmt], env: TypeEnv, mut: Dict[str, int], src: str) -> str:
    """Emit a sequence of statements, indented two spaces each."""
    lines = []
    for s in stmts:
        text = _emit_stmt(s, env, mut, src)
        for ln in text.splitlines():
            lines.append(f"  {ln}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "stmt_if or stmt_for or stmt_while or stmt_break or stmt_for_non_range"`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): control-flow codegen — if/while/for-range/return/continue/break

- if/elif/else with proper else-branch emission.
- while.
- for j in range(...) — emits `for (var j: u32 = start; j < stop; ...)`.
- bare and value-returning `return`.
- continue, break.
- Non-range for-loops raise unsupported-construct with a clear hint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Function emission + WGSL header (uniforms struct + bindings + `@compute` wrapper)

**Files:**
- Modify: `src/manifoldx/compute/transpile.py`
- Test: `tests/test_compute_transpile.py`

This is the integration task: stitch together the per-method emission with a header that declares uniforms, storage buffers, and wraps `main` with the `@compute @workgroup_size(W)` decoration.

- [ ] **Step 1: Write failing tests for full-class emission**

Append to `tests/test_compute_transpile.py`:

```python
def test_transpile_compute_emits_uniforms_struct():
    """transpile_compute(cls) emits the Uniforms struct from class _uniforms."""
    from manifoldx.compute import Compute, ReadsWrites, Uniform
    from manifoldx.compute.transpile import transpile_compute
    from manifoldx.components import Transform

    class K(Compute):
        transforms: ReadsWrites[Transform]
        G: Uniform[float] = 1.0
        n: Uniform[float] = "entity_count"
        workgroup_size = 64
        dispatch = "entity_count"
        def main(self, i: int):
            return

    wgsl = transpile_compute(K)
    assert "struct Uniforms" in wgsl
    assert "G: f32" in wgsl
    assert "n: f32" in wgsl


def test_transpile_compute_emits_bindings():
    from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
    from manifoldx.compute.transpile import transpile_compute
    from manifoldx.components import Transform, Component
    from manifoldx.types import Float, Vector3

    class Mass(Component):
        value: Float = 1.0

    class K(Compute):
        transforms: ReadsWrites[Transform]
        masses:     Reads[Mass]
        G: Uniform[float] = 1.0
        workgroup_size = 64
        dispatch = "entity_count"
        def main(self, i: int):
            return

    wgsl = transpile_compute(K)
    assert "@group(0) @binding(0) var<uniform> uniforms: Uniforms;" in wgsl
    assert "@group(0) @binding(1) var<storage, read> masses: array<f32>;" in wgsl
    assert "@group(0) @binding(2) var<storage, read_write> transforms: array<f32>;" in wgsl


def test_transpile_compute_wraps_main():
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import transpile_compute
    from manifoldx.components import Transform

    class K(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 32
        dispatch = "entity_count"
        def main(self, i: int):
            return

    wgsl = transpile_compute(K)
    assert "@compute @workgroup_size(32)" in wgsl
    assert "fn main(@builtin(global_invocation_id) gid: vec3<u32>)" in wgsl
    assert "let i: u32 = gid.x;" in wgsl


def test_transpile_compute_emits_helper_methods():
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import transpile_compute
    from manifoldx.compute.shader import vec3
    from manifoldx.components import Transform

    class K(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def make_zero(self) -> vec3:
            return vec3(0.0, 0.0, 0.0)
        def main(self, i: int):
            v: vec3 = self.make_zero()
            return

    wgsl = transpile_compute(K)
    assert "fn _K_make_zero() -> vec3<f32> {" in wgsl
    assert "return vec3<f32>(0.0, 0.0, 0.0);" in wgsl
    # Helpers come before main.
    assert wgsl.index("_K_make_zero") < wgsl.index("fn main(")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "transpile_compute_emits or transpile_compute_wraps"`
Expected: 4 FAILs.

- [ ] **Step 3: Replace `transpile_compute`'s NotImplementedError with the real emitter**

Replace the existing `transpile_compute` body:

```python
def transpile_compute(cls: type) -> str:
    """Walk the class's methods and emit a complete WGSL shader source."""
    methods = _collect_method_asts(cls)
    _check_no_recursion(methods)

    sigs = _build_method_signatures(cls)

    # Resolve binding component names from class _reads / _writes / Phase-1 metadata.
    bindings = _resolve_class_bindings(cls)
    uniforms = _resolve_class_uniforms(cls)

    # Header: Uniforms struct + bindings.
    header = _emit_header(uniforms, bindings)

    # Helper methods.
    helper_chunks: List[str] = []
    for name, fn in methods.items():
        if name == "main":
            continue
        helper_chunks.append(_emit_helper(cls, name, fn, sigs, bindings, uniforms))

    main_chunk = _emit_main(cls, methods["main"], sigs, bindings, uniforms)

    parts = [header, *helper_chunks, main_chunk]
    return "\n\n".join(p for p in parts if p)


def _resolve_class_bindings(cls: type) -> Dict[str, str]:
    """{binding_name: ComponentClassName} from class _reads + _writes."""
    out: Dict[str, str] = {}
    annotations = getattr(cls, "__annotations__", {})
    for name in list(cls._reads) + list(cls._writes):
        ann = annotations.get(name)
        # Marker types stash the parameter as ann.__args__[0] or similar.
        # Phase-1 stores it as Reads[X] / Writes[X] / ReadsWrites[X];
        # the marker stores X on the instance via __class_getitem__.
        comp = _extract_marker_param(ann)
        out[name] = comp.__name__
    return out


def _extract_marker_param(marker_instance) -> type:
    """Pull the Component class out of a Reads[X] / Writes[X] / ReadsWrites[X] marker."""
    # Phase-1's marker types implement __class_getitem__ that returns the marker
    # instance with `_param` set. Adapt as needed.
    if hasattr(marker_instance, "_param"):
        return marker_instance._param
    # Fallback: typing-style __args__.
    args = getattr(marker_instance, "__args__", None)
    if args:
        return args[0]
    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"could not extract component type from marker {marker_instance!r}",
        filename="<class>", line=0, col=0, source_line=None,
    )


def _resolve_class_uniforms(cls: type) -> Dict[str, str]:
    """{uniform_name: wgsl_type}. v1: every Uniform[T] emits as f32 (matches Phase-1 packing)."""
    out: Dict[str, str] = {}
    annotations = getattr(cls, "__annotations__", {})
    for name in cls._uniforms:
        ann = annotations.get(name)
        param = _extract_marker_param(ann)
        out[name] = _python_type_to_wgsl(param)
    return out


def _emit_header(uniforms: Dict[str, str], bindings: Dict[str, str]) -> str:
    lines: List[str] = []
    slot = 0
    if uniforms:
        struct_fields = ", ".join(f"{n}: {t}" for n, t in uniforms.items())
        lines.append(f"struct Uniforms {{ {struct_fields}, }};")
        lines.append(f"@group(0) @binding({slot}) var<uniform> uniforms: Uniforms;")
        slot += 1

    # Determine read vs read_write from the originating Compute class lists.
    # We pass them in via closure on `bindings` ordering; here we look up via the
    # Phase-1 helper that already classifies them.
    # Re-derive: query the calling cls's _reads and _writes split.
    # (Plumbed through by transpile_compute via two dicts in a real implementation.)
    return "\n".join(lines)
```

The two-dict simplification above is incomplete — change `transpile_compute` to pass `cls._reads` / `cls._writes` directly into `_emit_header`:

```python
def _emit_header(
    uniforms: Dict[str, str],
    reads: List[str],
    writes: List[str],
) -> str:
    lines: List[str] = []
    slot = 0
    if uniforms:
        struct_fields = ", ".join(f"{n}: {t}" for n, t in uniforms.items())
        lines.append(f"struct Uniforms {{ {struct_fields}, }};")
        lines.append(f"@group(0) @binding({slot}) var<uniform> uniforms: Uniforms;")
        slot += 1
    for name in reads:
        lines.append(
            f"@group(0) @binding({slot}) var<storage, read> {name}: array<f32>;"
        )
        slot += 1
    for name in writes:
        lines.append(
            f"@group(0) @binding({slot}) var<storage, read_write> {name}: array<f32>;"
        )
        slot += 1
    return "\n".join(lines)
```

And update `transpile_compute` to call `_emit_header(uniforms, list(cls._reads), list(cls._writes))`.

Helper + main emitters:

```python
def _build_env(cls: type, sigs, bindings, uniforms, *, fn_params) -> TypeEnv:
    env = TypeEnv()
    env.set_method_sigs(sigs, class_name=cls.__name__)
    for n, t in uniforms.items():
        env.set_uniform(n, t)
    for n, comp_name in bindings.items():
        env.set_binding(n, component_name=comp_name)
    for n, t in fn_params.items():
        env.set_param(n, t)
    return env


def _emit_helper(cls, name, fn, sigs, bindings, uniforms) -> str:
    sig = sigs[name]
    params = {k: v for k, v in sig.items() if k != "return"}
    env = _build_env(cls, sigs, bindings, uniforms, fn_params=params)

    mut = _scan_mutability(fn.body)
    body_lines = []
    for s in fn.body:
        body_lines.append(_emit_stmt(s, env, mut, ast.unparse(s)))
    body = "\n".join(f"  {ln}" for chunk in body_lines for ln in chunk.splitlines())

    param_list = ", ".join(f"{k}: {v}" for k, v in params.items())
    return (
        f"fn _{cls.__name__}_{name}({param_list}) -> {sig['return']} {{\n"
        f"{body}\n}}"
    )


def _emit_main(cls, fn, sigs, bindings, uniforms) -> str:
    env = _build_env(cls, sigs, bindings, uniforms, fn_params={"i": "u32"})
    mut = _scan_mutability(fn.body)
    body_lines = []
    for s in fn.body:
        body_lines.append(_emit_stmt(s, env, mut, ast.unparse(s)))
    body = "\n".join(f"  {ln}" for chunk in body_lines for ln in chunk.splitlines())

    return (
        f"@compute @workgroup_size({cls.workgroup_size})\n"
        f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
        f"  let i: u32 = gid.x;\n"
        f"{body}\n}}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v`
Expected: ALL passing — codegen tests, header tests, helper tests, full-class tests.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/transpile.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): full-class emission — header, helpers, main wrapper

transpile_compute(cls) now produces the complete WGSL source:

- Uniforms struct from class _uniforms.
- @group(0) @binding(K) declarations for the uniform buffer + every
  storage buffer, in the same order as the Phase-1 _bind_group_layout.
- Helper methods emit as `fn _<ClassName>_<name>(...) -> R { ... }`.
- main wraps as `@compute @workgroup_size(W) fn main(@builtin(...))
  { let i: u32 = gid.x; ... }`.

End-to-end transpile of a small Compute class now produces a complete
WGSL string consumable by device.create_shader_module().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Wire `Compute.compile()` default + validate at `engine.compute(cls)`

**Files:**
- Modify: `src/manifoldx/compute/_core.py` (default `compile()` calls transpiler)
- Modify: `src/manifoldx/compute/__init__.py` (re-export `transpile_compute` + `ComputeShaderCompileError`)
- Modify: `src/manifoldx/engine.py` (validate at registration time)
- Test: `tests/test_compute_transpile.py`

- [ ] **Step 1: Write failing test for default `compile()` and registration-time validation**

Append to `tests/test_compute_transpile.py`:

```python
def test_compute_default_compile_uses_transpiler():
    """A Compute subclass without an override now compiles via transpile_compute."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.components import Transform

    class K(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def main(self, i: int):
            return

    wgsl = K().compile()
    assert "@compute @workgroup_size(64)" in wgsl


def test_engine_compute_validates_at_registration():
    """engine.compute(cls) raises ComputeShaderCompileError synchronously for a bogus kernel."""
    from manifoldx.compute import Compute, ReadsWrites
    from manifoldx.compute.transpile import ComputeShaderCompileError
    from manifoldx.components import Transform

    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception:
        pytest.skip("no offscreen wgpu backend available")

    import manifoldx as mx
    engine = mx.Engine("k", canvas=canvas)

    class Bogus(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def main(self, i: int):
            ghost = 1.0  # plain Assign without prior AnnAssign — error.
            return

    with pytest.raises(ComputeShaderCompileError):
        engine.compute(Bogus)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compute_transpile.py -v -k "default_compile_uses or engine_compute_validates"`
Expected: 2 FAILs (`compile()` still raises `NotImplementedError`; engine validates lazily).

- [ ] **Step 3: Wire defaults & validation**

Edit `src/manifoldx/compute/_core.py`. Find `Compute.compile()` and replace its body:

```python
    def compile(self) -> str:
        """Default: trace `main` (and any helper methods) to WGSL via the Phase-2 transpiler.

        Override for hand-written WGSL kernels.
        """
        from manifoldx.compute.transpile import transpile_compute
        return transpile_compute(type(self))
```

Edit `src/manifoldx/compute/__init__.py` to re-export the new names alongside the existing ones:

```python
from manifoldx.compute._core import (
    Compute,
    ComputeRunner,
    Reads,
    ReadsWrites,
    Uniform,
    Writes,
    _AUTO_BOUND_UNIFORMS,
    _DISPATCH_SYMBOLS,
)
from manifoldx.compute import shader
from manifoldx.compute.transpile import (
    ComputeShaderCompileError,
    transpile_compute,
)

__all__ = [
    "Compute",
    "ComputeRunner",
    "Reads",
    "Writes",
    "ReadsWrites",
    "Uniform",
    "shader",
    "ComputeShaderCompileError",
    "transpile_compute",
]
```

Edit `src/manifoldx/engine.py`. Find the `compute(self, cls)` method and make it validate synchronously:

```python
    def compute(self, cls):
        """Register a Compute class. Validates the WGSL synchronously."""
        # Trigger transpile + wgpu validation now so errors surface at
        # registration, not on first frame.
        if self._device is not None:
            instance = cls()
            wgsl = instance.compile()
            try:
                self._device.create_shader_module(code=wgsl)
            except Exception as e:
                from manifoldx.compute.transpile import ComputeShaderCompileError
                raise ComputeShaderCompileError(
                    category="wgpu-validation",
                    message=str(e),
                    filename=getattr(cls, "__module__", "<class>"),
                    line=0, col=0, source_line=None,
                )
        self._compute_runner.register(cls)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compute_transpile.py -v`
Expected: all tests pass.

Run: `uv run pytest tests/test_compute.py tests/test_compute_integration.py -v --tb=short`
Expected: all Phase-1 tests still green.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/compute/_core.py src/manifoldx/compute/__init__.py src/manifoldx/engine.py tests/test_compute_transpile.py
git commit -m "$(cat <<'EOF'
feat(compute): default compile() uses transpiler + validate at engine.compute()

- Compute.compile() default body now calls
  manifoldx.compute.transpile.transpile_compute(type(self)).
- engine.compute(cls) calls instance.compile() and
  device.create_shader_module(code=...) synchronously, wrapping any
  wgpu validation error in ComputeShaderCompileError(category=
  "wgpu-validation"). Kernel authors see errors at registration time,
  not on first frame.
- Compute package re-exports ComputeShaderCompileError and
  transpile_compute alongside the Phase-1 surface.

User-overridden compile() bodies still take precedence — the escape
hatch for hand-tuned kernels (test_compute_integration.py +
examples/nbody_compute.py until Task 14 lands).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Rewrite `nbody_compute.py` + numeric integration test

**Files:**
- Create: `tests/test_compute_transpile_integration.py`
- Modify: `examples/nbody_compute.py` (replace WGSL string with `def main` + `def pair_accel`)

- [ ] **Step 1: Write failing integration test that compares transpiled-Gravity to hand-written-Gravity output**

Create `tests/test_compute_transpile_integration.py`:

```python
"""Integration: transpiled GravityKernel must produce the same numeric
output as the Phase-1 hand-written WGSL on identical inputs.
"""
import numpy as np
import pytest


def _make_offscreen_engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception:
        pytest.skip("no offscreen wgpu backend available")
    import manifoldx as mx
    return mx.Engine("test", canvas=canvas), canvas


def test_transpiled_gravity_matches_hand_written_wgsl_after_one_frame():
    """Same initial state → run one frame each → arrays agree to rtol=1e-5."""
    import manifoldx as mx
    from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
    from manifoldx.compute.shader import vec3, dot, sqrt
    from manifoldx.components import Component, Material, Mesh, Transform
    from manifoldx.types import Float, Vector3
    from manifoldx.resources import BasicMaterial, sphere

    class Velocity(Component):
        vector: Vector3

    class Mass(Component):
        value: Float = 1.0

    HAND_WGSL = """
    struct Uniforms { G: f32, softening: f32, dt: f32, n: f32, };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var<storage, read> masses: array<f32>;
    @group(0) @binding(2) var<storage, read_write> transforms: array<f32>;
    @group(0) @binding(3) var<storage, read_write> velocities: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        let n = u32(uniforms.n);
        if (i >= n) { return; }
        let pos_i = vec3<f32>(transforms[i*10u + 0u], transforms[i*10u + 1u], transforms[i*10u + 2u]);
        var accel = vec3<f32>(0.0);
        for (var j = 0u; j < n; j = j + 1u) {
            if (i == j) { continue; }
            let pos_j = vec3<f32>(transforms[j*10u + 0u], transforms[j*10u + 1u], transforms[j*10u + 2u]);
            let diff = pos_j - pos_i;
            let r2 = dot(diff, diff) + uniforms.softening * uniforms.softening;
            let inv_r3 = 1.0 / (r2 * sqrt(r2));
            accel = accel + uniforms.G * masses[j] * diff * inv_r3;
        }
        var vel = vec3<f32>(velocities[i*3u + 0u], velocities[i*3u + 1u], velocities[i*3u + 2u]);
        vel = vel + accel * uniforms.dt;
        velocities[i*3u + 0u] = vel.x;
        velocities[i*3u + 1u] = vel.y;
        velocities[i*3u + 2u] = vel.z;
        transforms[i*10u + 0u] = pos_i.x + vel.x * uniforms.dt;
        transforms[i*10u + 1u] = pos_i.y + vel.y * uniforms.dt;
        transforms[i*10u + 2u] = pos_i.z + vel.z * uniforms.dt;
    }
    """.strip()

    class HandWritten(Compute):
        transforms: ReadsWrites[Transform]
        masses:     Reads[Mass]
        velocities: ReadsWrites[Velocity]
        G:         Uniform[float] = 20.0
        softening: Uniform[float] = 0.05
        dt:        Uniform[float] = "frame_dt"
        n:         Uniform[float] = "entity_count"
        workgroup_size = 64
        dispatch = "entity_count"
        def compile(self) -> str:
            return HAND_WGSL

    class Transpiled(Compute):
        transforms: ReadsWrites[Transform]
        masses:     Reads[Mass]
        velocities: ReadsWrites[Velocity]
        G:         Uniform[float] = 20.0
        softening: Uniform[float] = 0.05
        dt:        Uniform[float] = "frame_dt"
        n:         Uniform[float] = "entity_count"
        workgroup_size = 64
        dispatch = "entity_count"
        def pair_accel(self, pos_i: vec3, pos_j: vec3, m_j: float) -> vec3:
            diff: vec3 = pos_j - pos_i
            r2: float = dot(diff, diff) + self.softening * self.softening
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

    rng = np.random.default_rng(7)
    N = 16
    positions = rng.uniform(-2.0, 2.0, size=(N, 3)).astype(np.float32)
    mass_vals = rng.uniform(0.5, 3.0, size=N).astype(np.float32)
    initial_vel = np.zeros((N, 3), dtype=np.float32)

    def run(kernel_cls):
        engine, _canvas = _make_offscreen_engine()
        engine.spawn(
            Mesh(sphere(0.5, 12)),
            Material(BasicMaterial("#fff")),
            Transform(pos=positions),
            Velocity(vector=initial_vel),
            Mass(value=mass_vals),
            n=N,
        )
        engine.compute(kernel_cls)
        engine.run_one_frame()  # advances exactly one CPU+compute+render cycle
        store = engine.store
        t_data = store.read("Transform").copy()
        v_data = store.read("Velocity").copy()
        return t_data, v_data

    t_hand, v_hand = run(HandWritten)
    t_trans, v_trans = run(Transpiled)

    np.testing.assert_allclose(t_hand[:, :3], t_trans[:, :3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v_hand,         v_trans,        rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run the test to verify it fails — likely missing `engine.run_one_frame` or `engine.store.read`**

Run: `uv run pytest tests/test_compute_transpile_integration.py -v --tb=short`
Expected: FAIL — either skip (no GPU) or numeric mismatch / missing helper. Inspect output.

If `engine.run_one_frame()` doesn't exist, locate the existing per-frame entrypoint with: `grep -n "def run\|def _frame\|def step" src/manifoldx/engine.py | head` — the existing Phase-1 integration test in `tests/test_compute_integration.py` does this (look at lines 59-100). Copy whatever idiom it uses.

- [ ] **Step 3: Adapt the test to whichever per-frame entrypoint exists**

Replace `engine.run_one_frame()` and `engine.store.read("Transform")` with the exact idioms from `tests/test_compute_integration.py`. The test must succeed with both kernels producing identical floats within `rtol=1e-5`.

If the integration test passes with the transpiled kernel matching the hand-written kernel, the transpiler is end-to-end correct.

- [ ] **Step 4: Rewrite `examples/nbody_compute.py`**

Replace the file body. New content (write whole file, replacing the existing contents):

```python
"""N-Body simulation with GPU compute shader.

A side-by-side counterpart to examples/nbody.py: same physics, same
visuals, but the all-pairs O(N²) gravity loop runs in a WGSL compute
shader instead of pure-numpy on CPU.

Demonstrates the Phase-2 Python-as-shader DSL: GravityKernel.main is
plain typed Python; the engine traces it to WGSL on engine.compute(...)
via manifoldx.compute.transpile.

Phase 1 (raw WGSL via compile() override) still works — see the test
suite for an explicit hand-written counterpart.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
from manifoldx.compute.shader import vec3, dot, sqrt
from manifoldx.resources import PhongMaterial, sphere
from manifoldx.types import Float, Vector3


# ── Custom components ────────────────────────────────────────────────────────
class Velocity(Component):
    """Per-entity velocity vector (3 floats)."""
    vector: Vector3


class Mass(Component):
    """Per-entity gravitational mass (1 float)."""
    value: Float = 1.0


# ── Simulation parameters ────────────────────────────────────────────────────
NUM_BODIES = 500
G = 20.0
SOFTENING = 0.05
SPHERE_RADIUS = 0.5
SIZE = 5 * NUM_BODIES ** (1 / 3)


# ── GPU compute kernel — plain typed Python ──────────────────────────────────
class GravityKernel(Compute):
    """N-body gravity + velocity-and-position integration on the GPU."""

    transforms: ReadsWrites[Transform]
    masses:     Reads[Mass]
    velocities: ReadsWrites[Velocity]

    G:         Uniform[float] = G
    softening: Uniform[float] = SOFTENING
    dt:        Uniform[float] = "frame_dt"
    n:         Uniform[float] = "entity_count"

    workgroup_size = 64
    dispatch = "entity_count"

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


# ── Engine setup ─────────────────────────────────────────────────────────────
engine = mx.Engine("N-Body Compute (transpiled)")
engine.camera.fit(SIZE)

positions = mx.random.positions_in_box(NUM_BODIES, half_size=SIZE, rng=7)
mass_values = mx.random.scalars_uniform(NUM_BODIES, low=0.5, high=3.0, rng=7)
scales = (mass_values ** (1 / 3)).reshape(-1, 1)
initial_velocities = np.zeros((NUM_BODIES, 3), dtype=np.float32)

engine.spawn(
    Mesh(sphere(SPHERE_RADIUS, 12)),
    Material(PhongMaterial("#ffaa44")),
    Transform(pos=positions, scale=scales),
    Velocity(vector=initial_velocities),
    Mass(value=mass_values),
    n=NUM_BODIES,
)

engine.compute(GravityKernel)


if __name__ == "__main__":
    engine.cli()
```

Note: `u32` must be importable as a name inside the kernel-body scope. The transpiler treats it specially as a cast (Task 6), so it doesn't need to actually be importable in Python — but kernel authors *write* it as if it were. Confirm by adding `from manifoldx.compute.shader import u32` if the linter complains. If `u32` isn't yet exported from `shader.py`, add it as another `@_shader_only` function.

If `u32` is missing from `shader.py`, edit `src/manifoldx/compute/shader.py` and add (alongside vec3/vec4):

```python
@_shader_only
def u32(x):
    """WGSL u32 cast."""

@_shader_only
def i32(x):
    """WGSL i32 cast."""

@_shader_only
def f32(x):
    """WGSL f32 cast."""
```

And update the `BUILTINS` set + `__all__` at the bottom of `shader.py` accordingly. (Actually these are *casts* not builtins — the transpiler handles them via `_CASTS` in Task 6 — but exposing them in `shader` makes the imports work.)

- [ ] **Step 5: Smoke-render the demo to confirm it runs**

Run: `uv run python examples/nbody_compute.py --render --duration 1 --fps 30 --output /tmp/nbody_compute_phase2.mp4`
Expected: video produced; final frame shows bodies clustering under gravity.

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add tests/test_compute_transpile_integration.py examples/nbody_compute.py src/manifoldx/compute/shader.py
git commit -m "$(cat <<'EOF'
feat(compute): nbody_compute uses Phase-2 transpiled kernel + integration test

- examples/nbody_compute.py: GravityKernel.main is plain typed Python;
  the engine traces it to WGSL via manifoldx.compute.transpile on
  engine.compute(GravityKernel). pair_accel(pos_i, pos_j, m_j) -> vec3
  factors the inner-loop math, mirroring how nbody.py factors via
  mx.physics.gravity.
- shader.py: expose f32 / i32 / u32 cast sentinels alongside vec3/vec4
  for kernel-body imports.
- tests/test_compute_transpile_integration.py: side-by-side run of
  hand-written WGSL Gravity vs. transpiled-Python Gravity on identical
  inputs; arrays must agree to rtol=1e-5 after one frame.

Closes the Phase-2 v1 success criterion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: CHANGELOG `[Unreleased]` entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the `[Unreleased]` entry**

Edit `CHANGELOG.md`. Find the line `## [Unreleased]` and add a `### Features` block immediately under it (preserving any existing content):

```markdown
## [Unreleased]

### Features

- **Compute systems Phase 2 — Python → WGSL shader compiler** — `Compute.compile()`'s default body now traces a typed-Python `def main(self, i)` body to WGSL via `manifoldx.compute.transpile`. Kernel authors write plain Python with PEP-526 annotations on every local; the transpiler emits the bind-group header, helper functions (`fn _<ClassName>_<name>`), and the `@compute @workgroup_size(W) fn main(...)` wrapper. `examples/nbody_compute.py` is now a Compute subclass with a `pair_accel(...) -> vec3` helper method instead of an inlined WGSL string; numerics agree with the Phase-1 hand-written kernel within `rtol=1e-5`.
- **`engine.compute(cls)` validates synchronously** — WGSL is compiled via `device.create_shader_module(...)` at registration time. Errors surface as `ComputeShaderCompileError` (category `wgpu-validation`) before any frame runs, alongside Python-source-level transpiler errors (`unsupported-construct`, `missing-annotation`, `unknown-name`, `type-mismatch`, `implicit-promotion`, `recursion`).
- **Per-Component `_layout` table** — `Component.__init_subclass__` now derives `{field_name: (offset_in_floats, length_in_floats)}` from the existing field annotations. Pre-base-class built-ins (`Transform`, `Mesh`, `Material`) ship explicit `_layout` class attrs. The transpiler reads this table to emit storage-buffer offset arithmetic without component-specific knowledge.

### Refactors

- **`compile()` override is the escape hatch, not the default.** Phase-1 user kernels that override `compile()` continue to work unchanged; only the default body changed.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): record compute Phase 2 — Python → WGSL transpiler

[Unreleased] entry covering the Phase-2 surface: typed-Python kernels
via transpile_compute, registration-time WGSL validation, per-Component
_layout table, and the nbody_compute demo rewrite.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Run the full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all tests green, including `test_compute.py` (Phase-1, 12 tests), `test_compute_integration.py` (Phase-1, 3 tests), `test_compute_transpile.py` (Phase-2 unit tests), and `test_compute_transpile_integration.py` (Phase-2 numeric parity).

- [ ] **Smoke-render the demo**

Run: `uv run python examples/nbody_compute.py --render --duration 2 --fps 30 --output /tmp/nbody_compute_smoke.mp4`
Expected: 60-frame mp4; final frame shows bodies clustering under gravity (visually identical to the Phase-1 demo).

- [ ] **Lint**

Run: `uv run ruff check src/manifoldx/compute/`
Expected: no new violations.

- [ ] **Push when ready**

```bash
git status   # confirm clean
git log origin/main..HEAD --oneline   # review the new commits
git push origin main
```
