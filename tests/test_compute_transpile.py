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

    with pytest.raises(ComputeShaderCompileError, match="must define a `main"):
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
    from manifoldx.components import Component
    from manifoldx.types import Float

    class Mass(Component):
        value: Float

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
    from manifoldx.components import Component
    from manifoldx.types import Float
    import ast as _ast

    class Mass(Component):
        value: Float

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
    from manifoldx.types import Float

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
    engine = mx.Engine("k", width=64, height=64)
    engine._init_canvas(canvas)

    class Bogus(Compute):
        transforms: ReadsWrites[Transform]
        workgroup_size = 64
        dispatch = "entity_count"
        def main(self, i: int):
            ghost = 1.0  # plain Assign without prior AnnAssign — error.
            return

    with pytest.raises(ComputeShaderCompileError):
        engine.compute(Bogus)
