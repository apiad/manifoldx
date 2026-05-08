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
        # self.<binding>[idx].<field> — indexed component field access.
        if (
            isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Attribute)
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id == "self"
        ):
            binding_name = node.value.value.attr
            component_name = env.lookup_binding(binding_name)
            from manifoldx.components import _COMPONENT_CLASSES, Material, Mesh, Transform
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

        # self.<uniform> — fall through to uniform branch.
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
                emit_name = ret_type if fname in {"vec3", "vec4"} else fname
                return f"{emit_name}({', '.join(arg_texts)})", ret_type

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
