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
