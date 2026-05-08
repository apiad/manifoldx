"""Shader-builtin sentinels for the Compute Python-DSL (Phase 2).

These names are recognized by the AST walker that compiles
`Compute.main` to WGSL. They have Python implementations that raise on
call, so accidentally invoking them outside a compiled kernel produces
a clear error message — and they carry full PEP-484 signatures so
type checkers see correct return types and operator overloads when
reading kernel bodies.

Recognized in v1:
- Constructors: `vec3(x, y, z)`, `vec4(x, y, z, w)`.
- Geometry: `length(v)`, `dot(a, b)`, `cross(a, b)`, `normalize(v)`.
- Numeric: `sqrt(x)`, `pow(x, y)`, `floor(x)`, `ceil(x)`, `abs(x)`,
  `min(a, b)`, `max(a, b)`, `clamp(x, lo, hi)`.
- Casts: `u32(x)`, `i32(x)`, `f32(x)`.
"""
from __future__ import annotations

from typing import NoReturn, TypeVar, Union


_SHADER_ONLY = (
    "{name}() is a shader primitive — only callable inside a "
    "Compute.main body that gets compiled to WGSL by the Phase-2 "
    "code generator. Outside a kernel, use the equivalent numpy or "
    "math operation."
)


def _raise(name: str) -> NoReturn:
    raise NotImplementedError(_SHADER_ONLY.format(name=name))


# ─────────────────────────────────────────────────────────────────────────────
# vec3 / vec4: dual-role types — annotation tags AND callable constructors.
# Real classes so static type checkers can resolve `pos: vec3` as a type
# and `pos_j - pos_i` against the arithmetic dunders.
# Calling a constructor or any dunder at runtime raises (transpile-time
# only). Static type checkers don't trace the body — they see signatures.
# ─────────────────────────────────────────────────────────────────────────────


class vec3:
    """3-component float vector. WGSL: `vec3<f32>`. Shader primitive."""

    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        _raise("vec3")

    def __add__(self, other: "vec3") -> "vec3": _raise("vec3.__add__")
    def __sub__(self, other: "vec3") -> "vec3": _raise("vec3.__sub__")
    def __mul__(self, other: "Union[vec3, float]") -> "vec3": _raise("vec3.__mul__")
    def __rmul__(self, other: float) -> "vec3": _raise("vec3.__rmul__")
    def __truediv__(self, other: "Union[vec3, float]") -> "vec3": _raise("vec3.__truediv__")
    def __neg__(self) -> "vec3": _raise("vec3.__neg__")
    def __iadd__(self, other: "vec3") -> "vec3": _raise("vec3.__iadd__")
    def __isub__(self, other: "vec3") -> "vec3": _raise("vec3.__isub__")
    def __imul__(self, other: "Union[vec3, float]") -> "vec3": _raise("vec3.__imul__")


class vec4:
    """4-component float vector. WGSL: `vec4<f32>`. Shader primitive."""

    x: float
    y: float
    z: float
    w: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 0.0) -> None:
        _raise("vec4")

    def __add__(self, other: "vec4") -> "vec4": _raise("vec4.__add__")
    def __sub__(self, other: "vec4") -> "vec4": _raise("vec4.__sub__")
    def __mul__(self, other: "Union[vec4, float]") -> "vec4": _raise("vec4.__mul__")
    def __rmul__(self, other: float) -> "vec4": _raise("vec4.__rmul__")
    def __truediv__(self, other: "Union[vec4, float]") -> "vec4": _raise("vec4.__truediv__")
    def __neg__(self) -> "vec4": _raise("vec4.__neg__")
    def __iadd__(self, other: "vec4") -> "vec4": _raise("vec4.__iadd__")
    def __isub__(self, other: "vec4") -> "vec4": _raise("vec4.__isub__")


# ─────────────────────────────────────────────────────────────────────────────
# Math builtins — typed signatures so kernel-body arithmetic checks.
# Each raises NotImplementedError at runtime; static checkers see the types.
# ─────────────────────────────────────────────────────────────────────────────


_VecLike = Union[vec3, vec4]
_T = TypeVar("_T", float, vec3, vec4)


def length(v: _VecLike) -> float:
    """Vector magnitude. WGSL: `length(v)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="length"))


def dot(a: _VecLike, b: _VecLike) -> float:
    """Vector dot product. WGSL: `dot(a, b)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="dot"))


def cross(a: vec3, b: vec3) -> vec3:
    """Vector cross product (3D). WGSL: `cross(a, b)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="cross"))


def normalize(v: _T) -> _T:
    """Unit-length vector in v's direction. WGSL: `normalize(v)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="normalize"))


def sqrt(x: float) -> float:
    """Element-wise square root. WGSL: `sqrt(x)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="sqrt"))


def pow(x: float, y: float) -> float:
    """Element-wise x^y. WGSL: `pow(x, y)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="pow"))


def floor(x: _T) -> _T:
    """Element-wise floor. WGSL: `floor(x)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="floor"))


def ceil(x: _T) -> _T:
    """Element-wise ceil. WGSL: `ceil(x)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="ceil"))


def abs(x: _T) -> _T:
    """Element-wise absolute value. WGSL: `abs(x)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="abs"))


def min(a: _T, b: _T) -> _T:
    """Element-wise minimum. WGSL: `min(a, b)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="min"))


def max(a: _T, b: _T) -> _T:
    """Element-wise maximum. WGSL: `max(a, b)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="max"))


def clamp(x: _T, lo: _T, hi: _T) -> _T:
    """Element-wise clamp. WGSL: `clamp(x, lo, hi)`."""
    raise NotImplementedError(_SHADER_ONLY.format(name="clamp"))


# ─────────────────────────────────────────────────────────────────────────────
# Numeric casts — kernel authors write `u32(x)` / `i32(x)` / `f32(x)` and
# the transpiler emits a WGSL cast (see _CASTS in transpile.py).
# ─────────────────────────────────────────────────────────────────────────────


def u32(x: float) -> int:
    """WGSL u32 cast. Inside a kernel only."""
    raise NotImplementedError(_SHADER_ONLY.format(name="u32"))


def i32(x: float) -> int:
    """WGSL i32 cast. Inside a kernel only."""
    raise NotImplementedError(_SHADER_ONLY.format(name="i32"))


def f32(x: float) -> float:
    """WGSL f32 cast. Inside a kernel only."""
    raise NotImplementedError(_SHADER_ONLY.format(name="f32"))


# Set of recognized shader-builtin names. The AST walker uses this to
# identify calls that should map to WGSL primitives.
BUILTINS = {
    "vec3", "vec4",
    "length", "dot", "cross", "normalize",
    "sqrt", "pow", "floor", "ceil", "abs",
    "min", "max", "clamp",
}


__all__ = list(BUILTINS) + ["BUILTINS", "u32", "i32", "f32"]
