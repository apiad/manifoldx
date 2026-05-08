"""Shader-builtin sentinels for the Compute Python-DSL (Phase 2).

These functions are recognized by name in the AST walker that compiles
`Compute.main` to WGSL. They have Python implementations that raise on
call, so accidentally invoking them outside a compiled kernel produces a
clear error message.

Recognized in v1:
- Constructors: `vec3(x, y, z)`, `vec4(x, y, z, w)`.
- Geometry: `length(v)`, `dot(a, b)`, `cross(a, b)`, `normalize(v)`.
- Numeric: `sqrt(x)`, `min(a, b)`, `max(a, b)`, `clamp(x, lo, hi)`,
  `abs(x)`, `pow(x, y)`, `floor(x)`, `ceil(x)`.
"""
from __future__ import annotations


_SHADER_ONLY = (
    "{name}() is a shader primitive — only callable inside a "
    "Compute.main body that gets compiled to WGSL by the Phase-2 "
    "code generator. Outside a kernel, use the equivalent numpy or "
    "math operation."
)


def _shader_only(fn):
    fn.__doc__ = (fn.__doc__ or "") + "\n\nShader primitive — see manifoldx.compute.shader."

    def _raise(*args, **kwargs):
        raise NotImplementedError(_SHADER_ONLY.format(name=fn.__name__))

    _raise.__name__ = fn.__name__
    _raise.__doc__ = fn.__doc__
    return _raise


@_shader_only
def vec3(x, y, z):
    """3-component float vector. WGSL: `vec3<f32>(x, y, z)`."""


@_shader_only
def vec4(x, y, z, w):
    """4-component float vector. WGSL: `vec4<f32>(x, y, z, w)`."""


@_shader_only
def length(v):
    """Vector magnitude. WGSL: `length(v)`."""


@_shader_only
def dot(a, b):
    """Vector dot product. WGSL: `dot(a, b)`."""


@_shader_only
def cross(a, b):
    """Vector cross product (3D). WGSL: `cross(a, b)`."""


@_shader_only
def normalize(v):
    """Unit-length vector in v's direction. WGSL: `normalize(v)`."""


@_shader_only
def sqrt(x):
    """Element-wise square root. WGSL: `sqrt(x)`."""


@_shader_only
def pow(x, y):
    """Element-wise x^y. WGSL: `pow(x, y)`."""


@_shader_only
def floor(x):
    """Element-wise floor. WGSL: `floor(x)`."""


@_shader_only
def ceil(x):
    """Element-wise ceil. WGSL: `ceil(x)`."""


@_shader_only
def abs(x):
    """Element-wise absolute value. WGSL: `abs(x)`."""


@_shader_only
def min(a, b):
    """Element-wise minimum. WGSL: `min(a, b)`."""


@_shader_only
def max(a, b):
    """Element-wise maximum. WGSL: `max(a, b)`."""


@_shader_only
def clamp(x, lo, hi):
    """Element-wise clamp. WGSL: `clamp(x, lo, hi)`."""


# WGSL numeric casts. The transpiler emits these as bare WGSL casts
# (see _CASTS in transpile.py); the Python sentinels exist so kernel
# authors can write `u32(x)` / `i32(x)` / `f32(x)` without lint
# complaining about an undefined name. Calling at runtime raises.


@_shader_only
def u32(x):
    """WGSL u32 cast. Inside a kernel only."""


@_shader_only
def i32(x):
    """WGSL i32 cast. Inside a kernel only."""


@_shader_only
def f32(x):
    """WGSL f32 cast. Inside a kernel only."""


# Set of recognized shader-builtin names. The AST walker uses this to
# identify calls that should map to WGSL primitives.
BUILTINS = {
    "vec3", "vec4",
    "length", "dot", "cross", "normalize",
    "sqrt", "pow", "floor", "ceil", "abs",
    "min", "max", "clamp",
}


__all__ = list(BUILTINS) + ["BUILTINS", "u32", "i32", "f32"]
