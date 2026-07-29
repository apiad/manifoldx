"""Transpile the composable `Field` algebra to a WGSL shader function.

A `Field` built from the sources/combinators in `fields` carries a symbolic
`_ast`; `field_to_wgsl` walks it and emits a WGSL function

    fn <name>(P: vec3<f32>) -> f32 { ... }

plus a shared noise prelude, so the *same* developer-composed field can drive a
GPU shader (displacement, detail-normals, colouring) instead of only baking on
the CPU. The GPU noise is a fast value-noise approximation — it is decorative
and does not reproduce the CPU permutation-table Perlin bit-for-bit, but the
composition (octaves, warp, remap, ridged/billow shaping, arithmetic) matches.

Sources supported: perlin, fbm, ridged, billow, constant, coord, distance.
Combinators: + - * / neg, min/max/mix, clamp, remap, abs, power, scale, bias,
shift, warp. Fields with no AST (hand-written callables) or unsupported sources
(e.g. worley) raise ``FieldNotTranspilable``.
"""

from __future__ import annotations


class FieldNotTranspilable(ValueError):
    """Raised when a Field has no AST or uses a node with no WGSL emitter."""


# Value-noise prelude: hash -> trilinear value noise -> fbm/ridged/billow octave
# loops. Signed per octave ((v*2-1)) to sit around zero like the CPU noise.
WGSL_NOISE_PRELUDE = """
fn _fhash13(p_in: vec3<f32>) -> f32 {
    var p3 = fract(p_in * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yxz + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn _fvnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = _fhash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = _fhash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = _fhash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = _fhash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = _fhash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = _fhash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = _fhash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = _fhash13(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn _foffset(seed: f32) -> vec3<f32> {
    return vec3<f32>(seed * 0.1731, seed * 0.9375, seed * 0.5723);
}
// shape: 0 = signed value (perlin/fbm), 1 = ridged (1-|v|)^2, 2 = billow |v|
fn _foctaves(p0: vec3<f32>, seed: f32, freq: f32, octaves: i32,
             lac: f32, gain: f32, shape: i32) -> f32 {
    var q = p0 * freq + _foffset(seed);
    var amp = 1.0;
    var norm = 0.0;
    var total = 0.0;
    for (var i: i32 = 0; i < octaves; i = i + 1) {
        let v = _fvnoise(q) * 2.0 - 1.0;
        var s = v;
        if (shape == 1) { let a = 1.0 - abs(v); s = a * a; }
        if (shape == 2) { s = abs(v); }
        total = total + amp * s;
        norm = norm + amp;
        q = q * lac;
        amp = amp * gain;
    }
    return total / max(norm, 1e-6);
}
""".strip()


def _f(x: float) -> str:
    return f"{float(x):.6f}"


class _Emitter:
    def __init__(self):
        self.lines: list[str] = []
        self._n = 0

    def _fresh(self, prefix: str) -> str:
        self._n += 1
        return f"_{prefix}{self._n}"

    def emit(self, node, pexpr: str) -> str:
        if node is None:
            raise FieldNotTranspilable("field has no AST (hand-written callable)")
        op = node[0]

        if op == "const":
            return _f(node[1])
        if op == "coord":
            return f"({pexpr}).{node[1]}"
        if op == "dist":
            cx, cy, cz = node[1]
            return f"length({pexpr} - vec3<f32>({_f(cx)}, {_f(cy)}, {_f(cz)}))"

        if op in ("perlin", "fbm", "ridged", "billow"):
            if op == "perlin":
                seed, freq = node[1], node[2]
                oct_, lac, gain, shape = 1, 2.0, 0.5, 0
            else:
                _, seed, freq, oct_, lac, gain = node
                shape = {"fbm": 0, "ridged": 1, "billow": 2}[op]
            return (f"_foctaves({pexpr}, {_f(seed)}, {_f(freq)}, {int(oct_)}, "
                    f"{_f(lac)}, {_f(gain)}, {shape})")

        if op in ("add", "sub", "mul", "div", "min", "max"):
            a = self.emit(node[1], pexpr)
            b = self.emit(node[2], pexpr)
            if op == "min":
                return f"min({a}, {b})"
            if op == "max":
                return f"max({a}, {b})"
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
            return f"({a} {sym} {b})"
        if op == "neg":
            return f"(-{self.emit(node[1], pexpr)})"
        if op == "abs":
            return f"abs({self.emit(node[1], pexpr)})"
        if op == "pow":
            return f"pow({self.emit(node[1], pexpr)}, {_f(node[2])})"
        if op == "clamp":
            return f"clamp({self.emit(node[1], pexpr)}, {_f(node[2])}, {_f(node[3])})"
        if op == "mix":
            a = self.emit(node[1], pexpr)
            b = self.emit(node[2], pexpr)
            t = self.emit(node[3], pexpr)
            return f"mix({a}, {b}, {t})"
        if op == "remap":
            e = self.emit(node[1], pexpr)
            a, b, c, d = node[2], node[3], node[4], node[5]
            return (f"({_f(c)} + ({e} - {_f(a)}) * ({_f(d)} - {_f(c)}) "
                    f"/ ({_f(b)} - {_f(a)}))")
        if op == "shift":
            ox, oy, oz = node[2]
            newp = f"({pexpr} + vec3<f32>({_f(ox)}, {_f(oy)}, {_f(oz)}))"
            return self.emit(node[1], newp)
        if op == "warp":
            _, base, fx, fy, fz, amount = node
            ex = self.emit(fx, pexpr)
            ey = self.emit(fy, pexpr)
            ez = self.emit(fz, pexpr)
            pv = self._fresh("p")
            self.lines.append(
                f"    let {pv} = {pexpr} + vec3<f32>({ex}, {ey}, {ez}) * {_f(amount)};"
            )
            return self.emit(base, pv)

        raise FieldNotTranspilable(f"no WGSL emitter for node '{op}'")


def field_to_wgsl(field, name: str = "sample_field", with_prelude: bool = True) -> str:
    """Emit a WGSL `fn <name>(P: vec3<f32>) -> f32` evaluating `field`.

    Set `with_prelude=False` to omit the shared noise helpers (include
    `WGSL_NOISE_PRELUDE` once yourself when emitting several fields).
    """
    ast = getattr(field, "_ast", None)
    em = _Emitter()
    expr = em.emit(ast, "P")
    body = "\n".join(em.lines)
    fn = (f"fn {name}(P: vec3<f32>) -> f32 {{\n"
          + (body + "\n" if body else "")
          + f"    return {expr};\n}}")
    return (WGSL_NOISE_PRELUDE + "\n\n" + fn) if with_prelude else fn
