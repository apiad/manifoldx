"""Rigid GPU skinning: per-entity joint blocks and the skinned vertex-shader transform.

A skinned geometry carries one joint index per vertex. Each posed entity owns a
block of MAX_JOINTS matrices in one storage buffer; the mesh pass uploads a
block index per instance, and the vertex shader moves position and normal by
joint_mats[block * MAX_JOINTS + joint] before the material's own code runs.
"""

import re

import numpy as np

MAX_JOINTS = 32

JOINT_LOCATION = 7  # vertex attribute location of the joint index
JOINTS_BINDING = 8  # group(0) storage: array<mat4x4<f32>>, all joint blocks
BLOCKS_BINDING = 9  # group(0) storage: array<u32>, block index per instance

_BLOCK_BYTES = MAX_JOINTS * 64
_IDENTITY_BLOCK = np.tile(np.eye(4, dtype=np.float32), (MAX_JOINTS, 1, 1))


def validate_joints(joints, n_vertices: int) -> np.ndarray:
    """Return joints as (N,) uint32 or raise ValueError."""
    j = np.asarray(joints)
    if j.ndim != 1 or len(j) != n_vertices:
        raise ValueError(f"joints must have shape ({n_vertices},), got {j.shape}")
    if not np.issubdtype(j.dtype, np.integer):
        raise ValueError(f"joints must be integers, got dtype {j.dtype}")
    if len(j) and (j.min() < 0 or j.max() >= MAX_JOINTS):
        raise ValueError(f"joints must be in [0, {MAX_JOINTS}), got [{j.min()}, {j.max()}]")
    return np.ascontiguousarray(j, dtype=np.uint32)


def append_joint_column(interleaved: np.ndarray, joints) -> np.ndarray:
    """(N, k) float32 vertex rows -> (N, k + 1), the last column the uint32 joint bits."""
    out = np.empty((len(interleaved), interleaved.shape[1] + 1), dtype=np.float32)
    out[:, :-1] = interleaved
    out[:, -1] = np.asarray(joints, dtype=np.uint32).view(np.float32)
    return out


class JointStore:
    """Joint blocks of every posed entity: a CPU mirror and its GPU storage buffer.

    Block 0 is all identity and never handed out; an entity that was never
    posed reads it. Matrices are stored transposed, so WGSL reads them as
    column-major mat4.
    """

    def __init__(self, max_entities: int):
        self.block_of = np.zeros(max_entities, dtype=np.uint32)
        self.mirror = np.tile(_IDENTITY_BLOCK, (2, 1, 1, 1))  # (blocks, MAX_JOINTS, 4, 4)
        self._next = 1
        self._free: list[int] = []
        self._device = None
        self.buffer = None
        self._buffer_blocks = 0

    def _current(self) -> bool:
        return self.buffer is not None and self._buffer_blocks == len(self.mirror)

    def _alloc(self) -> int:
        if self._free:
            return self._free.pop()
        if self._next == len(self.mirror):
            self.mirror = np.concatenate([self.mirror, np.tile(_IDENTITY_BLOCK, (len(self.mirror), 1, 1, 1))])
        self._next += 1
        return self._next - 1

    def set(self, entity: int, matrices: np.ndarray):
        """Write (J, 4, 4) row-major matrices into the entity's block; upload J*64 bytes."""
        block = int(self.block_of[entity])
        if block == 0:
            block = self._alloc()
            self.block_of[entity] = block
        n = len(matrices)
        self.mirror[block, :n] = matrices.transpose(0, 2, 1)
        if self._current():
            self._device.queue.write_buffer(self.buffer, block * _BLOCK_BYTES, self.mirror[block, :n].tobytes())

    def release(self, entities):
        """Free the blocks of destroyed or respawned entities, reset to identity."""
        entities = np.asarray(entities, dtype=np.int64)
        blocks = self.block_of[entities]
        blocks = np.unique(blocks[blocks > 0])
        if len(blocks) == 0:
            return
        self.block_of[entities] = 0
        for b in blocks.tolist():
            self.mirror[b] = _IDENTITY_BLOCK
            if self._current():
                self._device.queue.write_buffer(self.buffer, b * _BLOCK_BYTES, _IDENTITY_BLOCK.tobytes())
            self._free.append(b)

    def ensure_buffer(self, device):
        """The storage buffer holding every block; (re)created with the full mirror when it grew."""
        import wgpu

        if not self._current():
            self._device = device
            self.buffer = device.create_buffer(
                size=self.mirror.nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            device.queue.write_buffer(self.buffer, 0, self.mirror.tobytes())
            self._buffer_blocks = len(self.mirror)
        return self.buffer


_STRUCT = re.compile(r"struct VertexInput\s*\{(?P<body>.*?)\};", re.S)
_ENTRY = re.compile(r"fn vs_main\(in: VertexInput\)(?P<ret>[^{]*)\{")


def skin_shader(source: str) -> str:
    """Rewrite a mesh material's WGSL so vs_main sees joint-posed position and normal.

    Expects `struct VertexInput` with a `position` field and an instance_index
    builtin, and an entry point `fn vs_main(in: VertexInput)`.
    """
    struct = _STRUCT.search(source)
    entries = _ENTRY.findall(source)
    if struct is None or len(entries) != 1:
        raise ValueError(
            "skinning needs a WGSL `struct VertexInput { ... };` and one "
            "`fn vs_main(in: VertexInput)` entry point"
        )
    body = struct.group("body")
    instance = re.search(r"@builtin\(instance_index\)\s*(\w+)\s*:", body)
    if instance is None or not re.search(r"\bposition\s*:", body):
        raise ValueError("skinning needs VertexInput.position and an @builtin(instance_index) field")
    if f"@location({JOINT_LOCATION})" in body:
        raise ValueError(f"VertexInput already uses @location({JOINT_LOCATION}), reserved for joints")
    has_normal = re.search(r"\bnormal\s*:", body) is not None

    fields = body.rstrip()
    if not fields.endswith(","):
        fields += ","
    fields += f"\n    @location({JOINT_LOCATION}) joint: u32,\n"
    out = source[: struct.start("body")] + fields + source[struct.end("body"):]
    out = _ENTRY.sub(
        lambda m: f"fn vs_main(in_raw: VertexInput){m.group('ret')}{{\n    var in = skin_vertex(in_raw);",
        out,
        count=1,
    )
    normal = "    o.normal = normalize(skin_normal_matrix(m) * v.normal);\n" if has_normal else ""
    return out + f"""
@group(0) @binding({JOINTS_BINDING}) var<storage, read> skin_joints: array<mat4x4<f32>>;
@group(0) @binding({BLOCKS_BINDING}) var<storage, read> skin_blocks: array<u32>;

// Inverse transpose of m's 3x3 up to a positive scale: the cofactor matrix times
// sign(det). A singular 3x3 keeps the normal as it is, so nothing divides by zero.
// Singular is relative to the columns' lengths, so a small uniform scale is not.
fn skin_normal_matrix(m: mat4x4<f32>) -> mat3x3<f32> {{
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    let det = dot(a, cross(b, c));
    if abs(det) <= 1e-6 * length(a) * length(b) * length(c) {{
        return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    }}
    return mat3x3<f32>(cross(b, c), cross(c, a), cross(a, b)) * sign(det);
}}

fn skin_vertex(v: VertexInput) -> VertexInput {{
    var o = v;
    let m = skin_joints[skin_blocks[v.{instance.group(1)}] * {MAX_JOINTS}u + v.joint];
    o.position = (m * vec4<f32>(v.position, 1.0)).xyz;
{normal}    return o;
}}
"""
