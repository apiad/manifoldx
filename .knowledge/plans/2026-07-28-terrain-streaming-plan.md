# Terrain Streaming + Background Tasks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Real-time CPU terrain via background-generated patches swapped into a recycled slot pool, plus the reusable primitives. Design: `.knowledge/analysis/2026-07-28-terrain-streaming-and-background-tasks-design.md`.

**Tech Stack:** Python 3.13+, numpy, wgpu, `uv`. Tests: `.venv/bin/python -m pytest`.

## Global Constraints

- Pure numpy where applicable; wgpu only in the fog/GPU paths. Additive — existing scalar/textured/vcolor render paths and the current 670 tests must keep passing.
- Conventional commits `feat(...)`. Drop the proposed `camera.look_from` — `camera.set_pose(position, target)` already exists; use it.
- Demo imports lights/materials from `manifoldx.resources` (not top-level `mx`).

## Grounded symbols (verified on `main`)

- `store.spawn(n, **data) -> np.ndarray` (indices). `store._components["Transform"]` is `(max_entities, 10)`: pos `0:3`, rot `3:7`, scale `7:10`.
- Globals uniform: WGSL `struct Globals` at `resources.py` (the large one, ~line 128), CPU packer `globals_data = np.zeros(416)` at `renderer.py:1351`, buffer `size=416` at `renderer.py:342`. Trailing `_pad_spot0 @408`, `_pad_spot1 @412`.
- `StandardMaterial._compile(textured=False, vertex_colors=False)`; variants are string-derived from `_STANDARDMATERIAL_SHADER`, so a fog block added to the base `fs_main` tail is inherited by all variants.
- `engine._geometry_registry.get_gpu_buffers(geo_id)` → `{vertex_buffer, stride, vertex_count, has_colors, ...}`; `engine._device.queue.write_buffer(buf, 0, bytes)`.
- Mesh component object exposes `_geometry_id` after `get_data`.

---

### Task 1: `Field.shift(offset)`

**Files:** `src/manifoldx/modeling/fields.py`; test `tests/modeling/test_fields.py` (append).

- [ ] **Step 1: Failing test**
```python
def test_field_shift_samples_offset():
    from manifoldx.modeling.fields import Field
    f = Field(lambda p: p[:, 2])                      # z-coordinate
    pts = np.array([[0, 0, 0], [0, 0, 5]], dtype=np.float64)
    assert np.allclose(f.shift((0, 0, 10))(pts), [10, 15])   # samples at z + 10
```
- [ ] **Step 2:** `.venv/bin/python -m pytest tests/modeling/test_fields.py -k shift -q` → FAIL (no attribute `shift`).
- [ ] **Step 3:** add to `Field` (in `fields.py`):
```python
    def shift(self, offset):
        o = np.asarray(offset, dtype=np.float64).reshape(1, 3)
        return Field(lambda p: self(p + o))
```
- [ ] **Step 4:** rerun → PASS.
- [ ] **Step 5:** `git commit -m "feat(modeling): Field.shift (sample at translated coords)"`

---

### Task 2: `@engine.background` + `Task`

**Files:** `src/manifoldx/engine.py`; test `tests/test_background.py`.

**Interfaces:** `engine.background(fn)` → callable returning `Task`. `Task.ready` (bool), `Task.result` (value; raises if not ready), `Task.wait()` (block→value), `Task.on_ready(cb)` (cb fired on main thread via `engine._drain_tasks()`).

- [ ] **Step 1: Failing test**
```python
# tests/test_background.py
import time
import manifoldx as mx


def test_background_runs_off_thread_and_delivers():
    engine = mx.Engine("bg-test")

    @engine.background
    def work(a, b):
        time.sleep(0.05)
        return a + b

    task = work(2, 3)
    assert task.wait() == 5
    assert task.ready and task.result == 5


def test_on_ready_fires_on_drain():
    engine = mx.Engine("bg-test2")

    @engine.background
    def work(x):
        return x * 10

    got = []
    work(4).on_ready(lambda r: got.append(r))
    for _ in range(200):
        engine._drain_tasks()
        if got:
            break
        time.sleep(0.005)
    assert got == [40]
```
- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_background.py -q` → FAIL (`Engine` has no `background`). (Confirm `mx.Engine("x")` constructs headlessly without a GPU device.)
- [ ] **Step 3:** in `engine.py`, add a `Task` class (module level) and engine wiring.
```python
from concurrent.futures import ThreadPoolExecutor  # top of file


class Task:
    """Handle to a value being computed on the engine's background worker."""

    def __init__(self, future, engine):
        self._future = future
        self._engine = engine
        self._cb = None

    @property
    def ready(self) -> bool:
        return self._future.done()

    @property
    def result(self):
        if not self._future.done():
            raise RuntimeError("Task result not ready; check .ready or use .wait()")
        return self._future.result()

    def wait(self):
        return self._future.result()

    def on_ready(self, cb):
        self._cb = cb
        self._engine._pending_tasks.append(self)
        return self
```
In `Engine.__init__` add:
```python
        self._executor = None
        self._pending_tasks = []
```
Add methods:
```python
    def background(self, fn):
        """Decorator: run `fn` on a background worker thread; calls return a Task."""
        def submit(*args, **kwargs):
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1)
            return Task(self._executor.submit(fn, *args, **kwargs), self)
        return submit

    def _drain_tasks(self):
        """Fire on_ready callbacks for finished tasks (main thread)."""
        if not self._pending_tasks:
            return
        still = []
        for t in self._pending_tasks:
            if t._future.done():
                if t._cb is not None:
                    t._cb(t._future.result())
            else:
                still.append(t)
        self._pending_tasks = still
```
Call `self._drain_tasks()` once per frame in `_draw_frame` (near the top, before systems run).
- [ ] **Step 4:** rerun → PASS.
- [ ] **Step 5:** `git commit -m "feat(engine): @engine.background + Task (threaded work, main-thread delivery)"`

---

### Task 3: `engine.spawn(...) -> EntityHandle` + transform proxy

**Files:** `src/manifoldx/engine.py`; test `tests/test_entity_handle.py`.

**Interfaces:** `engine.spawn(...)` returns `EntityHandle` (for the first spawned entity) or `None` if `n == 0`. `handle.transform.pos/rot/scale` read+write the entity's `Transform` row. `handle.index`, `handle._geometry_id`.

- [ ] **Step 1: Failing test**
```python
# tests/test_entity_handle.py
import numpy as np
import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material


def test_spawn_returns_handle_and_transform_writes():
    engine = mx.Engine("handle-test")
    cube = mx.geometry.cube(1, 1, 1)
    mat = mx.material.standard("#ffffff")
    h = engine.spawn(Mesh(cube), Material(mat), Transform(pos=(1, 2, 3)))
    assert h is not None and h.index >= 0
    assert np.allclose(h.transform.pos, [1, 2, 3])
    h.transform.pos = (4, 5, 6)
    assert np.allclose(engine.store._components["Transform"][h.index, 0:3], [4, 5, 6])
```
- [ ] **Step 2:** run → FAIL (spawn returns None).
- [ ] **Step 3:** add `_TransformProxy` + `EntityHandle` (module level in `engine.py`):
```python
class _TransformProxy:
    def __init__(self, store, index):
        self._store, self._i = store, index

    def _col(self, a, b):
        return self._store._components["Transform"][self._i, a:b]

    @property
    def pos(self):
        return self._col(0, 3).copy()

    @pos.setter
    def pos(self, v):
        self._store._components["Transform"][self._i, 0:3] = np.asarray(v, np.float32)

    @property
    def rot(self):
        return self._col(3, 7).copy()

    @rot.setter
    def rot(self, v):
        self._store._components["Transform"][self._i, 3:7] = np.asarray(v, np.float32)

    @property
    def scale(self):
        return self._col(7, 10).copy()

    @scale.setter
    def scale(self, v):
        self._store._components["Transform"][self._i, 7:10] = np.asarray(v, np.float32)


class EntityHandle:
    def __init__(self, engine, index, geometry_id):
        self._engine = engine
        self.index = index
        self._geometry_id = geometry_id
        self.transform = _TransformProxy(engine.store, index)
```
In `spawn`, capture the Mesh geometry id and the spawned indices, and return a handle. Modify the tail of `spawn`:
```python
        mesh_geo_id = None
        # ... inside the kwargs loop, when name == "Mesh":
        #     processed_kwargs[name] = value.get_data(n, self._geometry_registry)
        #     mesh_geo_id = getattr(value, "_geometry_id", None)
        ...
        if n > 0:
            indices = self.store.spawn(n, **processed_kwargs)
            return EntityHandle(self, int(indices[0]), mesh_geo_id)
        return None
```
(Set `mesh_geo_id` in the branch that calls `value.get_data(..., self._geometry_registry)` for `name == "Mesh"`.)
- [ ] **Step 4:** run → PASS. Also run `tests/test_engine.py tests/test_render_mvp.py -q` to confirm existing spawn callers unaffected.
- [ ] **Step 5:** `git commit -m "feat(engine): spawn returns EntityHandle with a transform proxy"`

---

### Task 4: `EntityHandle.set_geometry(mesh)` + retrofit dynamic demos

**Files:** `src/manifoldx/engine.py`; `examples/modeling_ffd.py`, `examples/modeling_ridge_flyby.py`; test `tests/test_entity_handle.py` (append).

**Interfaces:** `handle.set_geometry(mesh)` overwrites the entity's vertex buffer in place from `mesh.to_geometry()`; **requires equal vertex count** and matching attribute layout (raises `ValueError` otherwise).

- [ ] **Step 1: Failing test**
```python
# append to tests/test_entity_handle.py
import pytest
from manifoldx.modeling import Mesh as GeoMesh


def test_set_geometry_rejects_vertex_count_mismatch():
    engine = mx.Engine("handle-test2")
    geo = GeoMesh.icosphere(subdivisions=2).to_geometry()
    h = engine.spawn(Mesh(geo), Material(mx.material.standard("#fff")), Transform())
    with pytest.raises(ValueError):
        h.set_geometry(GeoMesh.icosphere(subdivisions=3))   # different vertex count
```
(No GPU device is created in this headless test; `set_geometry` must validate the vertex count **before** touching GPU buffers so the check is testable without a backend.)
- [ ] **Step 2:** run → FAIL (no `set_geometry`).
- [ ] **Step 3:** add to `EntityHandle`:
```python
    def set_geometry(self, mesh):
        geo = mesh.to_geometry()
        n_new = geo["positions"].shape[0]
        reg = self._engine._geometry_registry
        registered = reg._geometries.get(self._geometry_id, {})
        n_old = registered.get("positions", geo["positions"]).shape[0]
        if n_new != n_old:
            raise ValueError(
                f"set_geometry requires equal vertex count (have {n_old}, got {n_new}); "
                "the in-place update cannot resize the buffer."
            )
        bufs = reg.get_gpu_buffers(self._geometry_id)
        if bufs is None:
            return  # buffers not created until first render
        stride = bufs["stride"] // 4
        inter = np.zeros((n_new, stride), dtype=np.float32)
        inter[:, 0:3] = geo["positions"]
        if stride >= 6:
            inter[:, 3:6] = geo["normals"]
        if bufs.get("has_colors"):
            inter[:, 6:9] = geo["colors"]
        elif bufs.get("has_uvs"):
            inter[:, 6:8] = geo["uvs"]
        self._engine._device.queue.write_buffer(bufs["vertex_buffer"], 0, inter.tobytes())
```
(Confirm `reg._geometries[geo_id]` holds the original geometry dict — it is set in `register`; if the key/shape differs, read `bufs["vertex_count"]` for `n_old` instead.)
- [ ] **Step 4:** run → PASS. Then retrofit `examples/modeling_ffd.py` and `examples/modeling_ridge_flyby.py`: replace the `engine._geometry_registry.get_gpu_buffers(...)` / `engine._device.queue.write_buffer(...)` blocks with a captured `handle = engine.spawn(...)` and `handle.set_geometry(m)`. Re-render each briefly (`--render --duration 2`) to confirm they still animate.
- [ ] **Step 5:** `git commit -m "feat(engine): EntityHandle.set_geometry (in-place vertex update) + retrofit demos"`

---

### Task 5: `engine.enable_fog` — shader distance fog via Globals

**Files:** `src/manifoldx/resources.py` (Globals WGSL + base shader), `src/manifoldx/renderer.py` (buffer size + packer), `src/manifoldx/engine.py` (`enable_fog`); test `tests/test_fog.py`.

**Interfaces:** `engine.enable_fog(start, end, color=None)` (color defaults to `background_color`). Fog fields packed into Globals; base `StandardMaterial` shader mixes toward `fog_color` by camera distance after tonemap.

- [ ] **Step 1: Failing test**
```python
# tests/test_fog.py
import numpy as np
import manifoldx as mx
from manifoldx.resources import StandardMaterial


def test_fog_in_standard_shader():
    src = StandardMaterial._compile()
    assert "fog_enabled" in src and "fog_color" in src
    assert "distance(globals.camera_pos, in.world_pos)" in src


def test_enable_fog_sets_params():
    engine = mx.Engine("fog")
    engine.background_color = (0.7, 0.8, 0.9)
    engine.enable_fog(10.0, 50.0)
    assert engine.fog_enabled and engine.fog_start == 10.0 and engine.fog_end == 50.0
    assert np.allclose(engine.fog_color, (0.7, 0.8, 0.9))
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:**
  - In the large WGSL `struct Globals` (`resources.py`), replace the two trailing pads:
    ```
        shadow_caster:   u32,           // offset 404
        fog_start:       f32,           // offset 408
        fog_end:         f32,           // offset 412
        fog_color:       vec3<f32>,     // offset 416
        fog_enabled:     u32,           // offset 428
    };
    ```
  - In `_STANDARDMATERIAL_SHADER` `fs_main`, before `return vec4<f32>(color, 1.0);`:
    ```
        if globals.fog_enabled != 0u {
            let fd = clamp((distance(globals.camera_pos, in.world_pos) - globals.fog_start)
                           / (globals.fog_end - globals.fog_start), 0.0, 1.0);
            color = mix(color, globals.fog_color, fd);
        }
    ```
  - `renderer.py`: buffer `size=416` → `432` (line ~342); `globals_data = np.zeros(416)` → `432` (line ~1351). After the `shadow_caster` write, add (using `engine` fog attrs; default off):
    ```python
        if getattr(engine, "fog_enabled", False):
            globals_data[408:412] = np.frombuffer(np.float32(engine.fog_start).tobytes(), dtype=np.uint8)
            globals_data[412:416] = np.frombuffer(np.float32(engine.fog_end).tobytes(), dtype=np.uint8)
            globals_data[416:428] = np.frombuffer(np.asarray(engine.fog_color, np.float32).tobytes(), dtype=np.uint8)
            globals_data[428:432] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.uint8)
    ```
  - `engine.py` `__init__`: `self.fog_enabled = False; self.fog_start = 0.0; self.fog_end = 1.0; self.fog_color = (0.1, 0.1, 0.2)`. Method:
    ```python
    def enable_fog(self, start, end, color=None):
        self.fog_enabled = True
        self.fog_start = float(start)
        self.fog_end = float(end)
        self.fog_color = tuple(color) if color is not None else tuple(self.background_color)
    ```
- [ ] **Step 4:** run `tests/test_fog.py -q` → PASS. Then `make test` → all pass (Globals size change must not break existing render tests). Render `examples/pbr_demo.py --render --duration 1` as a fog-off regression smoke.
- [ ] **Step 5:** `git commit -m "feat(engine): enable_fog — shader distance fog (Globals 416->432)"`

---

### Task 6: `examples/modeling_terrain_stream.py` + CHANGELOG

**Files:** create `examples/modeling_terrain_stream.py`; modify `CHANGELOG.md`; test `tests/modeling/test_terrain_stream_demo.py`.

- [ ] **Step 1: Failing test** — the patch builder is pure/valid:
```python
# tests/modeling/test_terrain_stream_demo.py
import numpy as np
from manifoldx.modeling import Mesh, fields, Gradient


def test_patch_builder_seamless_edges():
    HEIGHT, DEPTH = 8.0, 52.0
    tmpl = Mesh.plane(width=44, depth=DEPTH, segments=40)
    terrain = fields.ridged(seed=5, freq=0.12)

    def patch(z):
        return tmpl.displace(terrain.shift((0, 0, z)), amount=HEIGHT)

    a, b = patch(0.0), patch(DEPTH)
    # a's far edge (max local z) and b's near edge (min local z) sample the same
    # world z, so their heights must match → seamless.
    za, zb = a.positions[:, 2].max(), b.positions[:, 2].min()
    edge_a = np.sort(a.positions[np.isclose(a.positions[:, 2], za)][:, 0])
    edge_b = np.sort(b.positions[np.isclose(b.positions[:, 2], zb)][:, 0])
    ha = a.positions[np.isclose(a.positions[:, 2], za)]
    hb = b.positions[np.isclose(b.positions[:, 2], zb)]
    assert np.allclose(np.sort(ha[:, 1]), np.sort(hb[:, 1]), atol=1e-4)
```
- [ ] **Step 2:** run → PASS if `shift` works (guards seam continuity).
- [ ] **Step 3:** write `examples/modeling_terrain_stream.py` — the design's target code, with real imports (`from manifoldx.resources import DirectionalLight, StandardMaterial`), `engine.enable_fog(...)`, `@engine.background patch_at`, a K-slot pool via `engine.spawn(...)` handles, `engine.camera.set_pose((0,6.5,-5),(0,2.5,22))`, and the `stream` system. Patch build:
  ```python
  def patch_at(world_z):
      return (TEMPLATE
              .displace(terrain.shift((0, 0, world_z)), amount=HEIGHT)
              .color_by(fields.coord("y").remap(0.0, HEIGHT, 0.0, 1.0), palette))
  ```
- [ ] **Step 4:** render `.venv/bin/python examples/modeling_terrain_stream.py --render --duration 6 --fps 30 --output /tmp/stream.mp4`; extract 2–3 frames and visually confirm patches stream seamlessly into the fog. `make lint`.
- [ ] **Step 5:** CHANGELOG under `### Features` (streaming demo + `Field.shift` + `@engine.background`/`Task` + `EntityHandle`/`set_geometry` + `enable_fog`). `git commit`.

---

## Self-Review

**Coverage:** shift (T1), background/Task (T2), spawn→handle+transform (T3), set_geometry + retrofit (T4), enable_fog/Globals (T5), demo (T6) — all six design components + demo. `look_from` dropped in favor of existing `set_pose`. **Placeholders:** none — code + tests concrete; two "confirm" notes (mesh_geo_id capture site, `reg._geometries` shape) flag exact insert points to verify at edit time, with fallbacks. **Type consistency:** `Task.ready/result/wait/on_ready`, `EntityHandle.transform`/`set_geometry`/`index`, Globals fields (`fog_start/end/color/enabled`) used identically in shader, packer, and `enable_fog`. Globals grows 416→432 in all three places (WGSL struct, buffer size, packer array).
