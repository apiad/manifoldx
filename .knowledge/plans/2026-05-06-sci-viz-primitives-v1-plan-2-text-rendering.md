# Sci-Viz Primitives v1 — Plan 2: Text Rendering (TextLabel + LabelMaterial)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the world-anchored text-label primitive — a `TextLabel` ECS component, a `LabelTextureAtlas` that rasterizes strings via PIL into a 2D texture array, a `LabelMaterial` that draws camera-facing billboards textured with atlas slices, and a label pass in the renderer that runs after 3D opaque draws with depth-write off and alpha-blend on. Tests validate per-string idempotency, atlas overflow behavior, GPU upload correctness, and end-to-end rendering of a known string at a known pixel.

**Architecture:** `viz/text.py` introduces `LabelTextureAtlas`, a host-side cache that keeps a fixed-size RGBA8 NumPy slice per unique string (256×64, white text on transparent background, DejaVu Sans Mono TTF bundled with the package). The atlas owns a single `wgpu.TextureViewDimension.D2_ARRAY` texture (256 slices max in v1) and a dirty flag indicating whether the host cache has new slices to upload. `TextLabel` is a 1-float ECS component holding the atlas slice index (stored as `f32`, cast to `u32` in the shader — same precedent as the rest of the viz components). `LabelMaterial` registers a per-batch uniform (`pixel_width`, `pixel_height`, `anchor_mode`, `_pad`) plus a per-instance storage buffer (`label_indices`) and bind slots for the atlas texture array + sampler. The renderer extends `RenderPipeline.run` with a third batch group keyed on `LabelMaterial`, draws after the 3D opaque pass with depth-write off and alpha-blend on, and reuses the `SPRITE_QUAD` geometry already registered in Plan 1 (one geometry, two pipelines).

**Tech Stack:** Python 3.13+, NumPy, wgpu-py (rendercanvas), Pillow ≥10.0, pytest, uv.

**Spec:** `.knowledge/analysis/2026-05-05-sci-viz-primitives-v1-design.md` (commit `6068533`), §5 (TextLabel), §6.2 (LabelMaterial), §7.1–§7.2 (label batch list + label pass), §8 (text rendering implementation).

**Out of scope for this plan (later plans):** `AxisFrame` / `ScaleBar` / `colormap_legend` and screen-anchored labels (Plan 3 — `LabelMaterial` ships with an `anchor_mode` uniform field reserved at `0.0` = world; Plan 3 wires `1.0` = screen). Functional shim API (Plan 4). Visual regression infrastructure (Plan 5).

**Plan 1 prerequisites (must already be on `main`):**
- `manifoldx.viz` subpackage with `PointCloud`, `ScalarValue`, `Radius`, `ColormapMaterial`.
- `SPRITE_QUAD` registered in `GeometryRegistry`.
- Globals uniform layout `vp + view + proj + camera_pos + pad` (208 bytes).
- `_BatchBuffers` helper in `renderer.py`.
- Pipeline cache key `(geom_id, mat_type, mat_subtype, sprite)`.

If any of these are missing, stop and finish Plan 1 first.

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `src/manifoldx/viz/text.py` | `LabelTextureAtlas` class: PIL rasterization, host-side RGBA8 cache, dirty flag, `wgpu` texture-array creation + upload. |
| `src/manifoldx/viz/assets/__init__.py` | Empty marker so the bundled font ships with the package. |
| `src/manifoldx/viz/assets/DejaVuSansMono.ttf` | ~300KB bundled monospace TTF (downloaded once during Task 1). |
| `tests/viz/test_text_atlas.py` | Atlas rasterization, idempotency, overflow error, dirty-flag semantics. |
| `tests/viz/test_label_components.py` | `TextLabel` registration + spawn + data layout. |
| `tests/viz/test_label_material.py` | `LabelMaterial` uniform packing, pipeline cache key, shader compiles via `device.create_shader_module`. |
| `tests/viz/test_label_integration.py` | End-to-end: spawn one label at the origin, render to offscreen canvas, assert center pixel alpha > 0 and matches expected RGBA. |
| `tests/viz/golden/label_helloworld.png.sha256` | SHA256 of a fixed PIL render of the literal string `"Hello, world"` at 14pt — protects against silent font/PIL changes. |

### Modified files

| Path | Change |
|---|---|
| `src/manifoldx/viz/components.py` | Add `TextLabel` component (1 float column = atlas slice index). |
| `src/manifoldx/viz/materials.py` | Add `LabelMaterial` class with WGSL shader source; ship alongside `ColormapMaterial`. |
| `src/manifoldx/viz/__init__.py` | Re-export `TextLabel`, `LabelMaterial`, `LabelTextureAtlas`. |
| `src/manifoldx/renderer.py` | Extend `_BatchBuffers` with `label_indices` (parallels `scalar_values`); add `label_batches` group in `run()`; add `_render_label_pass()`; add `_make_label_bind_group()`; extend `_get_or_create_pipeline` with a `label=True` branch (depth-write off, alpha-blend on); add `_get_or_create_atlas_texture()` for atlas-managed uploads. |
| `src/manifoldx/engine.py` | Add `_label_atlas: LabelTextureAtlas | None` field on `Engine`, lazily constructed on first `LabelMaterial` use, surfaced via `engine.get_label_atlas()`. |
| `pyproject.toml` | The `[viz]` extra already pins `pillow>=10.0` from Plan 1 — no change. Add `package-data` entry for `manifoldx.viz.assets` so the TTF ships in the wheel. |
| `CHANGELOG.md` | New entry under `## [Unreleased]` recording Plan 2's surface. |

### Untouched

`ecs.py`, `commands.py`, `components.py` (the engine-core one — viz lives in its own folder), `systems.py`, `camera.py`, `types.py`, `backends.py`, `resources.py`. The existing public API surface is preserved; Plan 2 only extends `viz/` and `renderer.py`.

---

## Sequencing notes

Tasks are ordered to respect TDD and dependency ordering:

1. **Task 1:** Bundle the DejaVu Sans Mono TTF + `viz/assets/` package layout.
2. **Tasks 2–4:** `LabelTextureAtlas` — rasterization, slot allocation / idempotency, GPU upload.
3. **Task 5:** `TextLabel` ECS component.
4. **Tasks 6–8:** `LabelMaterial` — WGSL shader source, Python class, registry integration.
5. **Tasks 9–11:** Renderer extensions — `_BatchBuffers` adds `label_indices`; pipeline-factory `label=True` branch with depth-off + alpha-blend; `label_batches` grouping in `run()`.
6. **Task 12:** `_render_label_pass` + `_make_label_bind_group`; engine wires `LabelTextureAtlas` lifecycle.
7. **Task 13:** Public re-exports updated.
8. **Tasks 14–15:** End-to-end integration test + golden PNG hash.
9. **Task 16:** CHANGELOG entry, commit, push.

Each task ends with a commit. Use `cd /home/apiad/Workspace/repos/manifoldx` for all commands; the repo has its own git history independent of the workspace.

**Testing rule (carried from Plan 1):** Tests are written inline by the orchestrator. Implementation code may be delegated, but every test in this plan is authored and run by Claude. Tests are the verification layer — delegating them is self-grading.

---

## Task 1: Bundle DejaVu Sans Mono TTF in `viz/assets/`

**Files:**
- Create: `src/manifoldx/viz/assets/__init__.py`
- Create: `src/manifoldx/viz/assets/DejaVuSansMono.ttf`
- Modify: `pyproject.toml` (add `package-data` entry)
- Modify: `MANIFEST.in` if it exists (otherwise rely on `package-data`)

- [ ] **Step 1: Create the assets package marker**

```bash
mkdir -p src/manifoldx/viz/assets
touch src/manifoldx/viz/assets/__init__.py
```

- [ ] **Step 2: Place the bundled font**

DejaVu Sans Mono is BSD-licensed and widely available. Either copy the system file (Linux: `/usr/share/fonts/TTF/DejaVuSansMono.ttf`) or fetch from the upstream archive. Verify size and SHA256 to lock the bundled artifact:

```bash
cp /usr/share/fonts/TTF/DejaVuSansMono.ttf src/manifoldx/viz/assets/DejaVuSansMono.ttf
ls -l src/manifoldx/viz/assets/DejaVuSansMono.ttf  # expect ~300KB
sha256sum src/manifoldx/viz/assets/DejaVuSansMono.ttf
```

If `/usr/share/fonts/TTF/DejaVuSansMono.ttf` is missing, install via the system package manager (`pacman -S ttf-dejavu` on Arch) or download from `https://dejavu-fonts.github.io/`.

- [ ] **Step 3: Wire `package-data` in pyproject.toml**

In `pyproject.toml`, locate the `[tool.setuptools.package-data]` (or equivalent) section. Add:

```toml
[tool.setuptools.package-data]
"manifoldx.viz.assets" = ["*.ttf"]
```

If the project uses Hatch / Flit / Poetry instead, use the equivalent (Hatch: `[tool.hatch.build.targets.wheel.force-include]`). Inspect `pyproject.toml` first to confirm the build system, then add the right key.

- [ ] **Step 4: Verify the asset loads via importlib.resources**

Run a one-liner to confirm the file is reachable through the canonical Python resource API:

```bash
uv run python -c "
from importlib.resources import files
p = files('manifoldx.viz.assets').joinpath('DejaVuSansMono.ttf')
print(p, p.is_file(), p.stat().st_size)
"
```

Expected output: a path ending in `DejaVuSansMono.ttf`, `True`, and a size around 300000.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/assets/ pyproject.toml
git commit -m "feat(viz): bundle DejaVu Sans Mono TTF for label rasterization"
```

---

## Task 2: Write `LabelTextureAtlas.rasterize_string` (host-side PIL → RGBA8 NumPy)

**Files:**
- Create: `src/manifoldx/viz/text.py`
- Create: `tests/viz/test_text_atlas.py`
- Create: `tests/viz/golden/label_helloworld.png.sha256`

- [ ] **Step 1: Write the failing test for `rasterize_string`**

`tests/viz/test_text_atlas.py`:

```python
"""Unit tests for LabelTextureAtlas (host-side rasterization + slot allocation)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from manifoldx.viz.text import LabelTextureAtlas


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_rasterize_string_shape_and_dtype():
    """A rasterized string returns a (64, 256, 4) uint8 RGBA tile."""
    tile = LabelTextureAtlas.rasterize_string("Hello", font_size=14)
    assert tile.shape == (64, 256, 4)
    assert tile.dtype == np.uint8


def test_rasterize_string_alpha_nonzero_for_glyphs():
    """Pixels covered by glyphs must have alpha > 0; transparent background must be alpha == 0."""
    tile = LabelTextureAtlas.rasterize_string("Hi", font_size=14)
    # The background (top-right corner well past the last glyph) should be
    # fully transparent.
    assert tile[0, 250, 3] == 0, "background pixel is not transparent"
    # Some pixel inside the glyph zone must be visible.
    assert (tile[:, :40, 3] > 0).any(), "no visible glyphs found in left band"


def test_rasterize_string_text_is_white():
    """Visible pixels are white (255, 255, 255, alpha)."""
    tile = LabelTextureAtlas.rasterize_string("X", font_size=14)
    visible = tile[tile[..., 3] > 0]
    # All visible RGB triplets should equal (255, 255, 255).
    assert np.all(visible[:, :3] == 255), "non-white pixels found in glyph"


def test_rasterize_string_idempotent():
    """Calling rasterize_string twice with the same inputs returns identical bytes."""
    a = LabelTextureAtlas.rasterize_string("abc 123", font_size=14)
    b = LabelTextureAtlas.rasterize_string("abc 123", font_size=14)
    assert np.array_equal(a, b)


def test_rasterize_string_golden_hello_world():
    """Hash of `Hello, world` at 14pt matches committed golden hash.

    Detects silent font / PIL regressions. If this fails after a Pillow upgrade,
    review the diff and update the golden hash if intended.
    """
    tile = LabelTextureAtlas.rasterize_string("Hello, world", font_size=14)
    digest = hashlib.sha256(tile.tobytes()).hexdigest()
    golden_path = GOLDEN_DIR / "label_helloworld.png.sha256"
    expected = golden_path.read_text().strip()
    assert digest == expected, (
        f"rasterized 'Hello, world' hash drifted: {digest} != {expected}. "
        f"If this is intentional (font / PIL upgrade), regenerate the golden."
    )
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
uv run pytest tests/viz/test_text_atlas.py -v
```

Expected: `ImportError: cannot import name 'LabelTextureAtlas' from 'manifoldx.viz.text'` (or `ModuleNotFoundError` if `text.py` doesn't exist yet).

- [ ] **Step 3: Implement `LabelTextureAtlas.rasterize_string`**

Create `src/manifoldx/viz/text.py`:

```python
"""Text rasterization for sci-viz labels.

LabelTextureAtlas owns a 2D texture array of rasterized strings. Each unique
(text, font_size) pair occupies one slice (256x64 RGBA8). The host-side cache
holds the pixel data; upload_dirty pushes new slices to the GPU.
"""

from importlib.resources import files
from typing import Dict, Tuple

import numpy as np

# Fixed slice geometry for v1 — design §8.
TILE_WIDTH = 256
TILE_HEIGHT = 64
MAX_LABELS = 256

_FONT_PATH = files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf")


class AtlasOverflowError(RuntimeError):
    """Raised when a 257th unique label is requested in v1."""


class LabelTextureAtlas:
    """Host-side cache + lazy GPU texture array for rasterized labels."""

    def __init__(self):
        self._slices: np.ndarray = np.zeros(
            (MAX_LABELS, TILE_HEIGHT, TILE_WIDTH, 4), dtype=np.uint8
        )
        self._slice_count: int = 0
        # Maps (text, font_size) -> slice index for idempotency.
        self._index: Dict[Tuple[str, int], int] = {}
        # Set of slice indices written since the last GPU upload.
        self._dirty_slices: set[int] = set()
        self._gpu_texture = None  # wgpu.GPUTexture, lazy
        self._gpu_sampler = None  # wgpu.GPUSampler, lazy

    @staticmethod
    def rasterize_string(text: str, font_size: int = 14) -> np.ndarray:
        """Rasterize `text` via PIL into an RGBA8 (TILE_HEIGHT, TILE_WIDTH, 4) tile.

        White glyphs on transparent background. Text is left-aligned with a
        small left margin and vertically centered.
        """
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(str(_FONT_PATH), size=font_size)
        # PIL drawing target — RGBA, fully transparent background.
        img = Image.new("RGBA", (TILE_WIDTH, TILE_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Vertical centering: PIL textbbox returns (left, top, right, bottom)
        # of the rendered ink box. Center vertically against the tile height.
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        x = 4  # 4px left margin
        y = (TILE_HEIGHT - text_h) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        return np.array(img, dtype=np.uint8)
```

- [ ] **Step 4: Generate and commit the golden hash for `Hello, world`**

Run a one-liner that produces the expected hash (this writes the file the test reads):

```bash
mkdir -p tests/viz/golden
uv run python -c "
import hashlib
from manifoldx.viz.text import LabelTextureAtlas
tile = LabelTextureAtlas.rasterize_string('Hello, world', font_size=14)
print(hashlib.sha256(tile.tobytes()).hexdigest())
" > tests/viz/golden/label_helloworld.png.sha256
cat tests/viz/golden/label_helloworld.png.sha256  # sanity check
```

Expected: a single 64-character hex hash on its own line.

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
uv run pytest tests/viz/test_text_atlas.py -v
```

Expected: 5/5 pass.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/viz/text.py tests/viz/test_text_atlas.py tests/viz/golden/label_helloworld.png.sha256
git commit -m "feat(viz): add LabelTextureAtlas.rasterize_string with PIL rendering"
```

---

## Task 3: `LabelTextureAtlas.get_or_create` — slot allocation + idempotency + overflow

**Files:**
- Modify: `src/manifoldx/viz/text.py`
- Modify: `tests/viz/test_text_atlas.py`

- [ ] **Step 1: Write failing tests for `get_or_create`**

Append to `tests/viz/test_text_atlas.py`:

```python
def test_get_or_create_returns_int_slice_index():
    atlas = LabelTextureAtlas()
    idx = atlas.get_or_create("alpha")
    assert isinstance(idx, int)
    assert idx == 0  # first label fills slot 0


def test_get_or_create_idempotent_per_string():
    atlas = LabelTextureAtlas()
    a = atlas.get_or_create("alpha")
    b = atlas.get_or_create("alpha")
    assert a == b
    assert atlas.slice_count == 1


def test_get_or_create_distinct_strings_get_distinct_slices():
    atlas = LabelTextureAtlas()
    a = atlas.get_or_create("alpha")
    b = atlas.get_or_create("beta")
    c = atlas.get_or_create("gamma")
    assert {a, b, c} == {0, 1, 2}
    assert atlas.slice_count == 3


def test_get_or_create_distinct_font_sizes_get_distinct_slices():
    atlas = LabelTextureAtlas()
    a = atlas.get_or_create("alpha", font_size=14)
    b = atlas.get_or_create("alpha", font_size=18)
    assert a != b
    assert atlas.slice_count == 2


def test_get_or_create_marks_slice_dirty():
    atlas = LabelTextureAtlas()
    atlas.get_or_create("alpha")
    assert 0 in atlas.dirty_slices
    # Re-requesting the same label does not re-mark it dirty if already cached.
    atlas.clear_dirty()
    atlas.get_or_create("alpha")
    assert atlas.dirty_slices == set()


def test_get_or_create_overflow_at_max_labels():
    """The 257th unique label raises AtlasOverflowError."""
    from manifoldx.viz.text import AtlasOverflowError, MAX_LABELS

    atlas = LabelTextureAtlas()
    for i in range(MAX_LABELS):
        atlas.get_or_create(f"label_{i}")
    assert atlas.slice_count == MAX_LABELS
    with pytest.raises(AtlasOverflowError):
        atlas.get_or_create("one_too_many")


def test_rasterize_string_writes_into_slice_buffer():
    """get_or_create stores the rasterized bytes in the host slice array."""
    atlas = LabelTextureAtlas()
    idx = atlas.get_or_create("X", font_size=14)
    expected = LabelTextureAtlas.rasterize_string("X", font_size=14)
    assert np.array_equal(atlas._slices[idx], expected)
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/viz/test_text_atlas.py -v
```

Expected: the new tests fail with `AttributeError: 'LabelTextureAtlas' object has no attribute 'get_or_create'` (or similar).

- [ ] **Step 3: Implement `get_or_create`, `slice_count`, `dirty_slices`, `clear_dirty`**

Append to `src/manifoldx/viz/text.py` inside `LabelTextureAtlas`:

```python
    @property
    def slice_count(self) -> int:
        return self._slice_count

    @property
    def dirty_slices(self) -> set[int]:
        return self._dirty_slices

    def clear_dirty(self) -> None:
        self._dirty_slices.clear()

    def get_or_create(self, text: str, font_size: int = 14) -> int:
        key = (text, font_size)
        if key in self._index:
            return self._index[key]
        if self._slice_count >= MAX_LABELS:
            raise AtlasOverflowError(
                f"label atlas full at {MAX_LABELS} unique labels (v1 cap). "
                f"Refused to add {key!r}."
            )
        idx = self._slice_count
        self._slices[idx] = self.rasterize_string(text, font_size=font_size)
        self._index[key] = idx
        self._dirty_slices.add(idx)
        self._slice_count += 1
        return idx
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
uv run pytest tests/viz/test_text_atlas.py -v
```

Expected: 12/12 pass (5 from Task 2 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/text.py tests/viz/test_text_atlas.py
git commit -m "feat(viz): LabelTextureAtlas.get_or_create with idempotency and overflow guard"
```

---

## Task 4: `LabelTextureAtlas.upload_dirty` — GPU texture array creation + per-slice upload

**Files:**
- Modify: `src/manifoldx/viz/text.py`
- Modify: `tests/viz/test_text_atlas.py`

- [ ] **Step 1: Write a failing test that uploads to a real wgpu device**

Append to `tests/viz/test_text_atlas.py`:

```python
def _get_offscreen_device():
    """Get a wgpu device from an offscreen canvas. Skips if unavailable."""
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except (ImportError, Exception) as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    return device


def test_upload_dirty_creates_texture_on_first_call():
    atlas = LabelTextureAtlas()
    device = _get_offscreen_device()
    atlas.get_or_create("abc")
    atlas.upload_dirty(device, device.queue)
    assert atlas.gpu_texture is not None
    assert atlas.gpu_sampler is not None
    # After upload, dirty set is empty.
    assert atlas.dirty_slices == set()


def test_upload_dirty_idempotent_when_no_new_labels():
    atlas = LabelTextureAtlas()
    device = _get_offscreen_device()
    atlas.get_or_create("abc")
    atlas.upload_dirty(device, device.queue)
    tex_first = atlas.gpu_texture
    # Second call with no new dirty slices is a no-op.
    atlas.upload_dirty(device, device.queue)
    assert atlas.gpu_texture is tex_first


def test_upload_dirty_appends_new_slices_without_recreating_texture():
    atlas = LabelTextureAtlas()
    device = _get_offscreen_device()
    atlas.get_or_create("first")
    atlas.upload_dirty(device, device.queue)
    tex_before = atlas.gpu_texture
    atlas.get_or_create("second")
    atlas.upload_dirty(device, device.queue)
    # Texture array allocates MAX_LABELS slices up front, so adding new labels
    # writes into existing slices without recreating the texture.
    assert atlas.gpu_texture is tex_before
    assert atlas.dirty_slices == set()
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/viz/test_text_atlas.py::test_upload_dirty_creates_texture_on_first_call -v
```

Expected: `AttributeError: 'LabelTextureAtlas' object has no attribute 'upload_dirty'`.

- [ ] **Step 3: Implement `upload_dirty` + GPU accessors**

Append to `LabelTextureAtlas` in `src/manifoldx/viz/text.py`:

```python
    @property
    def gpu_texture(self):
        return self._gpu_texture

    @property
    def gpu_sampler(self):
        return self._gpu_sampler

    def upload_dirty(self, device, queue) -> None:
        """Create the texture array on first call; upload each dirty slice.

        The texture is allocated once at MAX_LABELS slices so subsequent
        `get_or_create` calls just write into already-allocated slots.
        """
        import wgpu

        if not self._dirty_slices and self._gpu_texture is not None:
            return

        if self._gpu_texture is None:
            # rgba8unorm-srgb: matches the LUT convention. Glyph alpha blending
            # benefits from sRGB-correct destination math; the framebuffer is
            # already sRGB-encoded on write so the round trip preserves intent.
            self._gpu_texture = device.create_texture(
                size=(TILE_WIDTH, TILE_HEIGHT, MAX_LABELS),
                format=wgpu.TextureFormat.rgba8unorm_srgb,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                dimension=wgpu.TextureDimension.d2,
            )
            self._gpu_sampler = device.create_sampler(
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
                address_mode_v=wgpu.AddressMode.clamp_to_edge,
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
            )

        for idx in sorted(self._dirty_slices):
            slice_data = self._slices[idx]  # (TILE_HEIGHT, TILE_WIDTH, 4) uint8
            queue.write_texture(
                {
                    "texture": self._gpu_texture,
                    "origin": (0, 0, idx),
                },
                slice_data.tobytes(),
                {
                    "bytes_per_row": TILE_WIDTH * 4,
                    "rows_per_image": TILE_HEIGHT,
                },
                (TILE_WIDTH, TILE_HEIGHT, 1),
            )
        self._dirty_slices.clear()
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
uv run pytest tests/viz/test_text_atlas.py -v
```

Expected: 15/15 pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/text.py tests/viz/test_text_atlas.py
git commit -m "feat(viz): LabelTextureAtlas.upload_dirty with lazy GPU texture array"
```

---

## Task 5: `TextLabel` ECS component

**Files:**
- Modify: `src/manifoldx/viz/components.py`
- Create: `tests/viz/test_label_components.py`
- Modify: `src/manifoldx/viz/__init__.py`

- [ ] **Step 1: Write the failing test**

`tests/viz/test_label_components.py`:

```python
"""Unit tests for TextLabel component."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.viz import TextLabel


def _make_engine():
    engine = mx.Engine("test")
    engine.store.register_component("TextLabel", np.dtype("f4"), (1,))
    return engine


def test_text_label_default_zero():
    """No `index` argument → all entities get slice 0."""
    data = TextLabel().get_data(n=10)
    assert data.shape == (10, 1)
    assert data.dtype == np.float32
    assert np.all(data == 0.0)


def test_text_label_scalar_broadcast():
    data = TextLabel(index=42).get_data(n=5)
    assert data.shape == (5, 1)
    assert np.all(data == 42.0)


def test_text_label_array_per_entity():
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    data = TextLabel(index=indices).get_data(n=4)
    assert data.shape == (4, 1)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data[:, 0], indices)


def test_text_label_shape_mismatch_raises():
    with pytest.raises(ValueError):
        TextLabel(index=np.array([1, 2, 3])).get_data(n=4)


def test_text_label_spawn_into_engine():
    engine = _make_engine()
    from manifoldx.components import Transform

    engine.spawn(
        Transform(pos=np.zeros((3, 3), dtype=np.float32)),
        TextLabel(index=np.array([0, 1, 2], dtype=np.int64)),
        n=3,
    )
    assert "TextLabel" in engine.store._components
    assert engine.store._components["TextLabel"].shape[1] == 1
    assert int(np.sum(engine.store._alive)) == 3
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/viz/test_label_components.py -v
```

Expected: `ImportError: cannot import name 'TextLabel'`.

- [ ] **Step 3: Implement `TextLabel`**

Append to `src/manifoldx/viz/components.py`:

```python
class TextLabel:
    """Per-entity atlas slice index for a rasterized label.

    Storage layout: 1 float per entity (column 0). The float holds an integer
    slice index in [0, MAX_LABELS); the shader casts it to u32. Float storage
    keeps `TextLabel` symmetric with `ScalarValue` and `Radius`, all of which
    flow through the same _FieldView path.

    Usage:
        TextLabel()                              # all 0
        TextLabel(index=7)                       # broadcast scalar
        TextLabel(index=array_shape_N_int)       # explicit per-entity
    """

    def __init__(self, index=None):
        self._index = index

    def get_data(self, n: int, registry=None) -> np.ndarray:
        data = np.zeros((n, 1), dtype=np.float32)
        if self._index is None:
            return data
        v = np.asarray(self._index, dtype=np.float32)
        if v.ndim == 0:
            data[:, 0] = float(v)
        elif v.ndim == 1 and v.shape[0] == n:
            data[:, 0] = v
        else:
            raise ValueError(f"TextLabel: index shape {v.shape} incompatible with n={n}")
        return data
```

Update `src/manifoldx/viz/__init__.py` to re-export it:

```python
from manifoldx.viz.components import PointCloud, Radius, ScalarValue, TextLabel
from manifoldx.viz.materials import ColormapMaterial
from manifoldx.viz.text import LabelTextureAtlas

__all__ = [
    "PointCloud",
    "ScalarValue",
    "Radius",
    "TextLabel",
    "ColormapMaterial",
    "LabelTextureAtlas",
]
```

(Note: `LabelMaterial` is added to `__all__` in Task 7; this step does not add it yet so the public surface lands incrementally with implementation.)

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
uv run pytest tests/viz/test_label_components.py -v
```

Expected: 5/5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/components.py src/manifoldx/viz/__init__.py tests/viz/test_label_components.py
git commit -m "feat(viz): add TextLabel component (atlas slice index per entity)"
```

---

## Task 6: `LabelMaterial` WGSL shader source

**Files:**
- Modify: `src/manifoldx/viz/materials.py`

- [ ] **Step 1: Author the shader source as a Python string constant**

Append to `src/manifoldx/viz/materials.py` (above `ColormapMaterial`'s class definition, alongside `_COLORMAP_SHADER`):

```python
# WGSL shader source for LabelMaterial.
#
# Bindings (group 0):
#   0: Globals uniform   { vp: mat4x4, view: mat4x4, proj: mat4x4, camera_pos: vec3, _pad: f32 }
#   1: transforms        storage<read> array<mat4x4<f32>>
#   2: material uniform  { pixel_width: f32, pixel_height: f32, anchor_mode: f32, _pad: f32 }
#   3: label_indices     storage<read> array<f32>     # cast to u32 in shader
#   4: atlas_texture     texture_2d_array<f32>
#   5: atlas_sampler     sampler
#
# Vertex inputs:
#   @location(0) position: vec3<f32>   — quad-local in [-1, 1]^2 (z = 0)
#
# Vertex outputs:
#   @location(0) uv:    vec2<f32>      — texture UV in [0, 1]^2
#   @location(1) slice: f32            — label slice index (f32-encoded u32)

_LABEL_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct MaterialUniform {
    pixel_width: f32,
    pixel_height: f32,
    anchor_mode: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> label_indices: array<f32>;
@group(0) @binding(4) var atlas_texture: texture_2d_array<f32>;
@group(0) @binding(5) var atlas_sampler: sampler;

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) slice: f32,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;

    // Convert quad-local position to a screen-pixel offset, then back into
    // view space at the anchor's depth so the label keeps a fixed pixel size
    // regardless of camera distance. The proj matrix's [0][0] and [1][1]
    // entries encode the focal lengths used by the perspective projection.
    let half_w_pixels = material.pixel_width * 0.5;
    let half_h_pixels = material.pixel_height * 0.5;
    let view_z = max(-view_center.z, 1e-3);
    let view_dx = in.position.x * half_w_pixels * 2.0 * view_z / globals.proj[0][0] / 1024.0;
    let view_dy = in.position.y * half_h_pixels * 2.0 * view_z / globals.proj[1][1] / 1024.0;

    let view_pos = vec4<f32>(
        view_center.x + view_dx,
        view_center.y + view_dy,
        view_center.z,
        1.0,
    );
    let clip = globals.proj * view_pos;

    var out: VSOut;
    out.clip_position = clip;
    // Map quad position [-1, 1]^2 to UV [0, 1]^2 with V flipped so PIL's
    // top-left origin lands at the top of the rendered quad.
    out.uv = vec2<f32>(in.position.x * 0.5 + 0.5, 0.5 - in.position.y * 0.5);
    out.slice = label_indices[iidx];
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let layer = i32(in.slice + 0.5);
    let texel = textureSample(atlas_texture, atlas_sampler, in.uv, layer);
    return texel;
}
""".strip()
```

The 1024.0 divisor in the vertex shader is a v1 calibration constant: it converts the pixel-space half-width into normalized device coordinates assuming a reference viewport of 1024 NDC units, which lands `pixel_width=256` at roughly 25% screen width when the anchor is one unit from the camera. This is intentionally simple — Plan 3 will tighten it once the screen-anchored path is live and we have a viewport-resolution uniform to divide by.

(No new test in this task; Task 8's test covers shader compilation. The standalone string-presence check below is enough as a smoke during this slice.)

- [ ] **Step 2: Quick sanity check that the constant exists and is non-empty**

```bash
uv run python -c "from manifoldx.viz.materials import _LABEL_SHADER; assert len(_LABEL_SHADER) > 200; print('ok', len(_LABEL_SHADER))"
```

Expected: prints `ok` and a length around 1500.

- [ ] **Step 3: Commit**

```bash
git add src/manifoldx/viz/materials.py
git commit -m "feat(viz): add LabelMaterial WGSL shader source"
```

---

## Task 7: `LabelMaterial` Python class + registry integration

**Files:**
- Modify: `src/manifoldx/viz/materials.py`
- Modify: `src/manifoldx/viz/__init__.py`
- Create: `tests/viz/test_label_material.py`

- [ ] **Step 1: Write the failing tests**

`tests/viz/test_label_material.py`:

```python
"""Unit tests for LabelMaterial."""
import numpy as np

from manifoldx.viz import LabelMaterial


def test_label_material_defaults():
    mat = LabelMaterial()
    assert mat.pixel_width == 256.0
    assert mat.pixel_height == 64.0
    assert mat.anchor_mode == "world"


def test_label_material_uniform_data_shape():
    mat = LabelMaterial(pixel_width=128, pixel_height=32)
    data = mat.get_data(n=5)
    assert data.shape == (5, 4)
    assert data.dtype == np.float32
    # Row 0 is the canonical row used by the renderer.
    np.testing.assert_array_equal(data[0], [128.0, 32.0, 0.0, 0.0])


def test_label_material_pipeline_subtype_is_world_or_screen():
    """The cache key includes anchor_mode so world and screen pipelines diverge."""
    a = LabelMaterial(anchor_mode="world").pipeline_subtype
    # `screen` is reserved for Plan 3; LabelMaterial accepts the literal but
    # keeps the renderer-side fallback to world for v2 (Plan 2).
    assert a == "world"


def test_label_material_compile_returns_wgsl_source():
    src = LabelMaterial._compile()
    assert "@vertex" in src
    assert "@fragment" in src
    assert "atlas_texture" in src


def test_label_material_uniform_type_layout():
    fields = LabelMaterial.uniform_type()
    assert list(fields.keys()) == ["pixel_width", "pixel_height", "anchor_mode", "_pad"]
    assert all(t == "f32" for t in fields.values())


def test_label_material_invalid_anchor_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        LabelMaterial(anchor_mode="bogus")
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/viz/test_label_material.py -v
```

Expected: `ImportError: cannot import name 'LabelMaterial'`.

- [ ] **Step 3: Implement `LabelMaterial`**

Append to `src/manifoldx/viz/materials.py`:

```python
_VALID_ANCHOR_MODES = ("world", "screen")


class LabelMaterial(Material):
    """Camera-facing billboard material textured with a label-atlas slice.

    Per-batch uniform (4 floats):
        pixel_width, pixel_height, anchor_mode, _pad
        (anchor_mode: 0.0 = world, 1.0 = screen — Plan 2 ships world only)

    Per-instance storage buffer:
        transforms (mat4x4)  — existing
        label_indices (f32)  — atlas slice index, cast to u32 in shader

    Texture binding: 2D texture array (TILE_WIDTH x TILE_HEIGHT x MAX_LABELS).

    Pipeline cache key: includes pipeline_subtype = anchor_mode so the
    world-anchored and screen-anchored pipelines stay separate even though
    they share a shader. (Screen-anchored is reserved for Plan 3.)
    """

    binding_slot = 3

    def __init__(
        self,
        *,
        pixel_width: float = 256.0,
        pixel_height: float = 64.0,
        anchor_mode: str = "world",
    ):
        if anchor_mode not in _VALID_ANCHOR_MODES:
            raise ValueError(
                f"LabelMaterial.anchor_mode must be one of {_VALID_ANCHOR_MODES}, got {anchor_mode!r}"
            )
        self.pixel_width = float(pixel_width)
        self.pixel_height = float(pixel_height)
        self.anchor_mode = anchor_mode

    @classmethod
    def _compile(cls) -> str:
        return _LABEL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "pixel_width": "f32",
            "pixel_height": "f32",
            "anchor_mode": "f32",
            "_pad": "f32",
        }

    @property
    def pipeline_subtype(self) -> str:
        return self.anchor_mode

    def get_data(self, n: int, registry=None) -> np.ndarray:
        anchor_f = 0.0 if self.anchor_mode == "world" else 1.0
        row = np.array(
            [self.pixel_width, self.pixel_height, anchor_f, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 4)).copy()
```

Update `src/manifoldx/viz/__init__.py` to re-export `LabelMaterial`:

```python
from manifoldx.viz.components import PointCloud, Radius, ScalarValue, TextLabel
from manifoldx.viz.materials import ColormapMaterial, LabelMaterial
from manifoldx.viz.text import LabelTextureAtlas

__all__ = [
    "PointCloud",
    "ScalarValue",
    "Radius",
    "TextLabel",
    "ColormapMaterial",
    "LabelMaterial",
    "LabelTextureAtlas",
]
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
uv run pytest tests/viz/test_label_material.py -v
```

Expected: 6/6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/viz/materials.py src/manifoldx/viz/__init__.py tests/viz/test_label_material.py
git commit -m "feat(viz): add LabelMaterial class with anchor_mode pipeline subtype"
```

---

## Task 8: `LabelMaterial` shader compiles in a real wgpu shader module

**Files:**
- Modify: `tests/viz/test_label_material.py`

This guards against shader-source regressions that the static string assertions can't catch.

- [ ] **Step 1: Add the failing test**

Append to `tests/viz/test_label_material.py`:

```python
def test_label_material_shader_module_creates_without_error():
    """The shader source must compile in a real wgpu shader module."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    src = LabelMaterial._compile()
    module = device.create_shader_module(code=src)
    assert module is not None
```

- [ ] **Step 2: Run to confirm it passes** (the shader source from Task 6 should already compile)

```bash
uv run pytest tests/viz/test_label_material.py -v
```

Expected: 7/7 pass. If the shader fails to compile, fix the WGSL in `_LABEL_SHADER` until it compiles, then run this task's test until green.

- [ ] **Step 3: Commit**

```bash
git add tests/viz/test_label_material.py
git commit -m "test(viz): assert LabelMaterial WGSL compiles in wgpu"
```

---

## Task 9: Extend `_BatchBuffers` with `label_indices`

**Files:**
- Modify: `src/manifoldx/renderer.py`

- [ ] **Step 1: Write a failing unit test**

Append to a new file `tests/viz/test_batch_buffers.py`:

```python
"""Unit test for _BatchBuffers extension to label_indices."""
import numpy as np
import pytest


def _get_offscreen_device():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter.request_device_sync()


def test_batch_buffers_upload_label_indices():
    from manifoldx.renderer import _BatchBuffers

    device = _get_offscreen_device()
    bufs = _BatchBuffers(device)
    data = np.array([0, 1, 2, 3], dtype=np.float32)
    bufs.upload_label_indices(data)
    assert bufs.label_indices_buf is not None
    assert bufs.label_indices_capacity >= data.nbytes


def test_batch_buffers_label_indices_grows_capacity():
    from manifoldx.renderer import _BatchBuffers

    device = _get_offscreen_device()
    bufs = _BatchBuffers(device)
    bufs.upload_label_indices(np.zeros(4, dtype=np.float32))
    cap_small = bufs.label_indices_capacity
    bufs.upload_label_indices(np.zeros(64, dtype=np.float32))
    assert bufs.label_indices_capacity > cap_small
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/viz/test_batch_buffers.py -v
```

Expected: `AttributeError: '_BatchBuffers' object has no attribute 'upload_label_indices'`.

- [ ] **Step 3: Add `upload_label_indices` to `_BatchBuffers`**

In `src/manifoldx/renderer.py`, locate the `_BatchBuffers` class (~line 14). Add fields in `__init__`:

```python
        self.label_indices_buf = None
        self.label_indices_capacity = 0
```

Add the upload method below `upload_radii`:

```python
    def upload_label_indices(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.label_indices_buf is None or self.label_indices_capacity < n_bytes:
            self.label_indices_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.label_indices_capacity = n_bytes
        self._device.queue.write_buffer(self.label_indices_buf, 0, data.tobytes())
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
uv run pytest tests/viz/test_batch_buffers.py -v
```

Expected: 2/2 pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/renderer.py tests/viz/test_batch_buffers.py
git commit -m "refactor(renderer): _BatchBuffers gains label_indices upload path"
```

---

## Task 10: Pipeline factory adds `label=True` branch (depth-write off, alpha-blend on)

**Files:**
- Modify: `src/manifoldx/renderer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/viz/test_label_material.py`:

```python
def test_renderer_creates_label_pipeline_with_alpha_blend():
    """The pipeline factory must accept label=True and configure alpha blending."""
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    from manifoldx.renderer import RenderPipeline
    from manifoldx.viz import LabelMaterial

    rp = RenderPipeline.__new__(RenderPipeline)
    rp._pipelines = {}
    rp._bind_group_layouts = {}
    rp._pipeline_layouts = {}
    rp._material_buffers = {}

    mat = LabelMaterial()
    pipeline, layout = rp._get_or_create_pipeline(
        device,
        wgpu.TextureFormat.rgba8unorm_srgb,
        geometry_id=1,  # any non-zero
        material=mat,
        registry=None,
        label=True,
    )
    assert pipeline is not None
    assert layout is not None
    # Same call returns cached pipeline.
    pipeline_again, _ = rp._get_or_create_pipeline(
        device,
        wgpu.TextureFormat.rgba8unorm_srgb,
        geometry_id=1,
        material=mat,
        registry=None,
        label=True,
    )
    assert pipeline_again is pipeline
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/viz/test_label_material.py::test_renderer_creates_label_pipeline_with_alpha_blend -v
```

Expected: `TypeError: _get_or_create_pipeline() got an unexpected keyword argument 'label'` or similar.

- [ ] **Step 3: Extend the pipeline factory signature**

In `src/manifoldx/renderer.py`, change the signature of `_get_or_create_pipeline`:

```python
    def _get_or_create_pipeline(
        self, device, texture_format, geometry_id, material, registry,
        sprite=False, label=False,
    ):
```

Update the cache-key block at the top to include `label`:

```python
        material_type = type(material).__name__
        material_subtype = getattr(material, "pipeline_subtype", None)

        if label:
            key = (geometry_id, material_type, material_subtype, "label")
        elif sprite:
            key = (geometry_id, material_type, material_subtype, True)
        else:
            key = (geometry_id, material_type)
```

Add a new branch *before* the existing `if sprite:` block (so the existing flow is unchanged when `label=False`):

```python
        if label:
            # 6 bindings: globals, transforms, material uniform, label_indices,
            # atlas texture array, atlas sampler.
            bind_group_entries = [
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2_array,
                    },
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

            shader_module = device.create_shader_module(code=material._compile())

            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 3 * 4,  # SPRITE_QUAD: position only
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x3,
                                    "offset": 0,
                                    "shader_location": 0,
                                },
                            ],
                        }
                    ],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": wgpu.FrontFace.ccw,
                    # Labels are double-sided: the camera-facing billboard
                    # always points toward the camera by construction, so
                    # back-face culling would just be a hidden footgun.
                    "cull_mode": wgpu.CullMode.none,
                },
                # Depth-test on (so labels behind opaque geometry are occluded)
                # but depth-write off (so overlapping labels alpha-blend cleanly).
                depth_stencil={
                    "format": wgpu.TextureFormat.depth24plus,
                    "depth_write_enabled": False,
                    "depth_compare": wgpu.CompareFunction.less_equal,
                },
                fragment={
                    "module": shader_module,
                    "entry_point": "fs_main",
                    "targets": [
                        {
                            "format": texture_format,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.src_alpha,
                                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                    "operation": wgpu.BlendOperation.add,
                                },
                            },
                        }
                    ],
                },
            )

            self._pipelines[key] = pipeline

            # Material uniform buffer for the label key (4 floats = 16 bytes).
            material_buffer = device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

            return pipeline, bind_group_layout
```

(Place this branch above `if sprite:` so the dispatch order is `label > sprite > mesh`.)

- [ ] **Step 4: Run the test to confirm it passes**

```bash
uv run pytest tests/viz/test_label_material.py::test_renderer_creates_label_pipeline_with_alpha_blend -v
```

Expected: 1/1 pass.

- [ ] **Step 5: Run the full viz suite to confirm no regressions**

```bash
uv run pytest tests/viz/ -v
```

Expected: every Plan 1 + Plan 2 test still passes; nothing red.

- [ ] **Step 6: Commit**

```bash
git add src/manifoldx/renderer.py tests/viz/test_label_material.py
git commit -m "feat(renderer): pipeline factory adds label=True branch (alpha-blend, depth-write off)"
```

---

## Task 11: Group label entities in `RenderPipeline.run`

**Files:**
- Modify: `src/manifoldx/renderer.py`

- [ ] **Step 1: Find the existing batch-grouping loop**

Around line 540–572 in `renderer.py`, the `run` method groups entities into `mesh_batches` and `sprite_batches`. The label grouping mirrors the sprite path: any alive entity with a `TextLabel` component routes into a `label_batches` dict keyed by `mat_id`.

- [ ] **Step 2: Update the run loop**

Add a new flag near the existing `has_point_cloud`:

```python
        has_text_label = "TextLabel" in self._store._components
```

Update the early-return guard so the renderer still runs when only labels are present:

```python
        if not has_mesh and not has_point_cloud and not has_text_label:
            return
```

Add a new dict initializer alongside `mesh_batches` and `sprite_batches`:

```python
        label_batches = {}  # mat_id -> list of local indices
```

Inside the per-entity loop, route by component presence — TextLabel takes priority over PointCloud (an entity rarely has both, but the routing must be deterministic):

```python
        for i, entity_idx in enumerate(alive_indices):
            mat_id = int(material_data[i, 0]) if material_data is not None else 0
            geom_id = int(mesh_data[i, 0]) if mesh_data is not None else 0

            is_label = has_text_label and self._is_label_entity(entity_idx, mat_id, engine)
            is_sprite = (not is_label) and has_point_cloud and geom_id == 0

            if is_label:
                if mat_id not in label_batches:
                    label_batches[mat_id] = []
                label_batches[mat_id].append(i)
            elif is_sprite:
                if mat_id not in sprite_batches:
                    sprite_batches[mat_id] = []
                sprite_batches[mat_id].append(i)
            else:
                if not has_mesh or geom_id == 0:
                    continue
                mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
                mat_type = type(mat_obj).__name__ if mat_obj else "BasicMaterial"
                key = (geom_id, mat_type)
                if key not in mesh_batches:
                    mesh_batches[key] = []
                mesh_batches[key].append(i)
```

The `_is_label_entity` helper checks the material type against `LabelMaterial` (a TextLabel component on its own does not imply label rendering — the entity must also carry a `LabelMaterial`):

```python
    def _is_label_entity(self, entity_idx, mat_id, engine):
        if mat_id <= 0:
            return False
        mat_obj = engine._material_registry.get(mat_id)
        if mat_obj is None:
            return False
        from manifoldx.viz import LabelMaterial
        return isinstance(mat_obj, LabelMaterial)
```

(Place `_is_label_entity` near `_render_sprite_batches`.)

- [ ] **Step 3: Add the call site for the label pass**

After the existing sprite-batch dispatch (around line 616–619), append:

```python
        # ---------------------------------------------------------------
        # Draw label batches (depth-write off, alpha-blend on)
        # ---------------------------------------------------------------
        if label_batches:
            self._render_label_pass(
                engine, render_pass, label_batches, model_matrices, material_data
            )
```

(`_render_label_pass` is defined in Task 12 — this task adds the dispatch site only. The renderer will fail at runtime if a label entity is spawned before Task 12 lands; that's fine — Task 12 is the very next task.)

- [ ] **Step 4: Smoke-run the existing suite**

```bash
uv run pytest tests/viz/ -v
```

Expected: all existing tests pass (no label entities spawned in any test yet).

- [ ] **Step 5: Commit**

```bash
git add src/manifoldx/renderer.py
git commit -m "refactor(renderer): route TextLabel + LabelMaterial entities into label_batches"
```

---

## Task 12: `_render_label_pass` + engine atlas lifecycle

**Files:**
- Modify: `src/manifoldx/renderer.py`
- Modify: `src/manifoldx/engine.py`

- [ ] **Step 1: Add `Engine.get_label_atlas()`**

In `src/manifoldx/engine.py`, locate the `Engine.__init__`. Add:

```python
        self._label_atlas = None  # type: LabelTextureAtlas | None
```

Add a method:

```python
    def get_label_atlas(self):
        """Lazily construct the label atlas on first use.

        Used by the renderer's label pass and by user code that needs to
        register strings ahead of time.
        """
        if self._label_atlas is None:
            from manifoldx.viz.text import LabelTextureAtlas
            self._label_atlas = LabelTextureAtlas()
        return self._label_atlas
```

- [ ] **Step 2: Implement `_render_label_pass`**

Append to `RenderPipeline` in `src/manifoldx/renderer.py` (next to `_render_sprite_batches`):

```python
    def _render_label_pass(
        self, engine, render_pass, label_batches, model_matrices, material_data
    ):
        """Draw all label batches with alpha-blend on, depth-write off.

        Mirrors `_render_sprite_batches` but with the label-specific bindings:
        atlas texture array + slice index per instance instead of LUT + scalar.
        """
        from manifoldx.viz.materials import LabelMaterial

        all_local_indices = []
        batch_draw_info = {}
        instance_offset = 0
        for mat_id, local_indices in label_batches.items():
            count = len(local_indices)
            batch_draw_info[mat_id] = (instance_offset, count)
            all_local_indices.extend(local_indices)
            instance_offset += count

        if not all_local_indices:
            return

        ent_arr = np.asarray(all_local_indices, dtype=np.int64)

        # Transforms for label entities (column-major upload, like sprites).
        all_matrices = model_matrices[ent_arr]
        all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
        # Reuse the sprite buffer is unsafe: the upload would clobber sprite
        # transforms within the same render pass. Use a dedicated label buffer.
        if self._label_batch_buffers is None:
            self._label_batch_buffers = _BatchBuffers(self._device)
        self._label_batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))

        # Per-instance label slice indices.
        if "TextLabel" in self._store._components:
            alive_indices = np.where(self._store._alive)[0]
            entity_indices = alive_indices[ent_arr]
            label_data = self._store.get_component_data("TextLabel", entity_indices)
            self._label_batch_buffers.upload_label_indices(
                label_data[:, 0].astype(np.float32)
            )
        else:
            self._label_batch_buffers.upload_label_indices(
                np.zeros(len(ent_arr), dtype=np.float32)
            )

        # Ensure the atlas's GPU texture is current.
        atlas = engine.get_label_atlas()
        atlas.upload_dirty(self._device, self._device.queue)
        if atlas.gpu_texture is None:
            return  # nothing to draw — no labels were registered

        # Sprite quad geometry (shared with the sprite path).
        sprite_geom_id = engine._geometry_registry.get_id("sprite_quad")
        gpu_buffers = engine._geometry_registry.get_gpu_buffers(sprite_geom_id)
        if gpu_buffers is None:
            geom_obj = engine._geometry_registry.get(sprite_geom_id)
            if geom_obj is not None:
                gpu_buffers = engine._geometry_registry.create_buffers(
                    sprite_geom_id, geom_obj, self._device.queue
                )
        if gpu_buffers is None:
            return

        for mat_id, local_indices in label_batches.items():
            mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
            if not isinstance(mat_obj, LabelMaterial):
                continue
            first_instance, instance_count = batch_draw_info[mat_id]

            pipeline, bind_group_layout = self._get_or_create_pipeline(
                self._device,
                engine._texture_format,
                sprite_geom_id,
                mat_obj,
                engine._material_registry,
                label=True,
            )

            mat_data = mat_obj.get_data(instance_count, engine._material_registry)
            material_type = type(mat_obj).__name__
            material_subtype = getattr(mat_obj, "pipeline_subtype", None)
            bkey = (sprite_geom_id, material_type, material_subtype, "label")
            mat_buffer = self._material_buffers.get(bkey)
            if mat_buffer is not None:
                first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
                self._device.queue.write_buffer(
                    mat_buffer, 0, first_row.astype(np.float32).tobytes()
                )

            bind_group = self._make_label_bind_group(
                bind_group_layout, atlas, mat_buffer
            )

            render_pass.set_pipeline(pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 0)
            render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
            render_pass.set_index_buffer(gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32)
            render_pass.draw_indexed(
                gpu_buffers["index_count"],
                instance_count,
                first_index=0,
                base_vertex=0,
                first_instance=first_instance,
            )

    def _make_label_bind_group(self, bind_group_layout, atlas, mat_buffer):
        """Bindings 0-5 for label rendering.

        0: globals uniform (208 bytes)
        1: transforms storage
        2: material uniform (16 bytes)
        3: label_indices storage
        4: atlas_texture (texture_2d_array)
        5: atlas_sampler
        """
        return self._device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self._globals_buffer, "offset": 0, "size": 208},
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._label_batch_buffers.transforms_buf,
                        "offset": 0,
                        "size": self._label_batch_buffers.transforms_capacity,
                    },
                },
                {
                    "binding": 2,
                    "resource": {"buffer": mat_buffer, "offset": 0, "size": 16},
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self._label_batch_buffers.label_indices_buf,
                        "offset": 0,
                        "size": self._label_batch_buffers.label_indices_capacity,
                    },
                },
                {
                    "binding": 4,
                    "resource": atlas.gpu_texture.create_view(),
                },
                {
                    "binding": 5,
                    "resource": atlas.gpu_sampler,
                },
            ],
        )
```

Add the new field in `RenderPipeline.__init__` near where `_sprite_batch_buffers` is initialized:

```python
        self._label_batch_buffers = None  # _BatchBuffers for the label pass
```

Initialize it during device setup, alongside the other batch buffers:

```python
        self._label_batch_buffers = _BatchBuffers(device)
```

- [ ] **Step 3: Smoke-run the suite**

```bash
uv run pytest tests/viz/ -v
```

Expected: existing tests still green (no label-rendering test yet).

- [ ] **Step 4: Commit**

```bash
git add src/manifoldx/renderer.py src/manifoldx/engine.py
git commit -m "feat(renderer): _render_label_pass with atlas-bound 2D texture array"
```

---

## Task 13: Public re-exports + smoke import test

**Files:**
- Modify: `src/manifoldx/viz/__init__.py` (already done in Tasks 5/7 — verify)
- Create: `tests/viz/test_public_surface.py`

- [ ] **Step 1: Write a failing test that asserts the public surface is correct**

`tests/viz/test_public_surface.py`:

```python
"""Plan 2 public surface — what users import."""
def test_plan2_public_surface_imports():
    from manifoldx.viz import (
        ColormapMaterial,
        LabelMaterial,
        LabelTextureAtlas,
        PointCloud,
        Radius,
        ScalarValue,
        TextLabel,
    )
    assert LabelMaterial is not None
    assert LabelTextureAtlas is not None
    assert TextLabel is not None


def test_plan2_public_surface_in_all():
    import manifoldx.viz as viz
    expected = {
        "PointCloud", "ScalarValue", "Radius", "TextLabel",
        "ColormapMaterial", "LabelMaterial", "LabelTextureAtlas",
    }
    assert expected.issubset(set(viz.__all__))
```

- [ ] **Step 2: Run to confirm it passes** (Tasks 5 and 7 already updated `__init__.py`)

```bash
uv run pytest tests/viz/test_public_surface.py -v
```

Expected: 2/2 pass. If fails, fix `src/manifoldx/viz/__init__.py` until green.

- [ ] **Step 3: Commit**

```bash
git add tests/viz/test_public_surface.py
git commit -m "test(viz): pin Plan 2 public surface (LabelMaterial, TextLabel, LabelTextureAtlas)"
```

---

## Task 14: End-to-end integration test — label renders to a non-transparent pixel

**Files:**
- Create: `tests/viz/test_label_integration.py`

- [ ] **Step 1: Write the failing test**

`tests/viz/test_label_integration.py`:

```python
"""End-to-end integration tests for sci-viz Plan 2 label rendering."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Transform
from manifoldx.viz import LabelMaterial, TextLabel


def _make_offscreen_engine(width=128, height=128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")

    try:
        engine = mx.Engine("test", width=width, height=height)
        engine._init_canvas(canvas)
    except Exception as e:
        pytest.skip(f"engine initialization failed: {e}")

    engine.store.register_component("TextLabel", np.dtype("f4"), (1,))
    engine._running = True
    return engine


def test_label_renders_visible_pixels_at_origin():
    """A label spawned at origin produces non-transparent pixels in the rendered frame."""
    engine = _make_offscreen_engine()
    engine.camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    atlas = engine.get_label_atlas()
    slice_idx = atlas.get_or_create("HELLO")  # blocky monospace, easy to read

    engine.spawn(
        Material(LabelMaterial()),
        Transform(pos=(0.0, 0.0, 0.0)),
        TextLabel(index=slice_idx),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    assert frame.shape == (128, 128, 4)
    assert frame.dtype == np.uint8

    # The label is white-on-transparent; on a black background the rendered
    # text region should contain bright pixels somewhere in the middle band.
    middle = frame[40:88, 16:112, :3]
    bright = (middle.max(axis=-1) > 100).any()
    assert bright, "no bright pixels found in label region — text not rendered"


def test_label_does_not_render_when_no_atlas_strings_registered():
    """If no string was ever registered, the label pass is a no-op (no crash)."""
    engine = _make_offscreen_engine()
    engine.camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    engine.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    engine.spawn(
        Material(LabelMaterial()),
        Transform(pos=(0.0, 0.0, 0.0)),
        TextLabel(index=0),
        n=1,
    )

    engine._draw_frame()
    frame = engine._render_canvas.draw()
    assert frame is not None  # no crash
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/viz/test_label_integration.py -v
```

Expected: either fails on the brightness assertion OR raises a binding/layout error. Both are valid red states; the next steps debug toward green.

- [ ] **Step 3: Iterate inline until green**

If the test fails, debug the label pipeline directly — do NOT delegate this. Common issues:

- Atlas slice 0 is empty (the test forgot to call `get_or_create`). Confirm the test uses `slice_idx = atlas.get_or_create(...)`.
- Bind group layout mismatch: count entries against the shader's `@binding(...)` declarations.
- Texture format mismatch: framebuffer is `rgba8unorm-srgb`, atlas is `rgba8unorm-srgb` — both should match. If a fork/host uses `rgba8unorm`, the GPU may reject the bind group.
- Quad sizing: the vertex shader's pixel-to-NDC math may put the label off-screen for a 128×128 canvas. Print `frame.max()` and visualize via `Pillow` if needed:

  ```python
  from PIL import Image
  Image.fromarray(frame, mode="RGBA").save("/tmp/label_debug.png")
  ```

  Open `/tmp/label_debug.png` to see what the renderer actually produced. Adjust the calibration constant in `_LABEL_SHADER` if necessary.

- [ ] **Step 4: Run the full viz suite to confirm green and no regressions**

```bash
uv run pytest tests/viz/ -v
```

Expected: every Plan 1 + Plan 2 test passes.

- [ ] **Step 5: Commit**

```bash
git add tests/viz/test_label_integration.py src/manifoldx/renderer.py src/manifoldx/viz/materials.py
git commit -m "test(viz): label end-to-end renders visible pixels via offscreen canvas"
```

(Include any source files that needed touching to make the test pass — likely none beyond what Tasks 6, 10, 12 wrote, but if a calibration constant changed in `_LABEL_SHADER`, commit it together.)

---

## Task 15: Run the full project test suite to confirm Plan 1 still works

**Files:** none modified; this is a guardrail step.

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: every test (engine core, ECS, components, materials, viz Plan 1, viz Plan 2) passes.

If any non-viz test now fails, the renderer's `run()` change in Task 11 is the most likely culprit — check that the new early-return guard `if not has_mesh and not has_point_cloud and not has_text_label: return` did not regress the no-component baseline.

- [ ] **Step 2: Run the orbital demo to confirm visual regressions are absent**

```bash
uv run python examples/point_cloud_demo.py --render --frames 30 --output /tmp/orbital_smoke.mp4
```

Expected: no exceptions; an `mp4` is produced. The visual content should match Plan 1's output (PBR star + inferno-colored disk). Plan 2 changes did not touch the sprite or mesh paths, so the frames should be byte-identical to Plan 1's output (modulo any non-deterministic float ops).

- [ ] **Step 3: Commit any cleanup**

If the smoke runs cleanly with no source changes, skip this step. If a renderer adjustment was needed, commit it:

```bash
git add src/manifoldx/renderer.py
git commit -m "fix(renderer): preserve Plan 1 visual output across label-pass refactor"
```

---

## Task 16: CHANGELOG entry + push

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append the Plan 2 entry under `## [Unreleased]`**

In `CHANGELOG.md`, locate `## [Unreleased]`. Add the Features / Refactors blocks below the existing Plan 1 ones (do NOT promote `[Unreleased]` to a versioned heading — that happens at release time):

```markdown
### Features
- **Sci-viz Plan 2 (text rendering)** — `manifoldx.viz` adds the `TextLabel` ECS component, the `LabelMaterial` camera-facing-billboard material with depth-test on / depth-write off / alpha-blend on, and the `LabelTextureAtlas` host-side cache that rasterizes strings via PIL (DejaVu Sans Mono bundled, 256×64 RGBA8 tiles, sRGB-correct) and uploads them lazily to a `texture_2d_array` (256-slice cap in v1).
- **Label render pass** — `RenderPipeline.run` now batches `TextLabel + LabelMaterial` entities into a third draw group dispatched after the 3D opaque pass. Pipeline cache key extended with a `"label"` fourth element so the world-anchored label pipeline never collides with sprite or mesh pipelines.
- **`engine.get_label_atlas()`** — lazy accessor for the per-engine atlas, used by the renderer and by user code that wants to register strings up front.

### Refactors
- **`_BatchBuffers` extended** — additional lazily-allocated `label_indices` buffer parallels the existing `transforms` / `scalar_values` / `radii` storage paths.
- **Pipeline factory `_get_or_create_pipeline`** — now accepts `label=True` for an alpha-blended, depth-write-off path with bind slots for a `texture_2d_array` + sampler.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record Plan 2 sci-viz text rendering"
```

- [ ] **Step 3: Push**

```bash
git push origin main
```

Expected: push succeeds. Plan 2 is now landed on `main`.

---

## Self-review

**Spec coverage:**

- §5 `TextLabel` component → Task 5.
- §6.2 `LabelMaterial` → Tasks 6 (shader), 7 (Python class + cache key), 8 (compile sanity).
- §7.1 batch construction (label batch list) → Task 11.
- §7.2 render order — label pass after 3D opaque → Task 11 (dispatch site) + Task 12 (implementation). Depth-write off + alpha-blend on encoded in Task 10's pipeline branch.
- §8 text rendering implementation — `LabelTextureAtlas` + PIL + bundled DejaVu Sans Mono + 256×64 fixed slices + 256 cap → Tasks 1 (asset), 2 (rasterize), 3 (slot allocation + overflow), 4 (GPU upload).
- Functional shim `axes(...)` ticks via labels → **deferred to Plan 3** (axes are Plan 3's primary deliverable).
- Visual regression infrastructure → **deferred to Plan 5**.

**Placeholder scan:** searched for "TBD", "TODO", "fill in details", "similar to Task N", and the writing-plans skill's other red-flag phrases. None present in the implementation steps. The one judgment call deferred to runtime — the 1024.0 calibration constant in the vertex shader — is documented in Task 6 with rationale and a Plan 3 follow-up note.

**Type consistency:**

- `TextLabel` uses `np.dtype("f4")` shape `(1,)` — matches `ScalarValue` and `Radius` (Plan 1 precedent for `_FieldView`).
- `LabelMaterial.uniform_type()` returns 4 `f32` fields totaling 16 bytes; matches the `_LABEL_SHADER` `MaterialUniform` struct.
- Pipeline cache key `(geom_id, mat_type, mat_subtype, "label")` is distinct from the sprite key `(geom_id, mat_type, mat_subtype, True)` — string vs bool ensures no collision even when subtypes match.
- `_BatchBuffers.label_indices_buf` parallels `scalar_values_buf` and `radii_buf` exactly.
- `_render_label_pass` uses the same flatten-then-dispatch shape as `_render_sprite_batches`. The `_label_batch_buffers` instance is distinct from `_sprite_batch_buffers` so cross-pass uploads in the same frame don't clobber.

**Architecture-level dependencies on Plan 1 (re-check before starting):**

- `Engine._init_canvas`, `Engine._draw_frame`, `Engine.spawn(Material(...), Transform(...), n=N)` API.
- `Engine.store.register_component(name, dtype, shape)`.
- `Engine._geometry_registry.get_id("sprite_quad")` returns a non-zero id.
- `RenderPipeline._globals_buffer` size 208 bytes.
- `_BatchBuffers` exists with `upload_transforms`, `upload_scalar_values`, `upload_radii`.
- `_get_or_create_pipeline(..., sprite=False, label=False)` extension point.

If any of these have changed since Plan 1's commit `e006ed3`, update the affected tasks before executing.

---

## Execution Handoff

Plan complete and saved to `.knowledge/plans/2026-05-06-sci-viz-primitives-v1-plan-2-text-rendering.md`. Two execution options:

**1. Subagent-Driven** — fresh subagent per task, review between tasks, fast iteration via the `delegate` skill (route code/refactor work through `opencode-go/deepseek-v4-pro`; keep all test authoring inline per the never-delegate-testing rule).

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?
