# GUI v1 — Plan 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the rendering pipe and non-interactive widget surface of the GUI layer (`Panel`, `Text`, `ValueDisplay`) so that researchers can dock a static control panel with text labels and live value readouts; no clicks, no drags — that's Plan 2.

**Architecture:** A new `gui` render pass at the end of the render order (`mesh → sprite → volume → label → gui`) consumes a tree of `Widget` objects living in `engine.gui`. Layout (stack + flex) and style (theme + named classes + per-widget overrides) are pure-Python; painting accumulates rect + glyph ops into a `Painter`; the pass issues exactly two instanced draws per frame — one through a new `RectMaterial` (WGSL SDF rounded rect) and one through the existing screen-anchored `LabelMaterial` pipeline (with a `"gui"` 5th pipeline-cache element so it doesn't collide with `label`).

**Tech Stack:** Python 3.13+, wgpu, numpy, WGSL, pytest. Reuses `LabelTextureAtlas` from sci-viz Plan 2 and the `Globals` uniform's `viewport_size` field from sci-viz Plan 3.

**Scope for this plan:** Static + dynamic-text rendering only. No `_GuiBridge`, no hit-testing, no `Button`/`Slider`/`Toggle` — those land in Plan 2.

**Sub-project context:** This is the first of two plans for the GUI v1 sub-project. Plan 2 (`2026-05-??-gui-v1-plan-2-interaction.md`) will add the bridge, three interactive widgets, an example, and the CHANGELOG entry for the whole sub-project. This plan adds a placeholder `[Unreleased]` line only.

**Design reference:** `.knowledge/analysis/2026-05-10-gui-v1-design.md`.

---

## File Structure

**New files (under `src/manifoldx/gui/`):**

- `__init__.py` — public API surface: `Panel`, `Text`, `ValueDisplay`, `style`. (`Button`/`Slider`/`Toggle` added in Plan 2.)
- `style.py` — `set_theme`, `define`, `resolve`, plus the padding-string parser and the color parser.
- `layout.py` — `LayoutBox` dataclass and `compute_layout(panel, viewport)` returning a `{Widget: LayoutBox}` map.
- `widgets.py` — abstract `Widget`, `Panel`, `Text`.
- `painter.py` — `Painter` accumulator (`draw_rect`, `draw_text` ops) plus `paint(widget, painter, layout)` tree walker.
- `material.py` — `RectMaterial` (Material subclass) — WGSL signed-distance rounded rect with optional 1 px border.
- `value_display.py` — `ValueDisplay` widget (kept separate from `widgets.py` because it owns getter-call + atlas-cache logic).

**New files (under `src/manifoldx/render/passes/`):**

- `gui.py` — `render_gui_pass(rp, engine, render_pass)` — paints all root panels, uploads rect-instance + glyph-instance buffers, issues two instanced draws.

**Modified files:**

- `src/manifoldx/engine.py` — add `self.gui = _GuiRoot()` in `__init__` after the input-bridge construction (line ~67); `_GuiRoot` is defined in `gui/widgets.py` and imported.
- `src/manifoldx/renderer.py` — register the gui pass in `RenderPipeline.render` after the label pass.
- `CHANGELOG.md` — placeholder bullet under `[Unreleased]`; the full entry lands in Plan 2.

**New test files (under `tests/gui/`, mirroring `tests/viz/`):**

- `tests/gui/__init__.py` — empty.
- `tests/gui/test_style.py` — pure-CPU.
- `tests/gui/test_layout.py` — pure-CPU.
- `tests/gui/test_widgets.py` — pure-CPU.
- `tests/gui/test_painter.py` — pure-CPU.
- `tests/gui/test_material.py` — pure-CPU (introspects `_compile()` output, no device).
- `tests/gui/test_render_gpu.py` — GPU-gated, uses `_make_offscreen_engine` pattern.

Each file has a single responsibility; each task ships independently green.

---

## Conventions (repeat per design / AGENTS.md)

- **Per-test invocation:** `uv run pytest tests/gui/<file>.py::<test> -v`
- **Full suite:** `make test`
- **Lint:** `make lint` (ruff check). Must stay clean.
- **Commit per task:** conventional commit, `feat(gui): …` / `test(gui): …` / `refactor(gui): …` / `docs(gui): …`. Each task ends with a single commit.
- **TDD discipline:** failing test first → confirm fail → minimal implementation → confirm pass → commit. Do not skip the "confirm fail" step.

---

## Task 1: Style system (theme, classes, overrides, parsers)

**Files:**
- Create: `src/manifoldx/gui/__init__.py`
- Create: `src/manifoldx/gui/style.py`
- Create: `tests/gui/__init__.py`
- Create: `tests/gui/test_style.py`

### Step 1.1: Create empty package + test scaffolding

- [ ] **Create `src/manifoldx/gui/__init__.py`** with placeholder content:

```python
"""ManifoldX in-engine GUI layer.

Public API will be filled in as Plan 1 tasks land.
"""

from manifoldx.gui import style  # noqa: F401
```

- [ ] **Create `tests/gui/__init__.py`** as an empty file (one empty line).

### Step 1.2: Write failing tests for `style.set_theme` and `style.resolve`

- [ ] **Create `tests/gui/test_style.py`:**

```python
"""Tests for manifoldx.gui.style — theme, named classes, overrides."""

import pytest

from manifoldx.gui import style


def setup_function(_):
    style.reset()  # Clear theme + named classes between tests.


def test_default_theme_resolves_when_no_class_or_overrides():
    resolved = style.resolve(class_name=None, overrides=None)
    # Defaults from the design doc style table.
    assert resolved["bg"] == "#222"
    assert resolved["fg"] == "#ddd"
    assert resolved["font_size"] == 12
    assert resolved["padding"] == 4
    assert resolved["gap"] == 4
    assert resolved["border"] == 0
    assert resolved["radius"] == 0
    assert resolved["direction"] == "v"


def test_set_theme_overrides_defaults():
    style.set_theme({"bg": "#111", "font_size": 16})
    resolved = style.resolve(class_name=None, overrides=None)
    assert resolved["bg"] == "#111"
    assert resolved["font_size"] == 16
    # Unchanged defaults still present.
    assert resolved["fg"] == "#ddd"


def test_define_class_and_resolve_merges_theme_under_class():
    style.set_theme({"bg": "#222", "fg": "#ddd"})
    style.define("hud", {"bg": "#1a1a1aD0", "padding": 8})
    resolved = style.resolve(class_name="hud", overrides=None)
    assert resolved["bg"] == "#1a1a1aD0"
    assert resolved["padding"] == 8
    # Theme fg still present.
    assert resolved["fg"] == "#ddd"


def test_overrides_take_precedence_over_class_and_theme():
    style.set_theme({"radius": 0})
    style.define("danger", {"radius": 3})
    resolved = style.resolve(class_name="danger", overrides={"radius": 8})
    assert resolved["radius"] == 8


def test_resolve_unknown_class_raises():
    with pytest.raises(KeyError):
        style.resolve(class_name="does_not_exist", overrides=None)
```

- [ ] **Run:** `uv run pytest tests/gui/test_style.py -v`
- [ ] **Expected:** all tests FAIL — `module 'manifoldx.gui.style' has no attribute 'reset' / 'set_theme' / 'resolve' / 'define'`.

### Step 1.3: Implement `style.py` (theme + classes + resolve)

- [ ] **Create `src/manifoldx/gui/style.py`:**

```python
"""Style resolution for the GUI layer.

Three layers, merged at paint time in this order (later wins):

    theme  →  named class  →  per-widget overrides

The style vocabulary is fixed (see DEFAULT_THEME keys). Unknown keys are
ignored at resolve time but preserved if you set them in a class — they
just never affect rendering.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_THEME: dict[str, Any] = {
    "bg": "#222",
    "fg": "#ddd",
    "font_size": 12,
    "padding": 4,
    "gap": 4,
    "border": 0,
    "border_color": "#444",
    "radius": 0,
    "width": None,
    "height": None,
    "flex": None,
    "direction": "v",
}

_theme: dict[str, Any] = deepcopy(DEFAULT_THEME)
_classes: dict[str, dict[str, Any]] = {}


def reset() -> None:
    """Reset theme to defaults and clear named classes. For tests."""
    global _theme, _classes
    _theme = deepcopy(DEFAULT_THEME)
    _classes = {}


def set_theme(theme: dict[str, Any]) -> None:
    """Merge `theme` over the current theme. Keys absent from `theme`
    keep their previous values."""
    _theme.update(theme)


def define(class_name: str, props: dict[str, Any]) -> None:
    """Register or replace a named style class."""
    _classes[class_name] = dict(props)


def resolve(
    class_name: str | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return effective style: theme < class < overrides."""
    out = dict(_theme)
    if class_name is not None:
        if class_name not in _classes:
            raise KeyError(f"unknown gui style class: {class_name!r}")
        out.update(_classes[class_name])
    if overrides is not None:
        out.update(overrides)
    return out
```

### Step 1.4: Run tests and confirm pass

- [ ] **Run:** `uv run pytest tests/gui/test_style.py -v`
- [ ] **Expected:** all 5 tests PASS.

### Step 1.5: Add color parser tests (failing)

- [ ] **Append to `tests/gui/test_style.py`:**

```python
def test_color_parser_short_hex():
    assert style.parse_color("#abc") == (
        pytest.approx(0xaa / 255),
        pytest.approx(0xbb / 255),
        pytest.approx(0xcc / 255),
        pytest.approx(1.0),
    )


def test_color_parser_full_hex():
    assert style.parse_color("#112233") == (
        pytest.approx(0x11 / 255),
        pytest.approx(0x22 / 255),
        pytest.approx(0x33 / 255),
        pytest.approx(1.0),
    )


def test_color_parser_hex_with_alpha():
    assert style.parse_color("#11223344") == (
        pytest.approx(0x11 / 255),
        pytest.approx(0x22 / 255),
        pytest.approx(0x33 / 255),
        pytest.approx(0x44 / 255),
    )


def test_color_parser_rejects_garbage():
    with pytest.raises(ValueError):
        style.parse_color("not a color")
    with pytest.raises(ValueError):
        style.parse_color("#xyzxyz")
```

- [ ] **Run:** `uv run pytest tests/gui/test_style.py -v -k parser`
- [ ] **Expected:** the 4 new tests FAIL — `module has no attribute 'parse_color'`.

### Step 1.6: Implement `parse_color`

- [ ] **Append to `src/manifoldx/gui/style.py`:**

```python
def parse_color(hex_str: str) -> tuple[float, float, float, float]:
    """Parse `#rgb`, `#rrggbb`, or `#rrggbbaa` to (r, g, b, a) floats in [0,1].

    Linear sRGB pass-through — the renderer handles gamma if needed.
    """
    if not isinstance(hex_str, str) or not hex_str.startswith("#"):
        raise ValueError(f"not a hex color: {hex_str!r}")
    body = hex_str[1:]
    if len(body) == 3:
        # Expand each digit: #abc -> #aabbcc.
        body = "".join(c + c for c in body) + "ff"
    elif len(body) == 6:
        body = body + "ff"
    elif len(body) == 8:
        pass
    else:
        raise ValueError(f"invalid hex length: {hex_str!r}")
    try:
        r = int(body[0:2], 16) / 255.0
        g = int(body[2:4], 16) / 255.0
        b = int(body[4:6], 16) / 255.0
        a = int(body[6:8], 16) / 255.0
    except ValueError as e:
        raise ValueError(f"invalid hex digits: {hex_str!r}") from e
    return (r, g, b, a)
```

- [ ] **Run:** `uv run pytest tests/gui/test_style.py -v -k parser`
- [ ] **Expected:** all 4 parser tests PASS.

### Step 1.7: Add padding parser tests (failing)

- [ ] **Append to `tests/gui/test_style.py`:**

```python
def test_padding_parser_int():
    assert style.parse_padding(8) == (8, 8, 8, 8)


def test_padding_parser_single_value_string():
    assert style.parse_padding("8") == (8, 8, 8, 8)


def test_padding_parser_two_values_vertical_horizontal():
    # CSS shorthand: "v h" -> top=bottom=v, right=left=h.
    assert style.parse_padding("6 12") == (6, 12, 6, 12)


def test_padding_parser_four_values_top_right_bottom_left():
    assert style.parse_padding("1 2 3 4") == (1, 2, 3, 4)


def test_padding_parser_rejects_three_values():
    with pytest.raises(ValueError):
        style.parse_padding("1 2 3")
```

- [ ] **Run:** `uv run pytest tests/gui/test_style.py -v -k padding`
- [ ] **Expected:** 5 tests FAIL — no `parse_padding`.

### Step 1.8: Implement `parse_padding`

- [ ] **Append to `src/manifoldx/gui/style.py`:**

```python
def parse_padding(value: int | str) -> tuple[int, int, int, int]:
    """Return (top, right, bottom, left) in pixels.

    Accepts either an int (all four equal) or a string of 1, 2, or 4
    space-separated ints. CSS-style 2-value shorthand: "v h" -> top=bottom=v,
    right=left=h. 3-value is rejected (rare in style sheets, error-prone).
    """
    if isinstance(value, int):
        return (value, value, value, value)
    if not isinstance(value, str):
        raise ValueError(f"padding must be int or str, got {type(value).__name__}")
    parts = value.split()
    nums = [int(p) for p in parts]
    if len(nums) == 1:
        v = nums[0]
        return (v, v, v, v)
    if len(nums) == 2:
        v, h = nums
        return (v, h, v, h)
    if len(nums) == 4:
        t, r, b, l = nums
        return (t, r, b, l)
    raise ValueError(f"padding string must have 1, 2, or 4 values; got {len(nums)}")
```

- [ ] **Run full file:** `uv run pytest tests/gui/test_style.py -v`
- [ ] **Expected:** all tests PASS.

### Step 1.9: Lint and commit

- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.
- [ ] **Commit:**

```bash
git add src/manifoldx/gui/__init__.py src/manifoldx/gui/style.py \
        tests/gui/__init__.py tests/gui/test_style.py
git commit -m "feat(gui): style system — theme, named classes, overrides, parsers"
```

---

## Task 2: Layout algorithm (stack + flex)

**Files:**
- Create: `src/manifoldx/gui/layout.py`
- Create: `tests/gui/test_layout.py`

### Step 2.1: Write failing tests for `compute_layout`

The layout module operates on lightweight "spec" dicts (not Widget objects) so it can be tested without dragging in widget classes. The widget code in Task 3 will build these specs from its own tree and call into here.

- [ ] **Create `tests/gui/test_layout.py`:**

```python
"""Tests for manifoldx.gui.layout — stack + flex algorithm."""

import pytest

from manifoldx.gui.layout import LayoutBox, compute_layout


def _leaf(width=None, height=None, flex=None):
    return {
        "direction": "v",
        "padding": (0, 0, 0, 0),
        "gap": 0,
        "width": width,
        "height": height,
        "flex": flex,
        "intrinsic": (10, 10),
        "children": [],
    }


def _container(direction="v", padding=(0, 0, 0, 0), gap=0, children=None):
    return {
        "direction": direction,
        "padding": padding,
        "gap": gap,
        "width": None,
        "height": None,
        "flex": None,
        "intrinsic": (0, 0),
        "children": children or [],
    }


def test_single_leaf_fills_container():
    root = _container(children=[_leaf()])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 50))
    assert boxes[id(root)] == LayoutBox(0, 0, 100, 50)
    assert boxes[id(root["children"][0])] == LayoutBox(0, 0, 100, 50)


def test_vertical_stack_with_fixed_heights():
    a = _leaf(height=10)
    b = _leaf(height=20)
    root = _container(direction="v", children=[a, b])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    assert boxes[id(a)] == LayoutBox(0, 0, 100, 10)
    assert boxes[id(b)] == LayoutBox(0, 10, 100, 20)


def test_horizontal_stack_with_fixed_widths():
    a = _leaf(width=10)
    b = _leaf(width=20)
    root = _container(direction="h", children=[a, b])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 50))
    assert boxes[id(a)] == LayoutBox(0, 0, 10, 50)
    assert boxes[id(b)] == LayoutBox(10, 0, 20, 50)


def test_flex_distributes_remaining_axis_space():
    a = _leaf(height=10)             # fixed
    b = _leaf(flex=1)                # gets 1 share of the rest
    c = _leaf(flex=3)                # gets 3 shares of the rest
    root = _container(direction="v", children=[a, b, c])
    # viewport=100h, fixed=10, remaining=90, shares=4 → b=22.5, c=67.5
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    assert boxes[id(a)] == LayoutBox(0, 0, 100, 10)
    assert boxes[id(b)] == LayoutBox(0, 10, 100, pytest.approx(22.5))
    assert boxes[id(c)] == LayoutBox(0, pytest.approx(32.5), 100, pytest.approx(67.5))


def test_padding_shrinks_children_area():
    a = _leaf()
    root = _container(padding=(2, 4, 6, 8), children=[a])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    # Children area: x=8, y=2, w=100-4-8=88, h=100-2-6=92.
    assert boxes[id(a)] == LayoutBox(8, 2, 88, 92)


def test_gap_separates_siblings_in_stack():
    a = _leaf(height=10)
    b = _leaf(height=10)
    root = _container(direction="v", gap=5, children=[a, b])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 100))
    assert boxes[id(a)].y == 0
    assert boxes[id(b)].y == 10 + 5  # gap inserted between siblings


def test_nested_panels_layout_independently():
    inner_leaf = _leaf(width=20, height=10)
    inner = _container(direction="h", padding=(1, 1, 1, 1), children=[inner_leaf])
    outer = _container(direction="v", padding=(5, 5, 5, 5), children=[inner])
    boxes = compute_layout(outer, viewport=LayoutBox(0, 0, 100, 100))
    # outer children area: x=5,y=5,w=90,h=90 → inner gets that whole box.
    assert boxes[id(inner)] == LayoutBox(5, 5, 90, 90)
    # inner children area: x=6,y=6,w=88,h=88 → leaf positioned at top-left.
    assert boxes[id(inner_leaf)] == LayoutBox(6, 6, 20, 10)


def test_intrinsic_size_used_when_no_explicit_size_or_flex():
    leaf = _leaf()
    leaf["intrinsic"] = (30, 8)
    root = _container(direction="v", children=[leaf])
    boxes = compute_layout(root, viewport=LayoutBox(0, 0, 100, 50))
    # On the cross axis the leaf still fills (no per-child cross-alignment in v1).
    # On the main axis it gets its intrinsic height.
    assert boxes[id(leaf)] == LayoutBox(0, 0, 100, 8)
```

- [ ] **Run:** `uv run pytest tests/gui/test_layout.py -v`
- [ ] **Expected:** all 8 tests FAIL — `No module named 'manifoldx.gui.layout'`.

### Step 2.2: Implement `layout.py`

- [ ] **Create `src/manifoldx/gui/layout.py`:**

```python
"""Stack + flex layout algorithm for the GUI layer.

Operates on plain spec dicts so it stays decoupled from widget classes.
A spec is::

    {
        "direction": "v" | "h",
        "padding":   (top, right, bottom, left),
        "gap":       int,
        "width":     int | None,        # explicit size; overrides intrinsic
        "height":    int | None,
        "flex":      int | None,        # share of free space (containers only)
        "intrinsic": (w, h),            # default size if no explicit/flex
        "children":  [spec, ...],
    }

`compute_layout(root, viewport)` returns ``{id(spec): LayoutBox}`` for every
spec in the tree, keyed by Python object id (callers retain the spec
references themselves).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LayoutBox:
    x: float
    y: float
    w: float
    h: float


def compute_layout(
    root: dict[str, Any],
    viewport: LayoutBox,
) -> dict[int, LayoutBox]:
    """Compute boxes for every node in the tree rooted at `root`."""
    out: dict[int, LayoutBox] = {}
    _layout_node(root, viewport, out)
    return out


def _layout_node(
    node: dict[str, Any],
    box: LayoutBox,
    out: dict[int, LayoutBox],
) -> None:
    out[id(node)] = box
    children = node.get("children") or []
    if not children:
        return

    top, right, bottom, left = node["padding"]
    inner = LayoutBox(
        box.x + left,
        box.y + top,
        max(0.0, box.w - left - right),
        max(0.0, box.h - top - bottom),
    )
    direction = node["direction"]
    gap = node["gap"]
    main_axis = "h" if direction == "v" else "w"
    main_size = getattr(inner, main_axis)

    total_gap = gap * max(0, len(children) - 1)
    free = main_size - total_gap

    # First pass: subtract fixed + intrinsic-without-flex from `free`.
    main_sizes: list[float] = []
    total_flex = 0
    for child in children:
        flex = child.get("flex")
        explicit = child.get("height") if direction == "v" else child.get("width")
        if flex is not None:
            total_flex += flex
            main_sizes.append(0.0)  # filled in second pass.
            continue
        if explicit is not None:
            size = float(explicit)
        else:
            iw, ih = child["intrinsic"]
            size = float(ih if direction == "v" else iw)
        main_sizes.append(size)
        free -= size

    # Second pass: distribute remaining `free` among flex children.
    if total_flex > 0 and free > 0:
        for i, child in enumerate(children):
            flex = child.get("flex")
            if flex is None:
                continue
            main_sizes[i] = free * (flex / total_flex)

    # Place children sequentially along the main axis.
    cursor = 0.0
    for child, size in zip(children, main_sizes):
        if direction == "v":
            child_box = LayoutBox(inner.x, inner.y + cursor, inner.w, size)
        else:
            child_box = LayoutBox(inner.x + cursor, inner.y, size, inner.h)
        _layout_node(child, child_box, out)
        cursor += size + gap
```

### Step 2.3: Run tests and confirm pass

- [ ] **Run:** `uv run pytest tests/gui/test_layout.py -v`
- [ ] **Expected:** all 8 tests PASS.

### Step 2.4: Lint and commit

- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.
- [ ] **Commit:**

```bash
git add src/manifoldx/gui/layout.py tests/gui/test_layout.py
git commit -m "feat(gui): layout algorithm — stack + flex on spec dicts"
```

---

## Task 3: Widget base + Panel + Text

**Files:**
- Create: `src/manifoldx/gui/widgets.py`
- Modify: `src/manifoldx/gui/__init__.py`
- Create: `tests/gui/test_widgets.py`

### Step 3.1: Write failing tests for `Widget`, `Panel`, `Text`

- [ ] **Create `tests/gui/test_widgets.py`:**

```python
"""Tests for manifoldx.gui.widgets — Widget base, Panel, Text."""

import pytest

from manifoldx.gui import style
from manifoldx.gui.widgets import Panel, Text, Widget, _GuiRoot


def setup_function(_):
    style.reset()


def test_text_construction_keeps_string_and_default_style():
    t = Text("hello")
    assert t.text == "hello"
    assert t.style is None
    assert t.style_overrides == {}


def test_text_effective_style_resolves_theme_then_class_then_overrides():
    style.set_theme({"font_size": 12})
    style.define("big", {"font_size": 20})
    t = Text("x", style="big", style_overrides={"font_size": 30})
    assert t.effective_style()["font_size"] == 30


def test_text_intrinsic_size_grows_with_font_size():
    # Intrinsic size is a function of font_size and char count;
    # exact values depend on the rasterizer, but the relationship must hold.
    small = Text("hello", style_overrides={"font_size": 10})
    big = Text("hello", style_overrides={"font_size": 40})
    sw, sh = small.intrinsic_size()
    bw, bh = big.intrinsic_size()
    assert bw > sw
    assert bh > sh


def test_panel_holds_children_in_order():
    a, b = Text("a"), Text("b")
    p = Panel(children=[a, b])
    assert list(p.children) == [a, b]


def test_panel_anchor_defaults_to_top_left():
    p = Panel(children=[])
    assert p.anchor == "top-left"
    assert p.offset == (0, 0)


def test_panel_rejects_unknown_anchor():
    with pytest.raises(ValueError):
        Panel(children=[], anchor="nowhere")


def test_panel_build_spec_includes_padding_gap_direction_from_style():
    style.set_theme({"padding": 8, "gap": 4, "direction": "h"})
    p = Panel(children=[Text("a"), Text("b")])
    spec = p.build_layout_spec()
    assert spec["padding"] == (8, 8, 8, 8)
    assert spec["gap"] == 4
    assert spec["direction"] == "h"
    assert len(spec["children"]) == 2


def test_widget_is_abstract():
    with pytest.raises(TypeError):
        Widget()  # type: ignore[abstract]


def test_gui_root_is_listlike_and_has_pointer_over_gui_flag():
    g = _GuiRoot()
    assert list(g) == []
    p = Panel(children=[])
    g.append(p)
    assert list(g) == [p]
    assert len(g) == 1
    assert g[0] is p
    assert g.pointer_over_gui is False
    g.pointer_over_gui = True
    assert g.pointer_over_gui is True
```

- [ ] **Run:** `uv run pytest tests/gui/test_widgets.py -v`
- [ ] **Expected:** all 9 tests FAIL.

### Step 3.2: Implement `widgets.py`

- [ ] **Create `src/manifoldx/gui/widgets.py`:**

```python
"""Widget classes for the GUI layer.

This module defines the non-interactive widgets — interactive ones
(Button/Slider/Toggle) land in Plan 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from manifoldx.gui import style
from manifoldx.gui.style import parse_padding

# Anchors are corners on the viewport; offset is added in pixels.
_VALID_ANCHORS = frozenset({
    "top-left", "top-right", "top-center",
    "bottom-left", "bottom-right", "bottom-center",
    "center-left", "center-right", "center",
})


class Widget(ABC):
    """Base class for all GUI widgets.

    A widget owns its style references (`style` is a class name, `style_overrides`
    is a per-instance dict) and exposes the surface the layout, painter, and
    bridge need.
    """

    def __init__(
        self,
        *,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.style = style
        self.style_overrides = dict(style_overrides) if style_overrides else {}

    def effective_style(self) -> dict[str, Any]:
        return _style_resolve(self.style, self.style_overrides)

    @abstractmethod
    def intrinsic_size(self) -> tuple[float, float]:
        """(width, height) in pixels — used when no explicit size or flex."""


def _style_resolve(class_name: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    # Indirection so tests can monkeypatch resolution if needed.
    return style.resolve(class_name, overrides)


class Panel(Widget):
    """A container of widgets. Lays out children in a stack (vertical or
    horizontal) per its style `direction`, with `padding` around and `gap`
    between children.

    `anchor` and `offset` apply only to root panels (those added directly to
    `engine.gui`). Nested panels are positioned by the parent layout and
    ignore these fields.
    """

    def __init__(
        self,
        children: list[Widget] | None = None,
        *,
        anchor: str = "top-left",
        offset: tuple[int, int] = (0, 0),
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if anchor not in _VALID_ANCHORS:
            raise ValueError(
                f"invalid anchor {anchor!r}; expected one of {sorted(_VALID_ANCHORS)}"
            )
        self.children: list[Widget] = list(children or [])
        self.anchor = anchor
        self.offset = tuple(offset)

    def intrinsic_size(self) -> tuple[float, float]:
        # Containers without explicit size grow to fit their viewport slot;
        # the intrinsic value is a fallback for nested-without-flex cases.
        return (0.0, 0.0)

    def build_layout_spec(self) -> dict[str, Any]:
        s = self.effective_style()
        return {
            "direction": s["direction"],
            "padding": parse_padding(s["padding"]),
            "gap": int(s["gap"]),
            "width": s.get("width"),
            "height": s.get("height"),
            "flex": s.get("flex"),
            "intrinsic": self.intrinsic_size(),
            "children": [_child_spec(c) for c in self.children],
        }


def _child_spec(widget: Widget) -> dict[str, Any]:
    """Convert a widget into a layout spec dict. Defers to Panel for nested
    containers; leaves use the widget's intrinsic_size + style."""
    if isinstance(widget, Panel):
        return widget.build_layout_spec()
    s = widget.effective_style()
    return {
        "direction": "v",  # ignored for leaves
        "padding": (0, 0, 0, 0),
        "gap": 0,
        "width": s.get("width"),
        "height": s.get("height"),
        "flex": s.get("flex"),
        "intrinsic": widget.intrinsic_size(),
        "children": [],
    }


class Text(Widget):
    """A static text label. Intrinsic size is derived from rasterized glyph
    extents at the effective font size; cached on construction (re-measured
    if you mutate `.text`)."""

    def __init__(
        self,
        text: str,
        *,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        self._text = text
        self._cached_intrinsic: tuple[float, float] | None = None

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if value != self._text:
            self._cached_intrinsic = None
        self._text = value

    def intrinsic_size(self) -> tuple[float, float]:
        if self._cached_intrinsic is None:
            font_size = int(self.effective_style().get("font_size", 12))
            self._cached_intrinsic = _measure_text(self._text, font_size)
        return self._cached_intrinsic


def _measure_text(text: str, font_size: int) -> tuple[float, float]:
    """Approximate the rasterized extents without going through wgpu.

    We deliberately use a coarse linear model here (no PIL dependency in
    Plan 1) — the actual atlas-side measurement lands in Task 7, and any
    discrepancy at that point triggers a re-layout via the dirty bit.
    """
    char_w = font_size * 0.55
    return (char_w * max(1, len(text)), float(font_size) * 1.25)


class _GuiRoot:
    """Container exposed as `engine.gui`. List-like over root panels, plus
    a `pointer_over_gui` flag the bridge will toggle (Plan 2)."""

    def __init__(self) -> None:
        self._panels: list[Panel] = []
        self.pointer_over_gui: bool = False

    def append(self, panel: Panel) -> None:
        self._panels.append(panel)

    def remove(self, panel: Panel) -> None:
        self._panels.remove(panel)

    def clear(self) -> None:
        self._panels.clear()

    def __iter__(self):
        return iter(self._panels)

    def __len__(self) -> int:
        return len(self._panels)

    def __getitem__(self, i: int) -> Panel:
        return self._panels[i]
```

### Step 3.3: Update `gui/__init__.py` to export Panel and Text

- [ ] **Edit `src/manifoldx/gui/__init__.py`:**

```python
"""ManifoldX in-engine GUI layer.

Public API:

- `Panel`, `Text`, `ValueDisplay` — non-interactive widgets (Plan 1).
- `Button`, `Slider`, `Toggle` — interactive widgets (Plan 2, not yet exposed).
- `style` — theme + named classes + per-widget overrides.
"""

from manifoldx.gui import style  # noqa: F401
from manifoldx.gui.widgets import Panel, Text  # noqa: F401

__all__ = ["Panel", "Text", "style"]
```

### Step 3.4: Run tests and confirm pass

- [ ] **Run:** `uv run pytest tests/gui/test_widgets.py -v`
- [ ] **Expected:** all 9 tests PASS.

### Step 3.5: Lint and commit

- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.
- [ ] **Commit:**

```bash
git add src/manifoldx/gui/__init__.py src/manifoldx/gui/widgets.py \
        tests/gui/test_widgets.py
git commit -m "feat(gui): Widget base + Panel + Text + _GuiRoot container"
```

---

## Task 4: Painter accumulator + tree walker

**Files:**
- Create: `src/manifoldx/gui/painter.py`
- Create: `tests/gui/test_painter.py`

### Step 4.1: Write failing tests for `Painter` + `paint`

- [ ] **Create `tests/gui/test_painter.py`:**

```python
"""Tests for manifoldx.gui.painter — rect + text op accumulation."""

from manifoldx.gui import style
from manifoldx.gui.layout import LayoutBox, compute_layout
from manifoldx.gui.painter import Painter, paint
from manifoldx.gui.widgets import Panel, Text


def setup_function(_):
    style.reset()


def test_painter_starts_empty():
    p = Painter()
    assert p.rect_ops == []
    assert p.text_ops == []


def test_painter_draw_rect_records_op():
    p = Painter()
    p.draw_rect(
        box=LayoutBox(10, 20, 100, 50),
        fill=(1, 0, 0, 1),
        border_color=(0, 1, 0, 1),
        border=2.0,
        radius=4.0,
    )
    assert len(p.rect_ops) == 1
    op = p.rect_ops[0]
    assert op.box == LayoutBox(10, 20, 100, 50)
    assert op.fill == (1, 0, 0, 1)
    assert op.border == 2.0
    assert op.radius == 4.0


def test_painter_draw_text_records_op():
    p = Painter()
    p.draw_text(box=LayoutBox(0, 0, 50, 12), text="hi", font_size=12, fg=(1, 1, 1, 1))
    assert len(p.text_ops) == 1
    op = p.text_ops[0]
    assert op.text == "hi"
    assert op.font_size == 12


def test_paint_walks_panel_emitting_rect_then_children():
    style.set_theme({"bg": "#222", "padding": 0, "gap": 0})
    style.define("filled", {"bg": "#ff0000"})
    panel = Panel(children=[Text("hi")], style="filled")
    spec = panel.build_layout_spec()
    boxes = compute_layout(spec, viewport=LayoutBox(0, 0, 100, 100))
    p = Painter()
    paint(panel, spec, boxes, p)
    # Panel emits a rect with its bg.
    assert len(p.rect_ops) == 1
    assert p.rect_ops[0].fill[0] == 1.0  # red channel from #ff0000
    # Child Text emits exactly one text op.
    assert len(p.text_ops) == 1
    assert p.text_ops[0].text == "hi"


def test_paint_nested_panels_walks_depth_first():
    inner = Panel(children=[Text("inner")])
    outer = Panel(children=[Text("outer"), inner])
    spec = outer.build_layout_spec()
    boxes = compute_layout(spec, viewport=LayoutBox(0, 0, 200, 200))
    p = Painter()
    paint(outer, spec, boxes, p)
    # Two panels → 2 rects; two text widgets → 2 text ops.
    assert len(p.rect_ops) == 2
    assert {op.text for op in p.text_ops} == {"inner", "outer"}
```

- [ ] **Run:** `uv run pytest tests/gui/test_painter.py -v`
- [ ] **Expected:** 5 tests FAIL — no painter module.

### Step 4.2: Implement `painter.py`

- [ ] **Create `src/manifoldx/gui/painter.py`:**

```python
"""Painter accumulator for the GUI layer.

The painter is the seam between the widget tree (pure Python, layout-aware)
and the render pass (GPU-aware, batches into two instanced draws).

Painting walks the tree top-down, emitting:

- One `RectOp` per Panel (and, in Plan 2, per Button/Slider/Toggle).
- One `TextOp` per Text / ValueDisplay glyph batch.

Ops carry resolved pixel boxes and resolved colors — the painter does no
style lookup itself; widgets pass already-resolved styles through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.style import parse_color
from manifoldx.gui.widgets import Panel, Text, Widget


@dataclass(frozen=True, slots=True)
class RectOp:
    box: LayoutBox
    fill: tuple[float, float, float, float]
    border_color: tuple[float, float, float, float]
    border: float
    radius: float


@dataclass(frozen=True, slots=True)
class TextOp:
    box: LayoutBox
    text: str
    font_size: int
    fg: tuple[float, float, float, float]


@dataclass
class Painter:
    rect_ops: list[RectOp] = field(default_factory=list)
    text_ops: list[TextOp] = field(default_factory=list)

    def draw_rect(
        self,
        *,
        box: LayoutBox,
        fill: tuple[float, float, float, float],
        border_color: tuple[float, float, float, float] = (0, 0, 0, 0),
        border: float = 0.0,
        radius: float = 0.0,
    ) -> None:
        self.rect_ops.append(
            RectOp(
                box=box,
                fill=fill,
                border_color=border_color,
                border=border,
                radius=radius,
            )
        )

    def draw_text(
        self,
        *,
        box: LayoutBox,
        text: str,
        font_size: int,
        fg: tuple[float, float, float, float],
    ) -> None:
        self.text_ops.append(TextOp(box=box, text=text, font_size=font_size, fg=fg))

    def clear(self) -> None:
        self.rect_ops.clear()
        self.text_ops.clear()


def paint(
    widget: Widget,
    spec: dict[str, Any],
    boxes: dict[int, LayoutBox],
    painter: Painter,
) -> None:
    """Walk the widget tree top-down, emitting ops into `painter`.

    `spec` is the layout spec for `widget` (as produced by Panel.build_layout_spec
    or the equivalent for a leaf); `boxes` is the result of compute_layout.
    """
    box = boxes[id(spec)]
    if isinstance(widget, Panel):
        s = widget.effective_style()
        painter.draw_rect(
            box=box,
            fill=parse_color(s["bg"]),
            border_color=parse_color(s["border_color"]),
            border=float(s["border"]),
            radius=float(s["radius"]),
        )
        for child, child_spec in zip(widget.children, spec["children"]):
            paint(child, child_spec, boxes, painter)
    elif isinstance(widget, Text):
        s = widget.effective_style()
        painter.draw_text(
            box=box,
            text=widget.text,
            font_size=int(s["font_size"]),
            fg=parse_color(s["fg"]),
        )
    # Plan 2 will add ValueDisplay / Button / Slider / Toggle branches.
```

### Step 4.3: Run tests and confirm pass

- [ ] **Run:** `uv run pytest tests/gui/test_painter.py -v`
- [ ] **Expected:** all 5 tests PASS.

### Step 4.4: Lint and commit

- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.
- [ ] **Commit:**

```bash
git add src/manifoldx/gui/painter.py tests/gui/test_painter.py
git commit -m "feat(gui): Painter accumulator and paint() tree walker"
```

---

## Task 5: RectMaterial (WGSL signed-distance rounded rect)

**Files:**
- Create: `src/manifoldx/gui/material.py`
- Create: `tests/gui/test_material.py`

### Step 5.1: Write failing tests for `RectMaterial`

- [ ] **Create `tests/gui/test_material.py`:**

```python
"""Tests for manifoldx.gui.material — RectMaterial WGSL + uniform layout.

No GPU device is required: we introspect _compile() and uniform_type()
directly. End-to-end pipeline-cache differentiation is exercised in
tests/gui/test_render_gpu.py (Task 6).
"""

from manifoldx.gui.material import RectMaterial


def test_rect_material_compile_returns_wgsl_with_sdf_helper():
    src = RectMaterial._compile()
    assert isinstance(src, str)
    # The signed-distance rounded-rect expression is the load-bearing geom.
    # We check for the canonical SDF helper name we'll define.
    assert "rounded_rect_sdf" in src
    # And for the per-instance struct.
    assert "RectInstance" in src
    # Vertex + fragment entry points.
    assert "@vertex" in src
    assert "@fragment" in src


def test_rect_material_uniform_type_describes_globals_and_instances():
    types = RectMaterial.uniform_type()
    # Per-instance fields, in the order they're packed into a row-of-floats.
    expected = {
        "xy": "vec2<f32>",
        "size": "vec2<f32>",
        "radius": "f32",
        "border": "f32",
        "bg": "vec4<f32>",
        "border_color": "vec4<f32>",
    }
    for k, v in expected.items():
        assert types.get(k) == v, f"missing/mismatched uniform field {k}"


def test_rect_material_pipeline_subtype_is_gui():
    # The design says: pipeline-cache 5th element is "gui" so this pass
    # doesn't share pipelines with sci-viz label.
    assert RectMaterial().pipeline_subtype == "gui"
```

- [ ] **Run:** `uv run pytest tests/gui/test_material.py -v`
- [ ] **Expected:** 3 tests FAIL — no `manifoldx.gui.material`.

### Step 5.2: Implement `RectMaterial`

- [ ] **Create `src/manifoldx/gui/material.py`:**

```python
"""WGSL signed-distance rounded-rect material for the GUI render pass.

One instanced draw call → one rounded rect per instance, with optional 1 px
(or wider) border. Per-instance state is packed into a 12-float row::

    [xy.x, xy.y, size.x, size.y, radius, border, bg.r, bg.g, bg.b, bg.a,
     border_color.r, border_color.g, border_color.b, border_color.a]

Total: 14 floats = 56 bytes per instance. The vertex shader expands a
unit quad in [-0.5, 0.5]^2 to the instance's pixel size at the instance's
top-left position; the fragment shader evaluates the rounded-rect SDF.
"""

from __future__ import annotations

from typing import Any

from manifoldx.resources import Material


_RECT_WGSL = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    viewport_size: vec2<f32>,
    _pad1: vec2<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct RectInstance {
    xy:           vec2<f32>,
    size:         vec2<f32>,
    radius:       f32,
    border:       f32,
    bg:           vec4<f32>,
    border_color: vec4<f32>,
};

@group(0) @binding(1) var<storage, read> instances: array<RectInstance>;

struct VsIn {
    @location(0) position: vec3<f32>,  // unit-quad corner in [-0.5, 0.5]
    @builtin(instance_index) instance_id: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,   // pixel offset from rect center
    @location(1) flat_id: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let inst = instances[in.instance_id];
    // Pixel-space center + corner.
    let center_px = inst.xy + inst.size * 0.5;
    let corner_px = center_px + vec2<f32>(in.position.x, in.position.y) * inst.size;
    // Convert pixel coords -> NDC. Viewport origin is top-left; flip Y.
    let ndc_x = (corner_px.x / globals.viewport_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (corner_px.y / globals.viewport_size.y) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local = vec2<f32>(in.position.x, in.position.y) * inst.size;
    out.flat_id = in.instance_id;
    return out;
}

fn rounded_rect_sdf(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) +
           min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let inst = instances[in.flat_id];
    let half = inst.size * 0.5;
    let d = rounded_rect_sdf(in.local, half, inst.radius);
    // 1 px smoothstep gives cheap AA at the outer edge.
    let outer = 1.0 - smoothstep(-1.0, 0.0, d);
    var color = inst.bg;
    if (inst.border > 0.0) {
        let inside_d = d + inst.border;
        let inner = 1.0 - smoothstep(-1.0, 0.0, inside_d);
        let border_mask = clamp(outer - inner, 0.0, 1.0);
        color = inst.bg * inner + inst.border_color * border_mask;
        // Pre-multiplied alpha for blending.
        color.a = outer;
    } else {
        color = inst.bg * outer;
        color.a = inst.bg.a * outer;
    }
    return color;
}
"""


class RectMaterial(Material):
    """Material for the GUI rect path. One instance per rounded rectangle."""

    binding_slot = 1  # storage buffer at @group(0) @binding(1)

    @classmethod
    def _compile(cls) -> str:
        return _RECT_WGSL

    @classmethod
    def uniform_type(cls) -> dict[str, str]:
        return {
            "xy": "vec2<f32>",
            "size": "vec2<f32>",
            "radius": "f32",
            "border": "f32",
            "bg": "vec4<f32>",
            "border_color": "vec4<f32>",
        }

    @property
    def pipeline_subtype(self) -> str:
        return "gui"

    @staticmethod
    def pack_instances(rect_ops: list[Any]) -> "np.ndarray":  # noqa: F821
        """Pack a list of painter.RectOp into a (N, 14) float32 buffer.

        Column order matches the WGSL `RectInstance` struct layout.
        """
        import numpy as np

        if not rect_ops:
            return np.zeros((0, 14), dtype=np.float32)
        rows = []
        for op in rect_ops:
            rows.append(
                [
                    op.box.x,
                    op.box.y,
                    op.box.w,
                    op.box.h,
                    op.radius,
                    op.border,
                    op.fill[0],
                    op.fill[1],
                    op.fill[2],
                    op.fill[3],
                    op.border_color[0],
                    op.border_color[1],
                    op.border_color[2],
                    op.border_color[3],
                ]
            )
        return np.asarray(rows, dtype=np.float32)
```

### Step 5.3: Run tests and confirm pass

- [ ] **Run:** `uv run pytest tests/gui/test_material.py -v`
- [ ] **Expected:** all 3 tests PASS.

### Step 5.4: Add a pack_instances unit test (failing → passing)

- [ ] **Append to `tests/gui/test_material.py`:**

```python
def test_rect_material_pack_instances_layout():
    import numpy as np
    from manifoldx.gui.layout import LayoutBox
    from manifoldx.gui.painter import RectOp

    ops = [
        RectOp(
            box=LayoutBox(10, 20, 100, 50),
            fill=(0.5, 0.6, 0.7, 0.8),
            border_color=(0.1, 0.2, 0.3, 0.4),
            border=2.0,
            radius=4.0,
        )
    ]
    arr = RectMaterial.pack_instances(ops)
    assert arr.dtype == np.float32
    assert arr.shape == (1, 14)
    np.testing.assert_allclose(
        arr[0],
        [10, 20, 100, 50, 4.0, 2.0, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4],
    )


def test_rect_material_pack_empty_returns_zero_rows():
    import numpy as np
    arr = RectMaterial.pack_instances([])
    assert arr.shape == (0, 14)
    assert arr.dtype == np.float32
```

- [ ] **Run:** `uv run pytest tests/gui/test_material.py -v`
- [ ] **Expected:** all 5 tests PASS.

### Step 5.5: Lint and commit

- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.
- [ ] **Commit:**

```bash
git add src/manifoldx/gui/material.py tests/gui/test_material.py
git commit -m "feat(gui): RectMaterial — WGSL signed-distance rounded rect"
```

---

## Task 6: GUI render pass (rect path only) + Engine integration

**Files:**
- Create: `src/manifoldx/render/passes/gui.py`
- Modify: `src/manifoldx/engine.py`
- Modify: `src/manifoldx/renderer.py`
- Create: `tests/gui/test_render_gpu.py`

This task wires up the rect path end-to-end: a `Panel` with one of each leaf type (in Plan 1 just `Text`, but we'll only check the rect for now since glyphs come in Task 7) should produce a non-empty framebuffer in the panel region.

### Step 6.1: Wire `engine.gui = _GuiRoot()` into `Engine.__init__`

- [ ] **Edit `src/manifoldx/engine.py`** — find the input-bridge construction block (around line 65–67):

```python
        # Input layer
        self.input = InputState()
        self._input_bridge = _InputBridge(self, self.input)
```

Append immediately after:

```python
        # GUI layer
        from manifoldx.gui.widgets import _GuiRoot
        self.gui = _GuiRoot()
```

### Step 6.2: Write a failing GPU smoke test for the rect path

Mirror `tests/test_input_e2e.py`'s offscreen-engine helper.

- [ ] **Create `tests/gui/test_render_gpu.py`:**

```python
"""GPU-gated tests for the gui render pass.

These tests need an offscreen wgpu device; they pytest.skip on machines
without one (matches the rest of the suite's gating).
"""

import pytest

import numpy as np

from manifoldx.gui import Panel, style


def _make_offscreen_engine(width: int = 128, height: int = 128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine, canvas


def setup_function(_):
    style.reset()


def test_engine_exposes_gui_root_listlike_with_pointer_flag():
    engine, _ = _make_offscreen_engine()
    assert hasattr(engine, "gui")
    assert len(engine.gui) == 0
    p = Panel(children=[])
    engine.gui.append(p)
    assert engine.gui[0] is p
    assert engine.gui.pointer_over_gui is False


def test_gui_pass_renders_panel_rect_into_framebuffer():
    style.set_theme({"bg": "#ff0000ff", "padding": 4, "gap": 0})
    engine, canvas = _make_offscreen_engine(width=128, height=128)
    panel = Panel(
        children=[],
        anchor="top-left",
        offset=(10, 20),
        style_overrides={"width": 40, "height": 30},
    )
    engine.gui.append(panel)
    engine._draw_frame()
    # Read back the offscreen target as a numpy array.
    frame = np.asarray(canvas.draw())
    # frame is (H, W, 4) RGBA uint8.
    # Inside panel: pixel should be ~red.
    inside = frame[30, 25]  # y=30 is inside [20, 50], x=25 is inside [10, 50]
    assert inside[0] > 200, f"expected red inside panel, got {inside.tolist()}"
    assert inside[1] < 80
    assert inside[2] < 80
    # Outside panel: should be the default clear color (NOT red).
    outside = frame[100, 100]
    assert outside[0] < 80 or outside[3] == 0, (
        f"expected clear outside panel, got {outside.tolist()}"
    )
```

- [ ] **Run:** `uv run pytest tests/gui/test_render_gpu.py -v`
- [ ] **Expected:** first test PASSES (engine.gui is wired from Step 6.1); second test FAILS — no gui pass registered, panel doesn't render.

### Step 6.3: Implement the gui render pass (rect path only)

- [ ] **Create `src/manifoldx/render/passes/gui.py`:**

```python
"""GUI render pass — runs last in the pipeline.

Issues at most two batched draws per frame (in Plan 1, only the first — the
glyph path lands in Task 7):

1. Rects — one instanced draw via `RectMaterial`.
2. Glyphs — one instanced draw reusing the screen-anchored label pipeline.

Pass attributes (set at pipeline-creation time in `_ensure_gui_pipeline`):
- depth-test off, depth-write off
- alpha blend on (pre-multiplied alpha)
- no culling
"""

from __future__ import annotations

import numpy as np
import wgpu

from manifoldx.gui.layout import LayoutBox, compute_layout
from manifoldx.gui.material import RectMaterial
from manifoldx.gui.painter import Painter, paint
from manifoldx.gui.widgets import Panel


# Unit-quad geometry shared by all rect instances (positions in [-0.5, 0.5]).
_UNIT_QUAD_VERTS = np.array(
    [
        [-0.5, -0.5, 0.0],
        [+0.5, -0.5, 0.0],
        [+0.5, +0.5, 0.0],
        [-0.5, +0.5, 0.0],
    ],
    dtype=np.float32,
)
_UNIT_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


def render_gui_pass(rp, engine, render_pass) -> None:
    """Entry point — called from RenderPipeline.render after the label pass."""
    if not engine.gui:
        return

    viewport = LayoutBox(0.0, 0.0, float(engine.w), float(engine.h))

    painter = Painter()
    for panel in engine.gui:
        spec = panel.build_layout_spec()
        boxes = compute_layout(spec, viewport=_anchored(panel, viewport))
        paint(panel, spec, boxes, painter)

    _ensure_gui_pipeline(rp, engine)

    # --- Rect draw ---
    rect_data = RectMaterial.pack_instances(painter.rect_ops)
    if rect_data.size > 0:
        _upload_rect_instances(rp, rect_data)
        _draw_rects(rp, engine, render_pass, rect_data.shape[0])

    # Glyph draw lands in Task 7.


def _anchored(panel: Panel, viewport: LayoutBox) -> LayoutBox:
    """Translate the viewport box according to the panel's anchor + offset.

    The returned box is the *layout viewport* that compute_layout sees — i.e.
    the panel's own slot — not the full screen.

    For now we anchor top-left only; full anchor map lands here as the GUI
    accumulates examples that need it (`top-right`, `bottom-right`, ...).
    """
    ox, oy = panel.offset
    return LayoutBox(viewport.x + ox, viewport.y + oy, viewport.w, viewport.h)


def _ensure_gui_pipeline(rp, engine) -> None:
    if getattr(rp, "_gui_pipeline", None) is not None:
        return
    device = rp._device
    shader = device.create_shader_module(code=RectMaterial._compile())

    # Bind group: 0 = globals uniform, 1 = instance storage buffer.
    bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

    pipeline = device.create_render_pipeline(
        layout=layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 12,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": wgpu.VertexFormat.float32x3},
                    ],
                },
            ],
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": rp._texture_format,
                    "blend": {
                        "color": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                        "alpha": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                    },
                    "write_mask": wgpu.ColorWrite.ALL,
                }
            ],
        },
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
        depth_stencil=None,  # depth off
        multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
    )
    rp._gui_pipeline = pipeline
    rp._gui_bgl = bgl

    # Unit-quad vertex + index buffers.
    rp._gui_quad_vbuf = device.create_buffer_with_data(
        data=_UNIT_QUAD_VERTS.tobytes(),
        usage=wgpu.BufferUsage.VERTEX,
    )
    rp._gui_quad_ibuf = device.create_buffer_with_data(
        data=_UNIT_QUAD_INDICES.tobytes(),
        usage=wgpu.BufferUsage.INDEX,
    )
    rp._gui_instance_buf = None
    rp._gui_instance_capacity = 0


def _upload_rect_instances(rp, data: np.ndarray) -> None:
    needed = data.nbytes
    if rp._gui_instance_buf is None or needed > rp._gui_instance_capacity:
        rp._gui_instance_buf = rp._device.create_buffer(
            size=max(needed, 4096),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        rp._gui_instance_capacity = max(needed, 4096)
    rp._device.queue.write_buffer(rp._gui_instance_buf, 0, data.tobytes())


def _draw_rects(rp, engine, render_pass, instance_count: int) -> None:
    bind_group = rp._device.create_bind_group(
        layout=rp._gui_bgl,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 224},
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._gui_instance_buf,
                    "offset": 0,
                    "size": rp._gui_instance_capacity,
                },
            },
        ],
    )
    render_pass.set_pipeline(rp._gui_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 0)
    render_pass.set_vertex_buffer(0, rp._gui_quad_vbuf)
    render_pass.set_index_buffer(rp._gui_quad_ibuf, wgpu.IndexFormat.uint32)
    render_pass.draw_indexed(6, instance_count, 0, 0, 0)
```

### Step 6.4: Register the gui pass in `RenderPipeline.render`

- [ ] **Edit `src/manifoldx/renderer.py`** — find the label-pass call (around line 904–907) and add the gui pass immediately after:

```python
        if label_batches:
            _label_pass.render_label_pass(
                self, engine, render_pass, label_batches, model_matrices, material_data
            )

        # GUI pass — last. Reads engine.gui directly; no batches needed.
        from manifoldx.render.passes import gui as _gui_pass
        _gui_pass.render_gui_pass(self, engine, render_pass)
```

### Step 6.5: Run the GPU test and confirm pass

- [ ] **Run:** `uv run pytest tests/gui/test_render_gpu.py -v`
- [ ] **Expected:** both tests PASS. If the second test fails because the panel renders but in the wrong place, debug via offscreen `canvas.draw()` returning the framebuffer — the rect should occupy `(10..50, 20..50)` in pixel coordinates.

### Step 6.6: Run the full suite — no regressions

- [ ] **Run:** `make test`
- [ ] **Expected:** all pre-existing tests still PASS (the new gui pass is a no-op when `engine.gui` is empty).

### Step 6.7: Lint and commit

- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.
- [ ] **Commit:**

```bash
git add src/manifoldx/engine.py src/manifoldx/renderer.py \
        src/manifoldx/render/passes/gui.py tests/gui/test_render_gpu.py
git commit -m "feat(gui): gui render pass + RectMaterial pipeline + engine.gui wiring"
```

---

## Task 7: Glyph path — Text widget renders via LabelTextureAtlas

**Files:**
- Modify: `src/manifoldx/render/passes/gui.py`
- Modify: `tests/gui/test_render_gpu.py`

This extends the gui pass with a second instanced draw call that reuses the existing screen-anchored label pipeline (with a `"gui"` 5th pipeline-cache element so it doesn't clash with the standalone `label` pass). The atlas API (`engine.get_label_atlas()`, `atlas.get_or_create(text, font_size)`) is already in place from sci-viz Plan 2.

### Step 7.1: Write a failing test — Text widget produces text in framebuffer

- [ ] **Append to `tests/gui/test_render_gpu.py`:**

```python
def test_gui_pass_renders_text_widget():
    from manifoldx.gui import Text
    style.set_theme({"bg": "#000000ff", "fg": "#ffffffff", "font_size": 16,
                     "padding": 2, "gap": 0})
    engine, canvas = _make_offscreen_engine(width=160, height=64)
    panel = Panel(
        children=[Text("hello")],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 100, "height": 40},
    )
    engine.gui.append(panel)
    engine._draw_frame()
    frame = np.asarray(canvas.draw())
    # Sum brightness over the text region — non-empty means glyphs drew.
    region = frame[6:30, 4:96, :3].astype(np.int32).sum()
    assert region > 5000, (
        f"expected glyphs to brighten the panel region; got region sum={region}"
    )
```

- [ ] **Run:** `uv run pytest tests/gui/test_render_gpu.py -v -k text`
- [ ] **Expected:** FAIL — gui pass does not yet emit glyph instances.

### Step 7.2: Extend the gui pass to emit glyph instances

The existing label pipeline expects each glyph to be a billboard quad in screen-anchored mode, fed by a TextureAtlas slice index. The simplest path: ask the atlas for a slice per `TextOp.text` (atlas already does `(text, font_size)` deduplication), then issue one instanced draw of `len(text_ops)` quads where each instance reads its xy/size/slice.

For Plan 1 we keep this minimal: one billboard per `TextOp`, not per glyph. Per-glyph subdivision (necessary for kerning + sub-pixel positioning) is out of scope.

- [ ] **Edit `src/manifoldx/render/passes/gui.py`** — add at the top:

```python
from manifoldx.viz.materials import LabelMaterial
```

Then inside `render_gui_pass` after the existing rect-draw block, add:

```python
    # --- Glyph draw ---
    if painter.text_ops:
        atlas = engine.get_label_atlas()
        slice_data = _pack_glyph_instances(painter.text_ops, atlas, engine)
        _ensure_glyph_pipeline(rp, engine)
        _upload_glyph_instances(rp, slice_data)
        _draw_glyphs(rp, engine, render_pass, slice_data.shape[0])
```

Then append these helpers to the same module:

```python
def _pack_glyph_instances(text_ops, atlas, engine) -> np.ndarray:
    """Pack TextOps into a (N, 8) float32 buffer:

        [center_ndc_x, center_ndc_y, pixel_width, pixel_height,
         slice_index, fg_r, fg_g, fg_b]

    Anchor is the center of the text box in NDC; pixel size is the box's
    width/height; slice_index is the atlas layer for the rasterized string.
    Color a is implied = 1.0 in v1 (no per-glyph alpha override).
    """
    rows = []
    for op in text_ops:
        slice_idx = atlas.get_or_create(op.text, font_size=op.font_size)
        cx_px = op.box.x + op.box.w * 0.5
        cy_px = op.box.y + op.box.h * 0.5
        ndc_x = (cx_px / engine.w) * 2.0 - 1.0
        ndc_y = 1.0 - (cy_px / engine.h) * 2.0
        rows.append(
            [ndc_x, ndc_y, op.box.w, op.box.h, float(slice_idx),
             op.fg[0], op.fg[1], op.fg[2]]
        )
    return np.asarray(rows, dtype=np.float32)


def _ensure_glyph_pipeline(rp, engine) -> None:
    """Cache key: (sprite_geom_id, LabelMaterial, "screen", "gui").

    The `"gui"` 5th element ensures we don't share the pipeline with the
    standalone label pass — the bind group layout differs (we read from a
    storage buffer, not the renderer's instance buffer).
    """
    if getattr(rp, "_gui_glyph_pipeline", None) is not None:
        return
    # Build via the same pattern as _ensure_gui_pipeline, but using
    # LabelMaterial's shader. The glyph instances buffer holds (N, 8) floats.
    device = rp._device
    shader = device.create_shader_module(code=LabelMaterial._compile())

    bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": wgpu.TextureViewDimension.d2_array}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.filtering}},
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipeline = device.create_render_pipeline(
        layout=layout,
        vertex={
            "module": shader, "entry_point": "vs_main",
            "buffers": [{
                "array_stride": 12, "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [{"shader_location": 0, "offset": 0,
                                "format": wgpu.VertexFormat.float32x3}],
            }],
        },
        fragment={
            "module": shader, "entry_point": "fs_main",
            "targets": [{
                "format": rp._texture_format,
                "blend": {
                    "color": {"src_factor": wgpu.BlendFactor.one,
                              "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                              "operation": wgpu.BlendOperation.add},
                    "alpha": {"src_factor": wgpu.BlendFactor.one,
                              "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                              "operation": wgpu.BlendOperation.add},
                },
                "write_mask": wgpu.ColorWrite.ALL,
            }],
        },
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                   "cull_mode": wgpu.CullMode.none},
        depth_stencil=None,
        multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
    )
    rp._gui_glyph_pipeline = pipeline
    rp._gui_glyph_bgl = bgl
    rp._gui_glyph_instance_buf = None
    rp._gui_glyph_instance_capacity = 0


def _upload_glyph_instances(rp, data: np.ndarray) -> None:
    needed = data.nbytes
    if rp._gui_glyph_instance_buf is None or needed > rp._gui_glyph_instance_capacity:
        rp._gui_glyph_instance_buf = rp._device.create_buffer(
            size=max(needed, 4096),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        rp._gui_glyph_instance_capacity = max(needed, 4096)
    rp._device.queue.write_buffer(rp._gui_glyph_instance_buf, 0, data.tobytes())


def _draw_glyphs(rp, engine, render_pass, instance_count: int) -> None:
    atlas = engine.get_label_atlas()
    atlas.upload_dirty(rp._device, rp._device.queue)
    bind_group = rp._device.create_bind_group(
        layout=rp._gui_glyph_bgl,
        entries=[
            {"binding": 0,
             "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 224}},
            {"binding": 1,
             "resource": {"buffer": rp._gui_glyph_instance_buf,
                          "offset": 0, "size": rp._gui_glyph_instance_capacity}},
            {"binding": 2, "resource": atlas.gpu_texture_view},
            {"binding": 3, "resource": atlas.sampler},
        ],
    )
    render_pass.set_pipeline(rp._gui_glyph_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 0)
    render_pass.set_vertex_buffer(0, rp._gui_quad_vbuf)
    render_pass.set_index_buffer(rp._gui_quad_ibuf, wgpu.IndexFormat.uint32)
    render_pass.draw_indexed(6, instance_count, 0, 0, 0)
```

> **Note for the implementer:** The glyph WGSL above reuses `LabelMaterial._compile()` directly. If the existing LabelMaterial shader expects a different storage-buffer layout than the (N, 8) packed-floats we're providing, write a new minimal shader inline in `gui.py` rather than fighting the abstraction — keeping LabelMaterial untouched. The 5th-cache-element rule is conceptual: it formalizes that this is a separate pipeline; in practice we own it locally.
>
> Inspect `src/manifoldx/viz/materials.py` (LabelMaterial) before deciding; if its bindgroup layout differs from what's coded here (e.g. it pulls per-instance uniforms from a different binding), define a small `_GLYPH_WGSL` constant in `gui.py` analogous to `_RECT_WGSL` and use it instead. Document the choice in a one-line code comment.

### Step 7.3: Run the failing glyph test

- [ ] **Run:** `uv run pytest tests/gui/test_render_gpu.py -v -k text`
- [ ] **Expected:** PASS. If the shader is incompatible (per the note above), drop in a local `_GLYPH_WGSL` mirroring `LabelMaterial`'s screen-anchored path against an (N, 8) storage buffer and re-run.

### Step 7.4: Run the full suite — no regressions

- [ ] **Run:** `make test`
- [ ] **Expected:** all pre-existing tests still PASS.

### Step 7.5: Lint and commit

- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.
- [ ] **Commit:**

```bash
git add src/manifoldx/render/passes/gui.py tests/gui/test_render_gpu.py
git commit -m "feat(gui): glyph draw path — Text widgets render via LabelTextureAtlas"
```

---

## Task 8: ValueDisplay widget (dynamic text + atlas-cache discipline)

**Files:**
- Create: `src/manifoldx/gui/value_display.py`
- Modify: `src/manifoldx/gui/__init__.py`
- Modify: `src/manifoldx/gui/painter.py`
- Modify: `tests/gui/test_widgets.py`
- Modify: `tests/gui/test_painter.py`
- Modify: `tests/gui/test_render_gpu.py`

### Step 8.1: Write failing tests for `ValueDisplay`

- [ ] **Append to `tests/gui/test_widgets.py`:**

```python
def test_value_display_calls_getter_each_frame_invocation():
    from manifoldx.gui import ValueDisplay
    calls = []
    def getter():
        calls.append(1)
        return "x"
    vd = ValueDisplay(getter=getter)
    vd.refresh()
    vd.refresh()
    assert len(calls) == 2
    assert vd.text == "x"


def test_value_display_min_width_locks_intrinsic_width():
    from manifoldx.gui import ValueDisplay
    vd = ValueDisplay(getter=lambda: "hello", min_width=120)
    vd.refresh()
    w, _ = vd.intrinsic_size()
    assert w >= 120


def test_value_display_intrinsic_size_does_not_change_when_string_same_length_fits_min_width():
    from manifoldx.gui import ValueDisplay
    counter = {"n": 0.0}
    def getter():
        counter["n"] += 1.0
        return f"fps: {counter['n']:.0f}"  # 1- to 3-digit fluctuation
    vd = ValueDisplay(getter=getter, min_width=200)
    vd.refresh(); size1 = vd.intrinsic_size()
    vd.refresh(); size2 = vd.intrinsic_size()
    # min_width clamps the width so it doesn't oscillate.
    assert size1 == size2
```

- [ ] **Run:** `uv run pytest tests/gui/test_widgets.py -v -k value_display`
- [ ] **Expected:** FAIL — no `ValueDisplay` export.

### Step 8.2: Implement `ValueDisplay`

- [ ] **Create `src/manifoldx/gui/value_display.py`:**

```python
"""Dynamic text widget — renders a getter()'s return value each frame.

Caching discipline:
- `refresh()` is called once per frame by the gui render pass.
- If the new string equals the cached one, intrinsic_size is reused (no atlas churn).
- If the new string differs, intrinsic_size is recomputed and the widget marks
  itself layout-dirty (Plan 2 will hook layout-dirty propagation into the bridge;
  Plan 1 ignores it since layout currently recomputes every frame).

For values that fluctuate frame-to-frame, callers SHOULD pass `min_width` so
the intrinsic width doesn't oscillate.
"""

from __future__ import annotations

from typing import Any, Callable

from manifoldx.gui.widgets import Widget, _measure_text


class ValueDisplay(Widget):
    def __init__(
        self,
        getter: Callable[[], str],
        *,
        min_width: float | None = None,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        self._getter = getter
        self._min_width = min_width
        self._text: str = ""
        self._cached_intrinsic: tuple[float, float] | None = None

    @property
    def text(self) -> str:
        return self._text

    def refresh(self) -> None:
        """Call the getter, update cached text/intrinsic if changed."""
        new = self._getter()
        if not isinstance(new, str):
            new = str(new)
        if new != self._text:
            self._text = new
            self._cached_intrinsic = None

    def intrinsic_size(self) -> tuple[float, float]:
        if self._cached_intrinsic is None:
            font_size = int(self.effective_style().get("font_size", 12))
            w, h = _measure_text(self._text, font_size)
            if self._min_width is not None:
                w = max(w, float(self._min_width))
            self._cached_intrinsic = (w, h)
        return self._cached_intrinsic
```

### Step 8.3: Wire `ValueDisplay` into the painter

- [ ] **Edit `src/manifoldx/gui/painter.py`** — at the top, replace the imports block with:

```python
from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.style import parse_color
from manifoldx.gui.value_display import ValueDisplay
from manifoldx.gui.widgets import Panel, Text, Widget
```

Then, in `paint`, add a branch for `ValueDisplay` after the `Text` branch:

```python
    elif isinstance(widget, ValueDisplay):
        s = widget.effective_style()
        painter.draw_text(
            box=box,
            text=widget.text,
            font_size=int(s["font_size"]),
            fg=parse_color(s["fg"]),
        )
```

### Step 8.4: Export `ValueDisplay` from the gui package

- [ ] **Edit `src/manifoldx/gui/__init__.py`:**

```python
"""ManifoldX in-engine GUI layer.

Public API (Plan 1):
- `Panel`, `Text`, `ValueDisplay` — non-interactive widgets.
- `style` — theme + named classes + per-widget overrides.

Plan 2 will add `Button`, `Slider`, `Toggle`.
"""

from manifoldx.gui import style  # noqa: F401
from manifoldx.gui.value_display import ValueDisplay  # noqa: F401
from manifoldx.gui.widgets import Panel, Text  # noqa: F401

__all__ = ["Panel", "Text", "ValueDisplay", "style"]
```

### Step 8.5: Have the gui pass call `refresh()` on every ValueDisplay each frame

- [ ] **Edit `src/manifoldx/render/passes/gui.py`** — at the top of `render_gui_pass`, before `painter = Painter()`:

```python
    # Drive dynamic text widgets once per frame.
    for panel in engine.gui:
        _refresh_value_displays(panel)
```

Add a helper near the other module-level helpers:

```python
def _refresh_value_displays(widget) -> None:
    from manifoldx.gui.value_display import ValueDisplay
    if isinstance(widget, ValueDisplay):
        widget.refresh()
    for child in getattr(widget, "children", []) or []:
        _refresh_value_displays(child)
```

### Step 8.6: Add a painter test that ValueDisplay emits a text op

- [ ] **Append to `tests/gui/test_painter.py`:**

```python
def test_paint_value_display_emits_text_op():
    from manifoldx.gui import ValueDisplay
    vd = ValueDisplay(getter=lambda: "fps: 60")
    vd.refresh()
    panel = Panel(children=[vd])
    spec = panel.build_layout_spec()
    boxes = compute_layout(spec, viewport=LayoutBox(0, 0, 100, 50))
    p = Painter()
    paint(panel, spec, boxes, p)
    assert any(op.text == "fps: 60" for op in p.text_ops)
```

### Step 8.7: Add a GPU test that ValueDisplay renders + getter is called every frame

- [ ] **Append to `tests/gui/test_render_gpu.py`:**

```python
def test_gui_pass_drives_value_display_each_frame():
    from manifoldx.gui import ValueDisplay
    style.set_theme({"bg": "#000000ff", "fg": "#ffffffff", "font_size": 14,
                     "padding": 2, "gap": 0})
    engine, canvas = _make_offscreen_engine(width=160, height=64)
    calls = []
    def getter():
        calls.append(1)
        return f"n={len(calls)}"
    vd = ValueDisplay(getter=getter, min_width=80)
    panel = Panel(
        children=[vd],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 100, "height": 40},
    )
    engine.gui.append(panel)
    engine._draw_frame()
    engine._draw_frame()
    engine._draw_frame()
    assert len(calls) == 3
```

### Step 8.8: Run all gui tests

- [ ] **Run:** `uv run pytest tests/gui/ -v`
- [ ] **Expected:** all tests in `tests/gui/` PASS.

### Step 8.9: Full suite + lint

- [ ] **Run:** `make test`
- [ ] **Expected:** all tests PASS (existing + new).
- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.

### Step 8.10: Commit

- [ ] **Commit:**

```bash
git add src/manifoldx/gui/__init__.py src/manifoldx/gui/value_display.py \
        src/manifoldx/gui/painter.py src/manifoldx/render/passes/gui.py \
        tests/gui/test_widgets.py tests/gui/test_painter.py \
        tests/gui/test_render_gpu.py
git commit -m "feat(gui): ValueDisplay widget with per-frame getter + atlas-cache discipline"
```

---

## Task 9: Public API freeze + CHANGELOG entry + Plan-1 closeout

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`

### Step 9.1: Add a `[Unreleased]` line for Plan 1

The full `[Unreleased]` GUI v1 entry lands when Plan 2 ships (one bullet per the whole sub-project). Plan 1 just leaves a footprint that the in-progress work is visible.

- [ ] **Edit `CHANGELOG.md`** — under `## [Unreleased]` add (creating the header if missing):

```markdown
## [Unreleased]

### Features

- **GUI v1 — foundation (in progress)** — new `manifoldx.gui` package with `Panel`, `Text`, `ValueDisplay`, stack + flex layout, theme + named-class styling, and a gui render pass driven by `RectMaterial` (signed-distance rounded rect) plus a glyph path that reuses `LabelTextureAtlas`. No interactivity yet — Plan 2 will add `Button`/`Slider`/`Toggle` + the `_GuiBridge`.
```

### Step 9.2: Update AGENTS.md "Sub-projects in flight" to reflect Plan 1 landed

- [ ] **Edit `AGENTS.md`** — in the `## Sub-projects in flight` section, add a new bullet (chronological — append after the existing ones):

```markdown
- **GUI v1 — Plan 1 (foundation)** — landed. `manifoldx.gui` with non-interactive widgets (`Panel`/`Text`/`ValueDisplay`), stack + flex layout, theme + named-class styling, `RectMaterial` (WGSL signed-distance rounded rect), and a new `gui` render pass at the end of the render order. Plan 2 (interaction: `_GuiBridge` + `Button`/`Slider`/`Toggle` + demo) not yet started. Plans at `.knowledge/plans/2026-05-11-gui-v1-plan-*.md`. Design at `.knowledge/analysis/2026-05-10-gui-v1-design.md`.
```

### Step 9.3: Run the full suite once more, all green

- [ ] **Run:** `make test`
- [ ] **Expected:** all tests PASS.
- [ ] **Run:** `make lint`
- [ ] **Expected:** clean.

### Step 9.4: Final commit

- [ ] **Commit:**

```bash
git add CHANGELOG.md AGENTS.md
git commit -m "docs(gui): CHANGELOG + AGENTS.md note for GUI v1 Plan 1 landing"
```

---

## Self-review checklist (run after the plan is written, before handoff)

- [ ] **Spec coverage.** Map design-doc sections to tasks:
  - Module layout → Task 1–7 (every file in the design appears at least once)
  - Render path order → Task 6 (registered after label pass)
  - Engine integration → Task 6 (`engine.gui = _GuiRoot()`)
  - Widgets — Panel/Text → Task 3; ValueDisplay → Task 8; Button/Slider/Toggle → **deferred to Plan 2** (explicitly noted).
  - Style vocabulary → Task 1
  - Layout (stack + flex) → Task 2
  - Material details (RectMaterial WGSL + per-instance struct) → Task 5
  - Testing targets (CPU layout, style, hit-test, widget events; GPU smoke; e2e) → CPU coverage in Tasks 1–5, 8; GPU smoke in Tasks 6–8. **Hit-test / widget-events / e2e tests are deferred to Plan 2** because the bridge is the unit under test there.
  - Examples (gui_demo.py) → **deferred to Plan 2** (no interactive widgets to demo yet in Plan 1).

- [ ] **Placeholders.** No "TODO", "TBD", or "similar to Task N". Every step has its own code block.

- [ ] **Type consistency.** Method/attribute names checked:
  - `Widget.effective_style()` → `(dict)` used in painter + widget builder
  - `Panel.build_layout_spec()` → `dict[str, Any]` consumed by `compute_layout` and `paint`
  - `Painter.draw_rect(box=, fill=, border_color=, border=, radius=)` matches `RectOp` dataclass fields
  - `RectMaterial.pack_instances(rect_ops) -> ndarray (N, 14)` matches WGSL `RectInstance` struct column order
  - `_GuiRoot.append / __iter__ / __len__ / __getitem__ / pointer_over_gui` matches the design's `engine.gui.append(panel)` + `engine.gui.pointer_over_gui` API
  - `ValueDisplay.refresh()` called by the gui pass each frame; `intrinsic_size()` honors `min_width`

- [ ] **Plan 2 prerequisites listed.** Anything Plan 1 leaves dangling that Plan 2 must wire up:
  - `_GuiRoot.pointer_over_gui` flag exists but no code toggles it (bridge owns that).
  - Per-frame `refresh()` call shape is in place; the bridge can hook off it without changes.
  - Pipeline cache key 5th element `"gui"` is conceptual in Plan 1 (we own a local pipeline outright). Plan 2 may revisit if Button/Slider/Toggle share rect pipelines.

---

## Execution Handoff

**Plan complete and saved to `repos/manifoldx/.knowledge/plans/2026-05-11-gui-v1-plan-1-foundation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because the GPU tests (Tasks 6–8) benefit from a clean context per task.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints. Lower latency, but risks context drift across the 9-task arc.

**Which approach?**
