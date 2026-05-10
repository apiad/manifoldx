# GUI Layer for ManifoldX — Design

**Date:** 2026-05-10
**Status:** Approved, ready for implementation planning.
**Scope:** A retained-mode in-engine GUI for sim controls + HUD readouts. Six widget classes, stack + flex layout, style dicts with named classes, callbacks via the existing event bus from `2026-05-08-event-driven-system-design.md`. No general application-UI ambitions — this is a researcher's control panel, not a windowing toolkit.

## Goal

Let researchers wire up a docked control panel for any example sim with ~10 lines of code:

```python
panel = gui.Panel(
    anchor="top-right",
    style="hud_panel",
    children=[
        gui.Text("N-Body"),
        gui.Slider(name="G",  min=0.1, max=10.0, value=1.0),
        gui.Toggle(name="trails", value=False, label="Trails"),
        gui.Button(name="reset", label="Reset"),
        gui.ValueDisplay(getter=lambda: f"fps: {engine.fps:.1f}"),
    ],
)
engine.gui.append(panel)

@engine.on("gui:slider:G:change")
def _(payload):
    sim.G = payload["value"]
```

## Non-goals (v1)

- **Text input fields**, dropdowns, scrollable lists, tree views.
- **Keyboard focus + tab order.** No widget consumes keyboard events in v1.
- **Tooltips, modals, popovers, context menus.**
- **Animation / transitions / pseudo-states** (`:hover`, `:active`, `:focus`). Buttons get a hardcoded slight darken on hover; nothing is configurable through style.
- **Real CSS-string parsing.** Style is Python dicts and named classes; no selector matching, no specificity rules.
- **Margin collapse, percentage units, calc(), or grid layout.** Stack + flex only.
- **Theming presets / multi-theme switching.** A single user-defined theme.
- **Z-order beyond append order.**
- **GUI as ECS entities.** Widgets are plain Python objects living in `engine.gui`; sim systems doing `query[Transform]` will not see them.
- **Per-widget `margin`.** Margin between siblings comes from the container's `gap`. Per-widget margin is a YAGNI complication.

## Architecture

### Module layout

```
src/manifoldx/gui/
├── __init__.py     # public API: Panel, Button, Slider, Toggle, Text, ValueDisplay, style
├── widgets.py      # widget classes + the abstract Widget base
├── layout.py       # stack + flex algorithm
├── style.py        # set_theme, define, resolve
├── painter.py      # Painter accumulator (rect + text ops)
├── bridge.py       # _GuiBridge (event-bus subscriber, hit-tester, dispatcher)
└── material.py     # RectMaterial — WGSL signed-distance rounded rect + 1px border

src/manifoldx/render/passes/
└── gui.py          # the new gui render pass — runs last

src/manifoldx/engine.py  # adds: self.gui = [], self._gui_bridge = _GuiBridge(self)
```

### Render path

A new pass at the end of the render order: `mesh → sprite → volume → label → gui`. Pass attributes:

- Depth-test off, depth-write off.
- Alpha blend on (pre-multiplied alpha).
- Issues exactly two batched draws per frame:
  1. **Rects** — one instanced draw via `RectMaterial` (one rounded-rect SDF per instance, with optional 1 px border, RGBA fill, RGBA border, radius, size).
  2. **Glyphs** — one instanced draw reusing the existing `LabelTextureAtlas` and the screen-anchored label pipeline. Pipeline cache key gets a `"gui"` 5th element so it doesn't collide with `label` pipelines.

### Engine integration

`Engine.__init__` adds:
- `self.gui: list[Panel] = []` — list of root panels.
- `self._gui_bridge = _GuiBridge(self)` — subscribes to `pointer_down`/`_move`/`_up`/`wheel` on `self._event_bus` in `_init_canvas` (parallel to `_input_bridge.attach(canvas)`).
- `engine.gui.pointer_over_gui: bool` — read by user systems that want to ignore game-world clicks when the pointer is over a panel. Set true at the start of any frame whose pointer coords hit a widget; reset at the start of the next frame's input-bridge `begin_frame()`.

The render pipeline registers the new `gui` pass after `label` in `RenderPipeline.run`.

### Data flow

```
event bus (from input layer)
        │
        ▼
_GuiBridge._on_pointer_event(evt)
        ├── walk engine.gui top-down
        ├── hit_test(evt.x, evt.y) → topmost widget or None
        ├── if hit: engine.gui.pointer_over_gui = True
        ├── update widget state (slider value, toggle bool, drag capture)
        └── engine.emit("gui:<kind>:<name>:<event>", payload)

frame loop (step N: render)
        │
        ▼
gui_pass(renderer)
        ├── for panel in engine.gui:
        │     compute_layout_if_dirty(panel)
        │     paint(panel, painter)            ← walks tree, calls painter.draw_rect/draw_text
        ├── upload rect instances buffer
        ├── upload glyph instances buffer (via LabelTextureAtlas)
        └── two instanced draw calls
```

## Public API

### Widgets (six classes)

| Widget          | Constructor                                                                | Notes                                                            |
| --------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `Panel`         | `Panel(children, anchor="top-left", offset=(0,0), style=…)` | Container. `anchor` + `offset` apply only to **root** panels in `engine.gui`; nested panels are positioned by the parent's layout pass and ignore them. `direction` (vertical/horizontal) is a style key, not a constructor kwarg. |
| `Text`          | `Text(text, style=…)`                                                      | Static label. Width derived from rasterized glyph extents unless `width` style is set. |
| `ValueDisplay`  | `ValueDisplay(getter, style=…, min_width=…)`                               | `getter()` runs once per frame; result rendered as text. Cache: re-rasterize through the atlas only when the string changes. |
| `Button`        | `Button(name, label, style=…)`                                             | Click → `gui:button:<name>:click`.                               |
| `Slider`        | `Slider(name, min, max, value, label=None, style=…)`                       | Horizontal drag. Live `slider.value`. Emits `:change` continuously and `:commit` on pointer-up. |
| `Toggle`        | `Toggle(name, value, label, style=…)`                                      | Click flips. Live `toggle.value`. Emits `:change`.               |

All widget kwargs are keyword-only after the first positional, matching the rest of the engine's API.

### Events

Every interactive widget emits via `engine.emit(...)`. `name` is **required** for `Button`, `Slider`, and `Toggle` — non-optional positional arg, validated at `__init__`. Non-interactive widgets (`Text`, `ValueDisplay`, `Panel`) don't have a `name` and don't emit.

| Source        | Event name                       | Payload              |
| ------------- | -------------------------------- | -------------------- |
| `Button`      | `gui:button:<name>:click`        | `{}`                 |
| `Slider`      | `gui:slider:<name>:change`       | `{"value": float}`   |
| `Slider`      | `gui:slider:<name>:commit`       | `{"value": float}`   |
| `Toggle`      | `gui:toggle:<name>:change`       | `{"value": bool}`    |

Pull access is always available: `slider.value` (live float), `toggle.value` (live bool), `button.was_clicked()` (returns `True` once if the button has been clicked since the previous `was_clicked()` call; consumed on read).

### Style

```python
gui.style.set_theme({
    "bg": "#222", "fg": "#ddd", "font_size": 12,
    "padding": 4, "gap": 4, "border": 0, "border_color": "#444",
    "radius": 0, "width": None, "height": None, "flex": None, "direction": "v",
})

gui.style.define("hud_panel", {
    "bg": "#1a1a1aD0", "padding": 8, "gap": 6, "border": 1, "radius": 4,
})

gui.style.define("danger_btn", {
    "bg": "#aa2222", "fg": "#fff", "radius": 3, "padding": "6 12",
})

gui.Button(name="reset", label="Reset",
           style="danger_btn",
           style_overrides={"radius": 8})
```

Resolution at paint time:

```
effective_style = merge(theme, named_class_style, per_widget_style_overrides)
```

Style vocabulary (fixed set, anything else is ignored):

| Key             | Value                                            | Applies to              |
| --------------- | ------------------------------------------------ | ----------------------- |
| `bg`            | hex color (`#rgb`, `#rrggbb`, `#rrggbbaa`)        | all                     |
| `fg`            | hex color                                        | text-bearing widgets    |
| `border`        | int (px width; 0 disables)                       | all                     |
| `border_color`  | hex color                                        | all                     |
| `radius`        | int (px corner radius)                           | all                     |
| `padding`       | int OR `"top right bottom left"` string           | containers              |
| `gap`           | int (px between children)                         | containers              |
| `font_size`     | int (px)                                         | text-bearing widgets    |
| `width`         | int (px) OR `None` (intrinsic)                    | all                     |
| `height`        | int (px) OR `None` (intrinsic)                    | all                     |
| `flex`          | int (share of free axis) OR `None`                | container children      |
| `direction`     | `"v"` or `"h"`                                   | containers              |

### Layout (stack + flex)

Each panel computes child boxes top-down when its tree changes or the viewport resizes. Algorithm per container:

1. Subtract padding from container box → children area.
2. Sum fixed sizes of children (those with explicit `width` or `height` along `direction`); compute remaining axis space.
3. Distribute remaining space among `flex=N` children proportionally to `N`.
4. Cross-axis: each child fills container's cross-axis size minus its own padding. (No per-child cross-axis alignment in v1.)
5. Place children sequentially with `gap` between them.

Layout is **not** recomputed every frame. Triggers:
- Tree mutation (add/remove child).
- Viewport resize (the input layer already emits `resize`; the GUI bridge listens and marks all root panels dirty).
- A widget's intrinsic size changed (currently only `Text` and `ValueDisplay` when their string changes — but they declare an intrinsic size based on rasterized glyph extents, so re-layout fires only when the new string's extents differ).

`ValueDisplay` users SHOULD pass `min_width` (or set `width` in style) for any value that fluctuates frame-to-frame — otherwise every numeric jitter triggers a layout pass. Documented in the docstring.

## Input flow

`_GuiBridge` subscribes to `pointer_down`, `pointer_move`, and `pointer_up` on the event bus. It does **not** subscribe to keyboard or wheel events (no v1 widget consumes them; sliders are drag-only, no scroll-to-fine-tune).

### Hit testing

Widget tree is walked top-down (last-appended panel checked first, matching draw order). For each panel, the bridge tests the panel's own bounding box; if hit, recurses into children. Returns the topmost interactive widget (`Button`, `Slider`, `Toggle`); `Text`, `ValueDisplay`, `Panel` are non-interactive and don't consume hits.

### Drag capture

`Slider` is the only widget that needs drag capture. On `pointer_down` over a slider, the bridge records `self._captured = slider`. While captured, all `pointer_move` events go to the slider regardless of where the cursor is. On `pointer_up` the bridge clears capture and emits `gui:slider:<name>:commit` if the pointer-down was on this same slider.

### Cooperative consume

The bus doesn't have stop-propagation. Instead:

- The bridge sets `engine.gui.pointer_over_gui = True` on any frame where the latest pointer position hit a widget.
- User-code systems that want to ignore game-world clicks when the pointer is over a panel check `if engine.gui.pointer_over_gui: return` early.
- The flag is reset at the start of each frame's `begin_frame()` step (we extend the existing 2.5 step the input bridge owns).

This is intentionally a weak consume — explicit and visible in user code, no magic.

## Material details

`RectMaterial` (`gui/material.py`) is a single WGSL fragment shader that, per fragment:

1. Computes a signed-distance field from the rect center using rounded-rect SDF (`length(max(abs(p)-half+r, 0)) - r`).
2. Decides fill / border / outside based on SDF sign and the `border` width uniform.
3. Returns pre-multiplied RGBA (border color, fill color, or 0 with alpha 0 outside). Smoothstep over 1 px gives cheap AA on the edge.

Per-instance uniforms (one struct per rect):
- `xy: vec2<f32>` — top-left corner in viewport pixels.
- `size: vec2<f32>` — width, height.
- `radius: f32`.
- `border: f32`.
- `bg: vec4<f32>`.
- `border_color: vec4<f32>`.

Single global uniform: `viewport_size: vec2<f32>` (already on the existing `Globals` from sci-viz Plan 3).

## Testing

Pure-CPU tests (no GPU):

- `test_layout.py` — feed a tree of widgets to `compute_layout(box)`, assert resulting child boxes for: vertical stack, horizontal stack, mixed flex children, padding, gap, nested panels.
- `test_style.py` — `merge(theme, named, overrides)` resolution; padding-string parser; color parser.
- `test_hit_test.py` — given a tree and a (`x`, `y`), assert the topmost interactive widget (or `None`).
- `test_widget_events.py` — drive widget state changes (slider drag, toggle click) by directly invoking bridge methods; assert the right `gui:*` events fire on the bus and that `pointer_over_gui` toggles correctly.

GPU smoke (gated on `get_offscreen_canvas`, like the rest of the suite):

- `test_render.py` — spawn a panel with one of each widget; render one frame to an offscreen target; assert the framebuffer is non-empty in the panel region and transparent outside.

End-to-end:

- `tests/gui/test_e2e.py` — instantiate an engine with a GUI, simulate `pointer_down` / `pointer_move` / `pointer_up` via the existing `canvas.submit_event` test affordance from the input layer; assert the user's `@engine.on("gui:slider:G:change")` handler runs with the expected payload.

Targets: ~25 new tests; the suite stays under 15 s wall-clock on the offscreen backend.

## Examples

- `examples/gui_demo.py` — N-body rerun with a docked control panel: sliders for `G`, `dt`, particle count; toggle for trails (no-op stub in v1 if trails aren't already implemented — placeholder); reset button; fps display via `ValueDisplay`.

That's the only new example; existing demos stay untouched.

## Open questions / risks

- **Glyph atlas footprint.** A `ValueDisplay` showing `f"fps: {x:.1f}"` produces a new string every time the value changes by 0.1. The atlas's 256-slice cap (Plan 2) could fill up fast on long-running sessions. Mitigation in v1: document the issue, recommend `min_width` + truncated format strings (`f"fps: {round(x):3d}"`). Proper mitigation (LRU eviction + slice reuse) is a follow-up to the atlas, not GUI-specific.
- **Rect WGSL borrow conflict.** The existing `Globals` uniform was extended for sci-viz Plan 3 to 224 bytes. RectMaterial reuses it; verify the binding layout matches the GUI pipeline's bind group.
- **`pointer_over_gui` timing.** Set during pointer-event dispatch (which happens at the head of next frame, per the event bus's one-frame-delayed model). User systems running in the same frame's `update` see the right value because both run after dispatch. Documented in the design doc; testable.

## Implementation order

1. RectMaterial + GUI render pass (just the rect path; render a fixed test rect from a unit test). Lays the rendering pipe.
2. Widget base + Text + Panel + layout + style. Static text panels render correctly.
3. ValueDisplay + atlas integration.
4. `_GuiBridge` + hit-testing + Button.
5. Slider (drag capture).
6. Toggle.
7. `examples/gui_demo.py` + CHANGELOG entry.

Each step is its own commit, each with a passing test before moving on.
