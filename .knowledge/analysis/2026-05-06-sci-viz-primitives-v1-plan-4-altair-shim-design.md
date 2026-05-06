# Sci-viz primitives v1 — Plan 4: Altair-style functional shim

**Status:** Design — approved through brainstorming on 2026-05-06.

**Predecessors:** Plans 1-3 landed the ECS primitives (PointCloud, ColormapMaterial, sprite path; TextLabel, LabelMaterial, label atlas; AxisFrame, AxisMaterial with world & screen anchoring; colormap legend via atlas slice).

**Goal:** A declarative, Altair-shaped API that wraps the imperative ECS so the 80% case ("show me a 3D scatter plot of these points colored by speed with axes and a legend") fits in 10 lines instead of 100.

## Why Altair, not matplotlib

Alex's brief: grammar of graphics. Three properties of Altair that matter here:

- **Declarative, not imperative.** Describe the chart shape; the library figures out how to render. No `engine.spawn(...)` / `engine.system(...)` / component pre-registration.
- **Composable via operators.** `mark + mark + mark` builds a layered scene without method-chain ceremony.
- **Encodings carry their own config.** Channel objects (`mxv.color(array, cmap="viridis", domain=(0, 10))`) co-locate the data and the visual mapping for that channel — instead of separate axes/legend/colormap setup calls.

Matplotlib's `plt.plot(...)` / `plt.xlabel(...)` / `plt.colorbar(...)` is imperative state mutation on a global figure. The pyplot API doesn't survive the jump from static 2D to live 3D — every imperative call would need an inverse to undo, and the order of calls would matter.

## The tension: Altair on top of a real-time engine

Altair's data is static — a DataFrame snapshot. Manifoldx's data mutates every frame via simulation callbacks. The shim has to bridge the two without inheriting Altair's static assumptions.

**Resolution (decided in brainstorming):** the chart holds *references* to the user's live numpy arrays. Per-frame mutation happens in `@chart.simulate(dt)` callbacks the user writes; the renderer re-reads the arrays each frame. Chart spec is declarative; data flow is mutable.

## API shape

### Marks (factory functions in `manifoldx.viz.shims`)

| Mark | Purpose | Underlying primitive |
|---|---|---|
| `mxv.points(positions, color=..., size=...)` | 3D point cloud (sprite billboards) | PointCloud + ColormapMaterial + ScalarValue + Radius |
| `mxv.mesh(geometry, transform=..., material=...)` | Single 3D mesh (PBR or unlit) | Mesh + Transform + Material |
| `mxv.axes(extent=..., labels=...)` | Three colored axes + tick labels | AxisFrame + AxisMaterial (world) + TextLabel + LabelMaterial |
| `mxv.legend(color_channel, title=..., position=...)` | Colormap legend | LabelMaterial(screen) + atlas LUT slice + TextLabel ticks |
| `mxv.scale_bar(length=..., label=..., position=...)` | Screen-anchored scale bar | AxisMaterial(screen) + LabelMaterial(screen) |
| `mxv.lights(specs)` | Light setup (declarative) | engine.set_lights(...) |

Tier 3 marks (text, lines as a generic primitive) deferred until real demand.

### Encodings — bare arrays + Channel-object wrappers

The 80% case takes raw numpy arrays directly:

```python
chart = mxv.points(
    positions=positions_array,    # (N, 3) float32
    color=speeds_array,            # (N,) float32 — auto vmin/vmax from data
    size=radii_array,              # (N,) float32 — world-space radius
)
```

Richer config wraps the array in a Channel object:

```python
chart = mxv.points(
    positions=positions_array,
    color=mxv.color(speeds_array, cmap="viridis", domain=(0, 10), title="Speed (m/s)"),
    size=mxv.size(radii_array),
)
```

Channel object types planned: `mxv.color(...)`, `mxv.size(...)`, `mxv.position(...)`. Each accepts the array as first positional arg and channel-specific kwargs. The chart factory normalizes bare arrays into bare-default Channel objects internally.

A Channel object referenced once by `mxv.points` and again by `mxv.legend` shares the same underlying array — that's how `legend(chart.color)` auto-syncs domain and colormap with the points it annotates.

### Composition — single-scene layering with `+`

```python
chart = (
    mxv.points(positions=p, color=mxv.color(s, cmap="viridis"), size=r)
    + mxv.mesh(sphere(0.5), pos=(0, 0, 0))
    + mxv.axes(extent=10)
    + mxv.legend("Speed")
    + mxv.scale_bar(length=2.0, label="2 m")
    + mxv.lights([mxv.PointLight(intensity=10)])
)
```

`+` returns a `Chart` carrying an ordered list of marks. No multi-viewport (`|`, `&`) in v1 — that's a Plan 5+ feature requiring renderer changes.

### Live data — `@chart.simulate(dt)`

```python
@chart.simulate
def physics(dt):
    positions[:] += velocities * dt
    speeds[:] = np.linalg.norm(velocities, axis=1)
```

The decorator stashes the callback; the chart compiles it to an `@engine.system` internally. The user mutates the same numpy arrays the chart holds references to — no copy, no synchronization step needed.

Multiple `@chart.simulate` callbacks can be registered; they run in registration order. (Open: how to inject `t` / `dt` and any `query`-like accessor — start with just `dt`, add more if needed.)

### Terminal — `cli()`, `run()`, escape hatch via `.engine`

```python
if __name__ == "__main__":
    chart.cli()   # parses --render / --duration / --fps / --output / --quality, dispatches
```

Programmatic:

```python
chart.run()                                   # interactive window, blocks until close
chart.render("scene.mp4", duration=10)        # offline render to MP4
```

Escape hatch:

```python
engine = chart.engine    # materializes if not yet built; returns the underlying Engine
# now imperative ECS ops are available — useful for one-off systems that don't fit the chart shape
@engine.system
def custom(query, dt):
    ...
```

## Architecture

### Module layout

```
src/manifoldx/viz/shims.py     # marks, Channel objects, Chart, build/run logic
src/manifoldx/viz/__init__.py  # re-exports the public API (mxv namespace)
```

Single new module. The shim layer is purely additive — every existing entry point under `manifoldx.viz` keeps working.

### Class hierarchy

```
Chart
  ├─ marks: list[Mark]
  ├─ simulate_callbacks: list[Callable]
  ├─ _engine: Engine | None  (lazily built)
  ├─ __add__(other) -> Chart
  ├─ build() -> Engine        (materializes marks into ECS spawns)
  ├─ run() / render() / cli() (terminal entry points)
  └─ simulate(func)           (decorator)

Mark (abstract)
  └─ apply(engine: Engine) -> None    (each mark spawns its entities)

Mark subclasses: PointsMark, MeshMark, AxesMark, LegendMark, ScaleBarMark, LightsMark

Channel (abstract)
  └─ resolve(n: int) -> np.ndarray    (per-frame data; bare arrays auto-wrap)

Channel subclasses: ColorChannel, SizeChannel, PositionChannel
```

### Build pipeline

1. User assembles a Chart via `+`.
2. First call to `chart.run()` / `chart.render()` / `chart.cli()` / `chart.engine` triggers `chart.build()`.
3. `build()` constructs an `Engine` (default canvas size, max_entities tuned to the largest mark) and walks marks in order, calling `mark.apply(engine)` to spawn entities and register live-data systems.
4. Each `@chart.simulate` callback compiles to an `@engine.system` that depends on `Transform` (so it ticks every frame).
5. `run()` / `render()` delegate to `engine.cli()` / `engine.render()` from there.

### Live-data update path

A Channel that wraps a mutable array installs a per-frame system that copies the array into the entities' component data:

```python
@engine.system
def _channel_color_sync(query, dt):
    query[ScalarValue].value = speeds_array.reshape(-1, 1)
```

For the position channel, the sync writes into Transform.pos. For size, into Radius. The user's `@chart.simulate` mutates the source array; the channel sync mirrors it into ECS each frame. Order: simulate callbacks run first, then channel syncs, then the renderer.

(Optimization deferred: when the channel array is the same object as the underlying ECS storage, skip the copy. Out of scope for v1 — start with the copy and profile.)

## Examples

### Minimal n-body scatter

```python
import manifoldx.viz as mxv
import numpy as np

N = 500
positions = np.random.uniform(-5, 5, (N, 3)).astype(np.float32)
velocities = np.random.normal(0, 0.5, (N, 3)).astype(np.float32)
speeds = np.linalg.norm(velocities, axis=1)
radii = np.full(N, 0.05, dtype=np.float32)

chart = (
    mxv.points(
        positions=positions,
        color=mxv.color(speeds, cmap="viridis", domain=(0, 2)),
        size=radii,
    )
    + mxv.axes(extent=6)
    + mxv.legend("Speed")
)


@chart.simulate
def step(dt):
    # Trivial example — replace with real physics.
    positions[:] += velocities * dt
    speeds[:] = np.linalg.norm(velocities, axis=1)


if __name__ == "__main__":
    chart.cli()
```

### Reusing a Channel across marks

```python
color = mxv.color(speeds, cmap="inferno", domain=(0, 5))
chart = (
    mxv.points(positions=positions, color=color, size=radii)
    + mxv.legend(color, title="Speed")  # legend reads the same Channel
)
```

### Escape to imperative ECS for one-off systems

```python
chart = mxv.points(...)
engine = chart.engine

@engine.system
def diagnostic(query, dt):
    if engine.elapsed > 5.0 and engine.elapsed < 5.05:
        print("snapshot at t=5:", positions[:3])

chart.run()
```

## Out of scope for v1

- **Multi-viewport composition** (`|`, `&`). Requires renderer rewrite to multi-camera. Plan 5+.
- **Generic `lines` mark.** Would require generalizing AxisMaterial beyond axis lines. Defer until concrete demand.
- **Standalone `text` mark.** Today's TextLabel + LabelMaterial covers it imperatively; users who need pure-text overlays can use the escape hatch. Reconsider in Plan 5.
- **Reactive transforms** (Altair's `transform_calculate`, `transform_filter`). The `@chart.simulate` callback covers all of this with a tiny fraction of the API surface.
- **Auto-scale derivation from data.** Domain inference (`vmin = data.min(), vmax = data.max()`) at build time only — not per-frame; user can pass an explicit domain or accept frozen build-time scale. Per-frame auto-scale would require re-uploading the LUT-relevant uniforms each frame; defer.
- **Theme / styling.** No `mxv.theme(...)` system. Channel-level kwargs (color hex, font size) cover what's needed for v1.

## Testing strategy

- **Unit:** Channel object semantics (bare-array auto-wrap, idempotent on bare arrays, kwarg defaults).
- **Unit:** Each Mark's `apply(engine)` spawns the right component shape — assertions on `engine.store._components` after build.
- **Unit:** `Chart.__add__` builds an ordered mark list; `chart.simulate` registers exactly once.
- **Integration:** End-to-end `chart.render("/tmp/out.mp4", duration=1)` on each example — confirms the full pipeline doesn't crash and produces non-empty output.
- **Integration:** Live-data path — write a chart that mutates `positions` in `@chart.simulate`; render two frames offline; assert the second frame's pixels differ from the first.

## File change inventory

### New files
- `src/manifoldx/viz/shims.py` (~600 lines: Chart, Mark base + 6 subclasses, Channel base + 3 subclasses, simulate hook)
- `tests/viz/test_shims.py` (~250 lines: unit tests per the testing strategy)
- `tests/viz/test_shims_integration.py` (~200 lines: end-to-end render test per mark + live-data test)
- `examples/scatter_plot.py` (~50 lines: the minimal-n-body example, swap for boids/orbital later)

### Modified files
- `src/manifoldx/viz/__init__.py` — re-export the new public surface (`points`, `mesh`, `axes`, `legend`, `scale_bar`, `lights`, `color`, `size`, `position`, `Chart`, `PointLight`).

### Untouched
- All existing imperative ECS primitives (`PointCloud`, `LabelMaterial`, etc.) keep working unchanged. The shim is purely additive.

## Self-review

- **Placeholder scan:** none.
- **Internal consistency:** Channel objects, Mark.apply, and the live-data sync system mention the same component pipeline (Transform / ScalarValue / Radius). The four terminal entry points (`cli`, `run`, `render`, `.engine`) all delegate to one `build()` call — no duplication.
- **Scope check:** 6 marks + 3 channel kinds + 1 chart class + 4 terminal entry points + 1 simulate decorator. ~600 lines of new code is realistic for a 10-15 task plan.
- **Ambiguity check:** the per-frame channel-sync system runs *after* simulate callbacks (so user's mutations are visible to the GPU same frame). Documented above — locked.
