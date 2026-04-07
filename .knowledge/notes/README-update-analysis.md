# README Update Analysis

## Current State

The README needs updates to reflect:
1. New run()/render() API (no Backend enum)
2. Video rendering capability
3. Optional dependencies (desktop/offline)
4. Updated example usage

## New Features to Document

### 1. Simplified API

**Before (old):**
```python
engine = mx.Engine("MyApp", backend=mx.Backend.DESKTOP)
engine.run()
```

**After (new):**
```python
engine = mx.Engine("MyApp")
engine.run()  # Auto-detects GLFW or Pyodide
```

### 2. Video Rendering

New `render()` method for headless video output:
```python
# Render to video file
engine.render(
    output="simulation.mp4",
    fps=30,
    duration=60,  # 60 seconds
    quality="high"  # low, medium, high
)
```

Examples now support `--render` flag:
```bash
python examples/nbody.py --render 60  # 60 second video
python examples/boids.py --render     # default 60 seconds
```

### 3. Optional Dependencies

```bash
# Desktop rendering (GLFW window)
pip install manifold-gfx[desktop]

# Video rendering (imageio-ffmpeg)
pip install manifold-gfx[offline]

# All backends
pip install manifold-gfx[all]
```

### 4. Backend Behavior

| Method | Backend | Environment |
|--------|---------|-------------|
| `run()` | GLFW | Desktop (Linux/macOS/Windows) |
| `run()` | Pyodide | Browser (emscripten) |
| `render()` | Offscreen | Headless / CI |

## Recommended README Changes

### Installation Section
Add optional dependencies:
```markdown
## Installation

```bash
pip install manifold-gfx           # Base (no rendering)
pip install manifold-gfx[desktop] # + GLFW for desktop window
pip install manifold-gfx[offline]  # + imageio-ffmpeg for video
pip install manifold-gfx[all]      # Everything
```
```

### Quick Start Section
Update to show simplified API:
```markdown
# Run with desktop window
engine.run()

# Render to video (headless)
engine.render(output="movie.mp4", duration=60)
```

### Examples Section
Add video rendering note:
```markdown
# Run example normally
python examples/nbody.py

# Render to video (requires offline extra)
python examples/nbody.py --render 60
```

## Related Commits

- `9737bf4` feat(examples): add --render flag for video output support
- `b72ccd6` refactor: extract _init_canvas and _draw_frame for code reuse
- `5f7a943` docs: mark simplification plan as completed
- `c8af2fe` feat: Phase 4 - update tests, remove Backend references
- `00a8161` feat: Phase 3 - render() uses offscreen unconditionally
- `e524bd3` feat: Phase 2 - run() is context-aware (GLFW desktop)
