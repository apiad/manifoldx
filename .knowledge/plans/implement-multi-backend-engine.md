---
id: implement-multi-backend-engine
created: 2025-04-07
modified: 2025-04-07
type: plan
status: active
expires: 2025-04-14
phases:
  - name: Phase 1: Refactor Backend enum and Optional Dependencies
    done: false
    goal: Define Backend enum and set up optional dependency groups in pyproject.toml
  - name: Phase 2: Implement Lazy Backend Imports
    done: false
    goal: Implement lazy canvas creation with graceful import error handling
  - name: Phase 3: Implement run() with Desktop/Browser backends
    done: false
    goal: Implement real-time rendering with event loop for Desktop/Browser
  - name: Phase 4: Implement render() with OFFSCREEN backend
    done: false
    goal: Implement offline video rendering with OffscreenRenderCanvas
  - name: Phase 5: Add CI Validation and Backend Tests
    done: false
    goal: Add CI workflow with offline-only tests and backend-specific tests
---

## Optional Dependencies Structure

### Installation Variants

```bash
# Base install (no rendering backends)
pip install manifold-gfx

# Desktop rendering only (GLFW)
pip install manifold-gfx[desktop]

# Browser rendering only (Pyodide)
pip install manifold-gfx[browser]

# Video rendering only (imageio-ffmpeg)
pip install manifold-gfx[offline]

# All backends
pip install manifold-gfx[all]
```

### pyproject.toml Structure

```toml
[project]
name = "manifold-gfx"
dependencies = [
    "numpy>=2.4.4",
    "rendercanvas>=2.6.3",
    "wgpu>=0.31.0",
]

[project.optional-dependencies]
desktop = ["glfw>=3.0"]
browser = ["pyodide>=0.25"]
offline = ["imageio-ffmpeg>=0.5"]
all = ["glfw>=3.0", "pyodide>=0.25", "imageio-ffmpeg>=0.5"]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "manifold-gfx[offline]",  # CI uses offline for pixel-perfect validation
]
```

### CI Configuration

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install with offline backend
        run: uv sync --extra offline --extra dev
      - name: Run tests
        run: pytest tests/
      - name: Pixel-perfect validation
        run: pytest tests/test_render_validation.py
```

# Plan: Multi-Backend Engine Refactoring

## Context
The Engine class currently has a Backend enum that is defined but never used - it always uses GLFW regardless of configuration. The goal is to properly implement:
- `Engine.run()` for real-time rendering (Desktop/Browser backends)
- `Engine.render()` for offline video rendering (Offscreen backend)

This refactor will enable video rendering capabilities while maintaining clean separation between live and offline rendering modes.

## Phases

### Phase 1: Refactor Backend Enum and Optional Dependencies
**Goal:** Define Backend enum and set up optional dependency groups in pyproject.toml

**Deliverable:** Updated project configuration with:
- Backend enum with string values: "glfw", "pyodide", "offscreen"
- Optional dependencies in pyproject.toml for desktop/browser/offline/all
- Dev dependency group including offline for CI

**Done when:**
- [ ] Backend enum uses string values matching backend module names
- [ ] `desktop` extra includes glfw
- [ ] `browser` extra includes pyodide dependencies
- [ ] `offline` extra includes imageio-ffmpeg
- [ ] `all` extra includes all backends
- [ ] Dev group includes manifold-gfx[offline] for CI
- [ ] `uv sync` installs successfully with base dependencies

**Depends on:** None

---

### Phase 2: Implement Lazy Backend Imports
**Goal:** Implement lazy canvas creation with graceful import error handling

**Deliverable:** Backend factory module with:
- Lazy import functions for each backend
- Clear error messages when backend dependency is missing
- `_create_canvas(backend)` method that defers imports until needed

**Done when:**
- [ ] Lazy import function for GlfwRenderCanvas raises clear error if glfw not installed
- [ ] Lazy import function for PyodideRenderCanvas raises clear error if pyodide not installed
- [ ] Lazy import function for OffscreenRenderCanvas works (always available via rendercanvas)
- [ ] Error message includes: what backend is needed, what package to install, example command
- [ ] `_create_canvas()` method properly handles all three backends

**Depends on:** Phase 1

---

### Phase 3: Implement run() with Desktop/Browser backends
**Goal:** Implement real-time rendering with event loop for DESKTOP and BROWSER backends

**Deliverable:** Refactored run() method that:
- Validates backend is DESKTOP or BROWSER
- Creates appropriate canvas via lazy import
- Handles fullscreen mode via GLFW
- Runs event loop appropriately

**Done when:**
- [ ] run() raises ValueError if OFFSCREEN backend used
- [ ] run() raises ImportError with clear message if desktop dependency missing
- [ ] DESKTOP backend creates GlfwRenderCanvas with correct size/fullscreen
- [ ] BROWSER backend creates PyodideRenderCanvas (if in Pyodide environment)
- [ ] BROWSER backend raises error if not running in Pyodide
- [ ] Event loop runs and exits cleanly

**Depends on:** Phase 2

---

### Phase 4: Implement render() with OFFSCREEN Backend
**Goal:** Implement offline video rendering with OffscreenRenderCanvas

**Deliverable:** New render() method that:
- Validates backend is OFFSCREEN
- Uses OffscreenRenderCanvas for headless rendering
- Writes frames to video file using imageio-ffmpeg
- Supports duration/frame_count parameters
- Shows progress bar during rendering

**Done when:**
- [ ] render() raises ValueError if Desktop/Browser backend used
- [ ] render() raises ImportError with clear message if offline dependency missing
- [ ] OffscreenRenderCanvas created with correct size
- [ ] Video writer configured with fps, codec, quality settings
- [ ] Frames rendered and encoded correctly to MP4
- [ ] Sample video file generated successfully

**Depends on:** Phase 2

---

### Phase 5: Add CI Validation and Backend Tests
**Goal:** Add CI workflow with offline-only tests and backend-specific tests

**Deliverable:** 
- Updated CI workflow using offline-only backend
- Test suite with backend-specific test cases
- Pixel-perfect validation tests

**Done when:**
- [ ] CI installs with `uv sync --extra offline --extra dev`
- [ ] CI runs tests without display (headless)
- [ ] run() raises ValueError in CI (no display available)
- [ ] render() works correctly in CI, produces valid video
- [ ] Backend-specific unit tests pass
- [ ] Pixel-perfect validation generates consistent output

**Depends on:** Phase 3, Phase 4

---

## Success Criteria
- [ ] `pip install manifold-gfx[desktop]` installs glfw and enables run()
- [ ] `pip install manifold-gfx[offline]` installs imageio-ffmpeg and enables render()
- [ ] `pip install manifold-gfx[all]` installs all backends
- [ ] Engine.run() works with DESKTOP (GLFW) backend when desktop extra installed
- [ ] Engine.run() works with BROWSER (Pyodide) backend when browser extra installed
- [ ] Engine.render() produces valid MP4 video file when offline extra installed
- [ ] Wrong backend usage raises descriptive ValueError with installation hint
- [ ] Missing backend dependency raises ImportError with pip install command
- [ ] CI runs tests with `uv sync --extra offline --extra dev` (headless)
- [ ] All tests pass in CI

## Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Missing backend dependency not caught | High | Lazy imports with clear error messages |
| Pyodide environment detection fails | Medium | Check sys.platform == "emscripten", raise clear error |
| Video codec compatibility | Low | Use mp4v as fallback, yuv420p pixel format |
| Frame timing not deterministic | Medium | Use fixed timestep (1/fps) in render() loop |
| imageio-ffmpeg binary not found | Low | Include in dependencies, fallback to opencv |

## Lazy Import Pattern

```python
# src/manifoldx/backends.py

def get_desktop_canvas(width, height, fullscreen, title):
    """Create desktop canvas (GLFW). Requires: pip install manifold-gfx[desktop]"""
    try:
        from rendercanvas.glfw import GlfwRenderCanvas
    except ImportError:
        raise ImportError(
            "Desktop backend requires glfw. "
            "Install with: pip install manifold-gfx[desktop]"
        )
    
    canvas = GlfwRenderCanvas(size=(width, height), title=title)
    
    if fullscreen:
        import glfw
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_monitor(
            canvas._window, monitor, 0, 0, *mode.size, mode.refresh_rate
        )
    
    return canvas


def get_browser_canvas(width, height, title):
    """Create browser canvas (Pyodide). Requires: pip install manifold-gfx[browser]"""
    import sys
    if sys.platform != "emscripten":
        raise RuntimeError(
            "Browser backend requires Pyodide environment. "
            "This backend only works when running in a web browser via Pyodide."
        )
    
    try:
        from rendercanvas.pyodide import PyodideRenderCanvas
    except ImportError:
        raise ImportError(
            "Browser backend requires pyodide. "
            "Install with: pip install manifold-gfx[browser]"
        )
    
    return PyodideRenderCanvas()


def get_offscreen_canvas(width, height):
    """Create offscreen canvas. Requires: pip install manifold-gfx[offline]"""
    try:
        import imageio_ffmpeg
    except ImportError:
        raise ImportError(
            "Offline rendering requires imageio-ffmpeg. "
            "Install with: pip install manifold-gfx[offline]"
        )
    
    from rendercanvas.offscreen import OffscreenRenderCanvas
    return OffscreenRenderCanvas(size=(width, height), format="rgba-u8")
```

## Related
- Research: rendercanvas backends investigation
- Issue: Video rendering for demos/testing
- Dependencies: imageio-ffmpeg for video encoding