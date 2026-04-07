---
id: simplify-engine-backend
created: 2025-04-07
type: plan
status: completed
expires: 2025-04-14
phases:
  - name: Phase 1: Remove Backend enum and simplify constructor
    done: true
  - name: Phase 2: Make run() context-aware (Pyodide/GLFW)
    done: true
  - name: Phase 3: Make render() use offscreen unconditionally
    done: true
  - name: Phase 4: Update tests
    done: true
---

# Plan: Simplify Engine - Remove Backend Enum

## Context
Current implementation has a `Backend` enum that adds complexity:
- Enum values must match rendercanvas modules
- Validation needed in `run()` and `render()`
- User must know which backend to select

**Better approach:** Context-based detection
- `run()` → creates Pyodide if in browser, else GLFW
- `render()` → always uses offscreen canvas

No enum. No validation. The method name tells you the backend.

## Design

### New API
```python
class Engine:
    def __init__(self, title, width=800, height=600, fullscreen=False):
        """Create engine.
        
        No backend parameter - run() and render() choose based on context.
        """
        self.title = title
        self.w = width
        self.h = height
        self.fullscreen = fullscreen
    
    def run(self):
        """Run with real-time display.
        
        Creates: PyodideRenderCanvas if in browser (emscripten), else GlfwRenderCanvas
        """
        if sys.platform == "emscripten":
            canvas = PyodideRenderCanvas(...)
        else:
            canvas = GlfwRenderCanvas(size=(self.w, self.h), title=self.title)
            if self.fullscreen:
                glfw.set_window_monitor(...)
        ...
    
    def render(self, output, fps=30, duration=None, frame_count=None, ...):
        """Render to video file.
        
        Always uses OffscreenRenderCanvas for headless rendering.
        """
        canvas = OffscreenRenderCanvas(size=(self.w, self.h))
        ...
```

### Benefits
1. **Simpler API** - No enum, no parameter
2. **Self-documenting** - Method name tells you the backend
3. **Impossible invalid state** - Can't set wrong backend for method
4. **Environment-aware** - Automatically uses right backend

### Installation Changes
```toml
[project.optional-dependencies]
desktop = ["glfw>=2.0"]      # For run() on desktop
offline = ["imageio-ffmpeg>=0.5"]  # For render()
all = ["glfw>=2.0", "imageio-ffmpeg>=0.5"]
```

(No browser extra needed - Pyodide only works in browser environment)

## Phases

### Phase 1: Remove Backend enum and simplify constructor
**Goal:** Remove Backend enum, update Engine.__init__

**Done when:**
- [ ] Backend enum removed from engine.py
- [ ] Backend removed from __init__ parameters
- [ ] Engine stores title, w, h, fullscreen only
- [ ] Backend enum removed from __init__.py exports

**Depends on:** None

---

### Phase 2: Make run() context-aware
**Goal:** Auto-detect Pyodide vs GLFW based on sys.platform

**Done when:**
- [ ] run() checks sys.platform == "emscripten"
- [ ] Creates PyodideRenderCanvas in browser
- [ ] Creates GlfwRenderCanvas on desktop
- [ ] Handles fullscreen via GLFW on desktop
- [ ] ImportError if GLFW not installed on desktop

**Depends on:** Phase 1

---

### Phase 3: Make render() use offscreen unconditionally
**Goal:** render() always uses OffscreenRenderCanvas

**Done when:**
- [ ] render() no longer checks backend
- [ ] Always creates OffscreenRenderCanvas
- [ ] Works without display (CI-friendly)
- [ ] Produces valid video file

**Depends on:** Phase 1

---

### Phase 4: Update tests
**Goal:** Remove Backend-related tests, add context-based tests

**Done when:**
- [ ] Backend enum tests removed
- [ ] Backend parameter tests removed
- [ ] run() raises tests updated (if GLFW not available)
- [ ] render() produces video test passes
- [ ] All tests pass

**Depends on:** Phase 2, Phase 3

---

## Files to Modify
- `src/manifoldx/engine.py` - Remove Backend enum, update run()/render()
- `src/manifoldx/__init__.py` - Remove Backend export
- `tests/test_engine.py` - Update/remove Backend tests
- `pyproject.toml` - Remove browser extra (optional)

## Risks
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Breaking change for users | Low | API is cleaner, easy migration |
| Can't test Pyodide locally | Medium | Only works in browser anyway |
