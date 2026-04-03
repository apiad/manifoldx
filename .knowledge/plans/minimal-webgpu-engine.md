# Plan: Minimal WebGPU Engine

**Status: COMPLETE**

## Problem

Create a working engine skeleton that initializes a window, starts a WebGPU rendering loop, and renders one empty frame. The goal is to validate the API from `hello_world.py` works end-to-end.

## Solution

Build a minimal Engine class using GLFW for windowing and wgpu-py for rendering, with FIFO swap chain for vsync.

## Phases

### Phase 1: Engine Class Skeleton
- Create `src/manifoldx/engine.py` with Engine class
- Implement `__init__` accepting name, h, w, fullscreen
- Implement `@engine.startup` and `@engine.shutdown` decorators
- Implement `engine.quit()` and `engine.run()`
- Export Engine from `__init__.py`

### Phase 2: GLFW Window Integration
- Import glfw
- Create window in `run()` before main loop
- Store native window handle (GLFW window pointer)
- Destroy window on shutdown
- Handle window close events

### Phase 3: WebGPU Context
- Request GPU adapter via wgpu
- Request GPU device
- Create wgpu canvas wrapping GLFW window
- Create swap chain with FIFO present mode
- Handle texture format (BGRA8_UNORM)

### Phase 4: Main Render Loop
- Acquire swap chain texture
- Create command encoder
- Create render pass (no attachments - just clear color)
- Submit commands
- Present swap chain
- Add proper frame timing (vsync-waited)

### Phase 5: Verify hello_world.py
- Run the example
- Confirm window opens, renders one frame, exits cleanly

## Success Criteria

1. `hello_world.py` runs without errors
2. Window opens with correct dimensions
3. At least one frame is rendered (visible vsync)
4. Program exits cleanly on `engine.quit()`

## Risks

- **No GPU available**: Will fail gracefully with descriptive error
- **wgpu-py version**: Need compatible version for system
- **GLFW init failure**: Should handle and report cleanly

## Dependencies

- `glfw` - windowing
- `wgpu-py` - WebGPU binding
- `numpy` - for future phases (not needed yet)