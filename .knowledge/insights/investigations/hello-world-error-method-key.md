---
id: investigation-hello-world-error-method-key
created: 2026-04-03
type: investigation
status: resolved
severity: critical
---

# Investigation: Hello World Error - Missing 'method' Key

## Problem Statement
Running `hello_world.py` throws `KeyError: 'method'` when trying to create WgpuContext.

## Symptoms
```
KeyError: 'method'
File "/home/apiad/Projects/personal/manifoldx/src/manifoldx/engine.py", line 61, in _init_webgpu
    self._canvas = WgpuContext(present_info)
```

## Root Cause
The `WgpuContext` class requires a `present_info` dictionary with a "method" key ("screen" or "bitmap"). Our engine manually constructs the dict without this required field.

## Evidence
From `rendercanvas/contexts/wgpucontext.py`:
```python
def __new__(cls, present_info: dict):
    present_method = present_info["method"]  # KeyError if missing
    if present_method == "screen":
        return super().__new__(WgpuContextToScreen)
    elif present_method == "bitmap":
        return super().__new__(WgpuContextToBitmap)
```

## Resolution
Use rendercanvas's `GlfwRenderCanvas` instead of manually creating GLFW windows. The rendercanvas library:
1. Handles GLFW window creation properly
2. Provides `get_wgpu_context()` method that returns a pre-configured WgpuContext
3. Has `get_glfw_present_info(window)` helper that creates correct present_info dict

**Correct approach:**
```python
from rendercanvas.glfw import GlfwRenderCanvas

# Create canvas (this handles window creation)
canvas = GlfwRenderCanvas()

# Get wgpu context
wgpu_context = canvas.get_wgpu_context()

# Request adapter and device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = asyncio.run(adapter.request_device())

# Configure
wgpu_context.configure(device=device, format=wgpu.TextureFormat.bgra8unorm)
```

## Prevention
- Use rendercanvas's GlfwRenderCanvas for windowing + wgpu integration
- Don't manually construct present_info dicts - use rendercanvas helpers