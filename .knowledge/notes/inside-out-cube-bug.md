# Inside-Out Cube Bug

## Root Cause

**No depth buffer.** The render pass in `engine.py` has no depth attachment, so triangles are drawn in index buffer order. Back faces (dark) are drawn after front faces (bright), overwriting them.

Confirmed by projecting face centers:
- Bottom (y-, dark, ndc_z=0.96) drawn LAST → overwrites Top (y+, bright, ndc_z=0.95)
- Back (z-, dark, ndc_z=0.96) drawn after Front (z+, bright, ndc_z=0.94)

## Fix (2 files)

### `engine.py`
1. Create a depth texture (`depth24plus`) matching canvas size
2. Add `depth_stencil_attachment` to `begin_render_pass()`
3. Recreate depth texture if canvas resizes

### `renderer.py`
1. Add `depth_stencil` state to `create_render_pipeline()`:
   - format: `depth24plus`
   - depth_write_enabled: True
   - depth_compare: `less`
2. Change `cull_mode` from `none` to `back`
