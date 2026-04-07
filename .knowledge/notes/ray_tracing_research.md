# GPU Ray Tracing in WebGPU/wgpu - Research Summary

## Current Status (April 2025)

### WebGPU Ray Tracing Support

WebGPU has **experimental** ray tracing support through two mechanisms:

1. **Ray Query (Inline Ray Tracing)** - Available now
2. **Ray Tracing Pipelines** - In development

### wgpu (Rust) - Experimental Ray Tracing

wgpu (Rust) has experimental ray tracing with `Features::EXPERIMENTAL_RAY_QUERY`:

**Acceleration Structures:**
- `Blas` - Bottom-Level Acceleration Structure (per-geometry)
- `Tlas` - Top-Level Acceleration Structure (instance matrix + custom data)

**Shader Support (via Naga):**
```wgsl
enable wgpu_ray_query;

// Ray query functions
rayQueryInitialize(...)
rayQueryProceed(...)
rayQueryGetCommittedIntersection(...)
rayQueryGenerateIntersection(...)
rayQueryConfirmIntersection(...)

// New shader stages
@ray_generation
@any_hit
@closest_hit
@miss
```

### wgpu-py (Python) - NOT SUPPORTED

The Python bindings (`wgpu-py`) **do not expose** ray tracing features:
- No `EXPERIMENTAL_RAY_QUERY` feature
- No acceleration structure APIs
- No ray query bindings

### PyGfx - NOT SUPPORTED

PyGfx is a **rasterization-based renderer**:
- No ray tracing implementation
- No path tracing
- No ray query usage
- Focuses on WGSL shaders and traditional rendering

## Practical Implications for ManifoldX

### Current Reality
- ManifoldX uses rasterization via wgpu-py
- Ray tracing requires Rust wgpu with experimental features
- Python bindings would need significant additions

### Options for Future

1. **Hybrid Rendering** (Medium-term)
   - Keep rasterization for primary rendering
   - Add ray queries for shadows/reflections
   - Requires wgpu-py to add `EXPERIMENTAL_RAY_QUERY`

2. **Full Ray Tracing** (Long-term)
   - Requires wgpu-py to add full ray tracing API
   - Major undertaking for Python bindings
   - Would need to expose Blas/Tlas creation

3. **Alternative Approaches** (Now)
   - Software ray tracing (Embree via pyembree)
   - Pre-computed light maps/bake AO
   - Screen-space approximations (SSAO, SSR)

## External Resources

- [wgpu Ray Tracing Spec](https://github.com/gfx-rs/wgpu/blob/trunk/docs/api-specs/ray_tracing.md)
- [WebGPU Ray Tracing Extension (webrtx)](https://github.com/codedhead/webrtx)
- [wgpu Ray Tracing PR](https://github.com/gfx-rs/wgpu/pull/6552)
- [three-gpu-pathtracer](https://github.com/gkjohnson/three-gpu-pathtracer) - JS path tracer reference

## Conclusion

**Ray tracing in WebGPU/wgpu is experimental and not available in Python.** The technology exists in Rust wgpu, but the Python bindings don't expose it yet. For ManifoldX:

- Current focus should remain on rasterization + PBR
- Shadow maps can be added without ray tracing
- Path tracing would require either:
  - Waiting for wgpu-py to add ray query support
  - Implementing software path tracing with numpy
