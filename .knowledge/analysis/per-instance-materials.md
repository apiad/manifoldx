# Per-Instance Material Properties Analysis

## Current State

### How Materials Work Now

The current implementation uses a **single uniform buffer per (geometry_id, material_type) batch**:

1. **Python side** (`resources.py`):
   - `BasicMaterial.get_data(n, registry)` returns `(n, 4)` array but only first row is used
   - `StandardMaterial.get_data(n, registry)` returns `(n, 8)` array but only first row is used

2. **Renderer** (`renderer.py` line 483-491):
   ```python
   mat_data = mat_obj.get_data(instance_count, engine._material_registry)
   # ...
   first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
   self._device.queue.write_buffer(mat_buffer, 0, first_row.astype(np.float32).tobytes())
   ```

3. **Result**: All instances in a batch share the same material properties.

### Material Classes

| Class | Properties | Uniform Size |
|-------|-----------|---------------|
| `BasicMaterial` | color (vec4) | 16 bytes |
| `StandardMaterial` | albedo(3), roughness, metallic, ao | 32 bytes |
| `PhongMaterial` | color, shininess | 16 bytes (uses Basic shader) |

---

## What PyGfx Does

PyGfx is a **scene graph system**, not an ECS:

- Each `Mesh` object has its own `Material` instance
- Material is attached to the mesh object, not stored in a flat array
- No built-in per-instance material properties in material classes
- Uses `color_mode` to switch between uniform color and vertex colors

---

## Options for Per-Instance Materials

### Option A: Vertex Attributes (Recommended)

Add per-instance data as vertex buffers with `step_mode: instance`:

**WGSL changes:**
```wgsl
// Add to vertex input
struct InstanceMaterialInput {
    @location(2) color: vec4<f32>,  // or albedo(3) + roughness + metallic + ao
    @location(3) ...
};

@vertex
fn vs_main(in: VertexInput, instance_mat: InstanceMaterialInput) -> VertexOutput {
    // Use instance_mat instead of material uniform
}
```

**Python changes:**
- Add new vertex buffer layout with per-instance material data
- Collect per-instance material data from entity Material component

**Pros:** Most GPU-efficient, native WebGPU instancing
**Cons:** Requires shader changes, limited data per instance (vs storage buffer)

---

### Option B: Storage Buffer

Similar to transforms, use a storage buffer for per-instance material:

**WGSL changes:**
```wgsl
struct InstanceMaterials {
    data: array<MaterialInstance>,
};

@group(0) @binding(4) var<storage, read> instance_materials: InstanceMaterials;

@fragment
fn fs_main(in: VertexOutput) {
    let mat = instance_materials[in.instance];
    // Use mat.color, mat.roughness, etc.
}
```

**Python changes:**
- Create storage buffer for instance materials
- Upload per-instance data from entity Material component

**Pros:** More flexible, larger data per instance, easier to extend
**Cons:** Slightly more overhead than vertex attributes

---

### Option C: Multiple Draw Calls

Call `draw_indexed` separately for each unique material combination.

**Python changes:**
- Group by (geometry_id, material_id) instead of just (geometry_id, material_type)
- Create pipeline per unique material instance

**Pros:** Simple, works with existing uniform buffer system
**Cons:** Many draw calls, poor scalability for many instances

---

## Implementation Path

### Phase 1: Add Instance Material Storage

1. **Entity Component**: Add material instance data to ECS
   - Create `MaterialInstance` component with per-instance properties
   - Or use existing `Material` component with per-instance values

2. **Renderer** (`renderer.py`):
   - Add storage buffer or vertex buffer for instance materials
   - Collect per-instance material data during render
   - Upload to GPU

3. **Shaders** (`resources.py`):
   - Modify to accept per-instance data instead of uniform
   - Use `@builtin(instance_index)` to read correct instance data

### Phase 2: Material Class Changes

1. **Update `Material.get_data()`** to accept instance indices
2. **Add instance-aware `get_instance_data()` method**
3. **Support per-property overrides** (e.g., only color varies, roughness constant)

### Phase 3: API Simplification

1. **Easy setter**: `entity.material.color = "#ff0000"` per instance
2. **Batch setter**: `set_all_colors(entities, "#ff0000")`
3. **Property masks**: Track which properties vary per-instance

---

## Key Technical Details

### WGSL Instance Index

```wgsl
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};
```

The `instance_index` builtin gives you the current instance ID (0 to instance_count-1).

### Memory Layout

**StandardMaterial per instance (8 floats = 32 bytes):**
- offset 0: albedo (vec3) - 12 bytes
- offset 12: roughness (f32) - 4 bytes  
- offset 16: metallic (f32) - 4 bytes
- offset 20: ao (f32) - 4 bytes
- offset 24-31: padding - 8 bytes

### Vertex Buffer vs Storage Buffer

| Aspect | Vertex Buffer | Storage Buffer |
|--------|--------------|----------------|
| Max size per instance | 32 bytes (4 vec4) | Unlimited |
| Access in vertex shader | Direct | Via buffer read |
| Access in fragment shader | Interpolated or via varyings | Direct |
| Performance | Slightly faster | Good |

For material properties, either works. Vertex buffer is simpler if properties are limited.

---

## Files to Modify

1. **`src/manifoldx/resources.py`** - Material classes with new `get_instance_data()`
2. **`src/manifoldx/renderer.py`** - Add instance material buffer, collect and upload data
3. **`src/manifoldx/ecs.py`** - May need new component for per-instance material data
4. **`src/manifoldx/components.py`** - Update Mesh/Material components

---

## References

- [Learn Wgpu: Instancing Tutorial](https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/)
- [Three.js InstancedMesh colors](https://stackoverflow.com/questions/76493040/applying-a-different-color-to-a-threejs-shader-material-at-each-instance)
- WebGPU spec: `@builtin(instance_index)`