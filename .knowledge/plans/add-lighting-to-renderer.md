---
id: add-lighting-to-renderer
created: 2026-04-06
modified: 2026-04-06
type: plan
status: active
expires: 2026-04-13
phases:
  - name: Move camera back
    done: false
    goal: Increase default camera distance to see more cubes
  - name: Add normals to cube geometry
    done: false
    goal: Generate face normals for flat shading
  - name: Update shader with simple lighting
    done: false
    goal: Add ambient + diffuse from directional light
---

# Plan: Add Simple Lighting to Renderer

## Context
The cubes show as flat silhouettes because there's no lighting. Need to add basic illumination.

## Phases

### Phase 1: Move Camera Back
**Goal:** Increase default camera distance

**Done when:**
- [ ] Change default camera position from (0, 0, 5) to (0, 10, 20)
- [ ] Look-at stays at (0, 0, 0)

### Phase 2: Add Normals to Geometry
**Goal:** Compute face normals for each triangle

**Done when:**
- [ ] Add normals array to cube geometry (6 faces * 2 triangles * 3 vertices = 36 normals)
- [ ] Update vertex format to include normals (vec3)
- [ ] Geometry registry creates vertex buffer with normals

### Phase 3: Lighting in Shader
**Goal:** Add ambient + diffuse to fragment shader

**Done when:**
- [ ] Pass light direction uniform (e.g., normalized (1, 1, 1))
- [ ] Pass ambient intensity (0.3)
- [ ] Compute: `color = material_color * (ambient + diffuse * max(0, dot(normal, light_dir)))`
- [ ] Update vertex input to include normal

## Success Criteria
- [ ] Running cube.py shows a visibly lit 3D cube (not flat silhouette)
- [ ] Running cubes.py shows cubes with shading that reveals their 3D form
- [ ] Depth is perceivable (cubes closer to light are brighter)

## Technical Notes
- Use flat shading (same normal for all 3 vertices of each triangle) - simpler
- Light direction can be uniform for now (no per-instance lights)
- Keep existing unlit path as fallback if no normals