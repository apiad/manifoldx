---
id: nbody-simulation-demo
created: 2026-04-06
type: plan
status: done
expires: 2026-04-13
phases:
  - name: phase-1-spawn-with-scales
    done: false
  - name: phase-2-numpy-physics
    done: false
  - name: phase-3-demo-integration
    done: false
---

# Plan: N-Body Simulation Demo

## Context

Create a visually impressive N-body gravity simulation demo that showcases:
- Pure numpy vectorized physics (no Python loops)
- Instanced rendering of 100 bodies
- Size scaling by mass (cubic law: radius ∝ mass^(1/3))
- Initial random cloud state

## Phases

### Phase 1: Spawn with Scales
**Goal:** Spawn 100 bodies with different scales based on mass

**Deliverable:** 
- Modify spawn to include per-body scale in Transform
- Random positions in (-15, 15)³
- Random masses in (0.5, 3.0)
- Compute scale: `scale = mass ** (1/3)`

**Done when:**
- [ ] 100 entities spawned with varying scales
- [ ] Spheres render at different sizes
- [ ] Single draw call confirmed

### Phase 2: Numpy Physics  
**Goal:** Implement pure numpy gravitational force calculation

**Deliverable:**
- System that computes all-pairs gravitational forces
- No Python loops - pure numpy broadcasting
- Update velocities and positions

**Formula (vectorized):**
```python
# diff[i,j] = pos[j] - pos[i] for all pairs
diff = positions[None, :] - positions[:, None]  # (100, 100, 3)
dist = np.linalg.norm(diff, axis=2)[:, :, np.newaxis]  # (100, 100, 1)
dist = np.maximum(dist, 0.1)  # Softening to avoid div by zero

# F = G * m1 * m2 / r^2
force_mag = G * masses[None, :] * masses[:, None] / (dist ** 3)

# Force vectors: F * direction
forces = force_mag * diff  # (100, 100, 3)

# Net force on each body
net_force = np.sum(forces, axis=1)  # (100, 3)
```

**Done when:**
- [ ] Physics system runs each frame
- [ ] Bodies move under gravitational influence
- [ ] No Python for-loops in physics code

### Phase 3: Demo Integration
**Goal:** Complete demo file with camera and lighting

**Deliverable:**
- `examples/nbody.py` - complete runnable demo
- Camera positioned to view the whole cloud
- Point light for visibility
- Prints to console on startup

**Done when:**
- [ ] Demo runs and shows 100 orbiting/falling bodies
- [ ] Clean exit works (quit works)
- [ ] Test passes: `python -m examples.nbody`

## Success Criteria

1. 100 bodies render in single draw call
2. Different body sizes visible (small to large)
3. Bodies move under gravity - chaotic motion
4. No slowdown (smooth framerate)
5. Clean exit with no crash

## Technical Notes

- Use Transform.scale to encode size (handled by renderer)
- Mass stored in a numpy array, not in ECS (simpler for physics)
- Positions read from Transform component each frame
- Update Transform with new positions after physics