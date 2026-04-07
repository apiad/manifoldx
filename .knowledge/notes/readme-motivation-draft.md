# README Motivation Draft

## The Problem

Domain researchers (biologists, physicists, epidemiologists, traffic engineers) need to:
1. Run large-scale simulations with 10⁴-10⁶ entities
2. Visualize results in 3D to understand behavior

**Currently they can't do both in pure Python:**

- **Matplotlib/Plotly** — 2D only, struggles with 10k+ animated points
- **Vispy** — OpenGL, steep learning curve
- **PyGfx** — Great for graphics, but no ECS structure for simulation
- **Game engines (PyGame, Arcade)** — 2D mostly, not designed for scientific viz
- **Specialized simulators (OpenMM, SUMO)** — Output data only, no integrated 3D view

**The gap:** No Python tool combines data-driven simulation + accessible 3D visualization in one package.

## The Solution

ManifoldX is an ECS-based rendering engine where:

1. **Simulation logic lives in pure NumPy** — vectorized operations over arrays, no Python loops
2. **Visualization is automatic** — spawn entities with meshes/materials, engine handles GPU rendering
3. **No graphics knowledge required** — researchers focus on physics, not shaders

```python
# Researcher writes physics (pure numpy)
@engine.system
def nbody_physics(query, dt):
    forces = compute_gravity_all_pairs(query[Transform].pos.data)  # vectorized
    velocities += forces * dt
    query[Transform].pos += velocities * dt

# Engine handles: GPU buffers, WGSL shaders, instanced draw calls
# Researcher gets: instant 3D visualization of their simulation
```

## Target Use Cases

| Domain | What They Simulate | What They See |
|--------|-------------------|---------------|
| Astrophysics | N-body gravitational clustering | Galaxy formation |
| Molecular dynamics | Protein folding trajectories | Atomic movement |
| Epidemiology | Disease spread (SIR model) | Infection spread in 3D |
| Crowd science | Evacuation flows | Pedestrian movement |
| Traffic | Vehicle car-following | Traffic jams forming |
| Weather | Lagrangian particle advection | Storm movement |

## Notebook Integration

Since it's pure Python + wgpu, works in:
- Jupyter notebooks (with wgpu ipywidgets)
- Quarto documents
- Streamlit/Gradio dashboards
- Standalone scripts

## Why Not Just Use X?

- **PyGfx** — Great rendering, but no ECS for simulation structure
- **VisPy** — OpenGL, high learning curve
- **Plotly** — 2D, animation performance issues at scale
- **Game engines** — Not designed for scientific visualization
- **Specialized simulators** — No integrated 3D output

## The Vision

Researchers should be able to:
1. Import manifoldx
2. Write their simulation in pure numpy (using ECS for data layout)
3. Spawn entities with built-in primitives and materials
4. See their simulation in 3D immediately

No graphics programming. No shader code. Just physics + visualization.

---

## Stats to Include

- 250 bodies at 60fps with N² gravity + collisions (example/nbody.py)
- Single draw call for all instances (instanced rendering)
- Microseconds ECS overhead per frame
- Supports 100k+ entities (limited by GPU, not engine)