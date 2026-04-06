# ManifoldX

[![PyPI version](https://img.shields.io/pypi/v/manifold-gfx)](https://pypi.org/project/manifold-gfx/)
[![Python versions](https://img.shields.io/pypi/pyversions/manifold-gfx)](https://pypi.org/project/manifold-gfx/)
[![License](https://img.shields.io/pypi/l/manifold-gfx)](LICENSE)
[![Tests](https://github.com/apiad/manifoldx/actions/workflows/test.yml/badge.svg)](https://github.com/apiad/manifoldx/actions/workflows/test.yml)

A real-time 3D rendering engine built on pure [wgpu](https://github.com/gfx-rs/wgpu) with an Entity Component System (ECS) architecture. Written in Python with numpy for high-performance data handling.

> ⚠️ **Beta / Academic Project** — This is an experimental proof-of-concept exploring the extent to which Python can be used for high-performance graphics via wgpu. Not recommended for production use. Expect bugs, breaking changes, and missing features.

## Motivation

Domain researchers need to run large-scale simulations (10⁴-10⁶ entities) and visualize results in 3D. **Currently they can't do both in pure Python.**

| Tool                       | Simulation | Visualization          | Language      |
| -------------------------- | ---------- | ---------------------- | ------------- |
| Matplotlib/Plotly          | ✅          | 2D only, slow at scale | Pure Python   |
| VisPy                      | ❌          | OpenGL, steep curve    | Python + GLSL |
| PyGfx                      | ❌          | Excellent rendering    | Pure Python   |
| Game engines (PyGame)      | ✅          | 2D mostly              | Pure Python   |
| Specialized (OpenMM, SUMO) | ✅          | Data export only       | Fortran/C++   |

**The gap:** No Python tool combines data-driven simulation with accessible 3D visualization in one package.

### The Vision

ManifoldX gives researchers both:

1. **Simulation in pure NumPy** — vectorized operations over entity arrays, no Python loops in the hot path
2. **3D visualization without graphics knowledge** — spawn entities with meshes/materials, the engine handles GPU rendering

```python
# Write physics in pure numpy (data-driven, not OOP)
@engine.system
def nbody_physics(query, dt):
    forces = compute_gravity_all_pairs(query[Transform].pos.data)
    velocities += forces * dt
    query[Transform].pos += velocities * dt

# Engine handles GPU buffers, WGSL shaders, instanced draw calls
# You get 3D visualization of your simulation instantly
```

**Target domains:** Astrophysics (galaxy formation), molecular dynamics (protein folding), epidemiology (disease spread), crowd science (evacuation flows), traffic engineering (vehicle flow), weather (particle advection).

**Works in:** Jupyter notebooks, Quarto documents, Streamlit dashboards, standalone scripts.

> ⚠️ **Spoiler:** Python is surprisingly capable at real-time 3D. The N-body demo runs 500 bodies with N² gravity — 250,000 force pairs per frame in pure numpy.

### Why Not PyGfx?

[PyGfx](https://github.com/pygfx/pygfx) is an excellent rendering engine and has been a key source of inspiration — we've learned a lot from its WGSL patterns, material system, and API design. However, it's fundamentally a **scene graph** engine (object-oriented hierarchy of transforms), not a data-driven ECS. For large-scale simulations with vectorized physics, the OOP overhead of traversing a scene graph becomes a bottleneck. ManifoldX uses a flat SoA data layout where simulation logic operates directly on numpy arrays.

**Technically this is a game engine** — the ECS + instanced rendering + PBR pipeline is exactly what you'd use for a game. But that's not the focus. The goal is making 3D visualization accessible to researchers who don't know anything about graphics.

## Installation

```bash
pip install manifold-gfx
# or
uv add manifold-gfx
```

**Requirements:**
- Python 3.13+
- GPU with WebGPU support (via wgpu backend)
  - Vulkan on Linux
  - Metal on macOS
  - D3D12 on Windows

## Quick Start

```python
import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, PointLight, cube, sphere

# Create engine with default settings
engine = mx.Engine("My First Scene")

# Create a cube and sphere
cube_geo = cube(1, 1, 1)
sphere_geo = sphere(0.7, 32)

# Create PBR materials (roughness: 0-1, metallic: 0-1)
red_shiny = StandardMaterial(color="#ff3333", roughness=0.15, metallic=0.9)
blue_dull = StandardMaterial(color="#3366ff", roughness=0.8, metallic=0.0)

# Spawn entities
engine.spawn(
    Mesh(cube_geo),
    Material(red_shiny),
    Transform(pos=(-1.5, 0, 0)),
)

engine.spawn(
    Mesh(sphere_geo),
    Material(blue_dull),
    Transform(pos=(1.5, 0, 0)),
)

# Add an orbiting light
light = PointLight(color="#ffffff", intensity=15.0, position=(5, 5, 5))
engine.set_lights([light])

# Animate
@engine.system
def animate_lights(query: mx.Query[Transform], dt: float):
    t = engine.elapsed
    light.position = (
        5 * np.cos(t * 0.7),
        3 + np.sin(t * 0.5) * 2,
        5 * np.sin(t * 0.7),
    )

# Auto-fit camera to view the scene
engine.camera.fit(radius=5.0, azimuth=30, elevation=35)

# Run!
engine.run()
```

Save as `my_scene.py` and run:

```bash
python my_scene.py
```

## N-Body Simulation

A pure-numpy gravitational simulation running 500 bodies in real-time at a single draw call (instanced rendering).

**Physics:** All pairwise forces are computed with a single vectorized numpy expression — no Python loops in the hot path. For N bodies this means N² = 250,000 force computations per frame, each a 3-component vector.

```python
@engine.system
def nbody_physics(query: mx.Query[Transform], dt: float):
    global velocities
    pos = query[Transform].pos.data

    # All-pairs position differences (N, N, 3) — one numpy broadcast
    diff = pos[None, :] - pos[:, None]

    # Pairwise distances (N, N)
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, SOFTENING)

    # Gravitational force magnitude for every pair
    force_mag = G * (masses[None, :] * masses[:, None]) / dist**2

    # Net force on each body: sum over all other bodies
    direction = diff / dist[:, :, None]
    forces = force_mag[:, :, None] * direction
    net_force = forces.sum(axis=1)

    # Integrate: F = ma → a = F/m
    velocities += (net_force / masses[:, None]) * dt
    query[Transform].pos += velocities * dt
```

> See `examples/nbody.py` for the full implementation with velocity damping and speed clamping.

### Ideal Gas Simulation

The `examples/gas.py` demo shows the other side: **no gravity, all collisions**. 500 particles bounce inside an invisible box with elastic collisions and wall reflections — a kinetic theory simulation in pure numpy.

**Collisions** find overlapping pairs with a vectorized comparison, filter with `np.where(np.triu(...))`, then resolve impulse with `np.add.at` for safe accumulation. **Wall reflections** are a single vectorized mask: `velocities[next_pos < wall] = np.abs(...)`.

> See `examples/gas.py` for the full implementation.

## Examples

| Example          | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `hello_world.py` | Minimal empty window                                     |
| `cube.py`        | Rotating cube with Phong material                        |
| `pbr_demo.py`    | 3×2 grid demonstrating PBR materials + 3 orbiting lights |
| `spheres.py`     | Many spheres with physics-like behavior                  |
| `nbody.py`       | 500-body gravitational simulation with pure-numpy physics |
| `gas.py`         | 500-particle ideal gas with collisions and virtual walls  |
| `boids.py`       | 300-agent flocking simulation with emergent swarm behavior |

Run an example:

```bash
python -m examples.nbody   # N-body gravitational simulation
python -m examples.gas     # Ideal gas with elastic collisions
python -m examples.boids   # Boids flocking with soft boundary
python -m examples.pbr_demo
```

## Features

### ECS Architecture
- **Structure of Arrays (SoA)** layout for each component
- Vectorized numpy operations for batch transforms
- Free-list for efficient entity reuse
- Component view with operator overloads (`+=`, `*=`, etc.)

### Rendering
- **Instanced drawing** — single draw call per (geometry, material) batch
- **Material-specific pipelines** — each material type compiles its own WGSL shader
- **Transform caching** — dirty-flag optimization to avoid recomputing matrices
- **Shared transform buffer** — all instance transforms uploaded once per frame

### Materials & Lighting
- **BasicMaterial** — unlit flat color with simple diffuse
- **StandardMaterial** — full PBR with GGX BRDF
  - Roughness/metallic workflow
  - Multiple point lights with inverse-square attenuation
  - Reinhard tonemapping + gamma correction
- **External lights** — passed to engine like camera (not in ECS)

### Camera
- Perspective projection (WebGPU NDC)
- Spherical coordinate orbit controls
- Fit/fit_bounds for automatic framing

### Geometries
- Cube (with normals)
- UV Sphere (with normals, CCW winding)
- Plane (with normals)

## Architecture Highlights

The ECS uses numpy arrays for all component data. When you call `query[Transform].pos += velocity * dt`, it's a single vectorized numpy operation spanning thousands of entities.

**Real-world examples:** The N-body demo (`examples/nbody.py`) simulates 500 bodies with 250,000 pairwise gravitational force computations per frame. The ideal gas demo (`examples/gas.py`) runs 500 particles with elastic collisions and wall reflections. Both are pure numpy with zero Python loops.

## Limitations (Known)

- ❌ No shadows
- ❌ No texture support
- ❌ No environment/IBL mapping
- ❌ Single material params per draw call (not per-instance)
- ❌ Only point lights in PBR shader
- ❌ Limited to ~100k entities

## Future Ideas

This is an academic/experimental project. Ideas for future development:

1. **Per-instance material data** — Storage buffer for varying roughness/metallic per instance in a single draw
2. **Shadow mapping** — Shadow pass + PCF sampling
3. **Texture maps** — Diffuse, normal, roughness textures via storage buffers
4. **Spot/Directional lights** — Extend PBR shader
5. **Environment mapping** — IBL with prefiltered radiance
6. **Skinned animation** — Bone transforms in vertex shader
7. **Post-processing** — Bloom, DOF, TAA
8. **Deferred rendering** — Forward+ / clustered lighting for many lights

## Contributing

Contributions welcome! This is an educational project — all skill levels encouraged.

**Areas needing work:**
- Bug fixes and stability improvements
- Additional geometry types (torus, cylinder, etc.)
- More material types (toon, unlit with texture)
- Shadow implementation
- Performance profiling and optimization

**Getting started:**

```bash
# Clone and set up
git clone https://github.com/apiad/manifoldx.git
cd manifoldx
pip install -e ".[dev]"

# Run tests
make test

# Run an example
python -m examples.cube
```

## Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_ecs.py -v
```

Current test coverage: **150+ tests** covering ECS operations, components, materials, rendering, and camera.

## License

MIT License — See LICENSE file.

## Credits

- [wgpu](https://github.com/gfx-rs/wgpu) — Pure Python WebGPU bindings
- [PyGfx](https://github.com/pygfx/pygfx) — Reference for WGSL shader patterns
- [rendercanvas](https://github.com/pygfx/rendercanvas) — Window management

---

**Disclaimer:** This project is for educational and research purposes. Not optimized for production use. Performance characteristics will vary by hardware and Python version.
