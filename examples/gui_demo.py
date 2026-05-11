"""N-Body gravitational simulation with docked GUI control panel.

Demonstrates the GUI layer: interactive sliders to control physics parameters,
a reset button to re-randomize positions, a toggle (stub) for trails, and live
FPS display.
"""

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.gui import Panel, Text, Button, Slider, Toggle, ValueDisplay
from manifoldx.resources import sphere, PhongMaterial
import numpy as np

NUM_BODIES = 200
G = 20.0  # gravitational constant (mutable via slider)
DT = 0.01  # timestep (mutable via slider)
SOFTENING = 0.05  # prevents singularities at close range
MAX_SPEED = 20.0  # velocity clamp
SPHERE_RADIUS = 0.5  # base mesh radius
SIZE = 5 * NUM_BODIES ** (1 / 3)

engine = mx.Engine("N-Body with GUI")
engine.camera.fit(SIZE)

# Random initial positions spread uniformly + random masses (cube-root so
# volume ∝ mass for the visual sphere scale).
positions = mx.random.positions_in_box(NUM_BODIES, half_size=SIZE, rng=7)
masses = mx.random.scalars_uniform(NUM_BODIES, low=0.5, high=3.0, rng=7)
scales = (masses ** (1 / 3)).reshape(-1, 1)  # (N, 1) → broadcasts to (N, 3)

# Spawn all bodies at once (instanced rendering)
bodies = engine.spawn(
    Mesh(sphere(SPHERE_RADIUS, 12)),
    Material(PhongMaterial("#ffaa44")),
    Transform(pos=positions, scale=scales),
    n=NUM_BODIES,
)

# Velocities (not part of ECS — pure numpy)
velocities = np.zeros((NUM_BODIES, 3), dtype=np.float32)

# Mutable references for sliders (closures capture by value for primitives,
# by reference for containers).
G_ref = [G]
dt_ref = [DT]
trails_enabled = [False]

# Flag for deferred reset — set by reset_positions(), consumed by the system.
needs_reset = [False]


def reset_positions():
    """Signal the physics system to re-randomize positions on its next tick."""
    needs_reset[0] = True


@engine.system
def nbody_gravity(query: mx.Query[Transform], dt: float):
    """Physics step: compute gravity, integrate velocities and positions."""
    global velocities, positions
    if needs_reset[0]:
        positions[:] = mx.random.positions_in_box(NUM_BODIES, half_size=SIZE, rng=7)
        velocities[:] = 0
        query[Transform].pos.data[:NUM_BODIES] = positions
        needs_reset[0] = False
    pos = query[Transform].pos.data
    accel = mx.physics.gravity(pos, masses=masses, G=G_ref[0], softening=SOFTENING)
    velocities += accel * dt_ref[0]
    # Clamp speed to prevent runaway
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    scale = np.minimum(1.0, MAX_SPEED / np.maximum(speeds, 1e-6))
    velocities *= scale
    pos += velocities * dt_ref[0]
    query[Transform].pos.data[:] = pos


# Build GUI panel (docked top-right, but we'll use top-left + offset for safety)
panel = Panel(
    children=[
        Text("N-Body Controls"),
        Slider(name="G", min=0.1, max=20.0, value=G),
        Slider(name="dt", min=0.001, max=0.05, value=DT),
        Toggle(name="trails", value=False, label="Trails (stub)"),
        Button(name="reset", label="Reset"),
        ValueDisplay(
            getter=lambda: f"fps={1.0 / max(engine._last_dt, 1e-6):.1f}",
            min_width=140,
        ),
    ],
    anchor="top-left",
    offset=(20, 20),
    style_overrides={
        "width": 220,
        "padding": 8,
        "gap": 6,
        "bg": "#1a1a1aE0",
        "radius": 4,
    },
)

engine.gui.append(panel)


@engine.on("gui:slider:G:change")
def on_g_change(payload):
    G_ref[0] = payload["value"]


@engine.on("gui:slider:dt:change")
def on_dt_change(payload):
    dt_ref[0] = payload["value"]


@engine.on("gui:toggle:trails:change")
def on_trails_change(payload):
    trails_enabled[0] = payload["value"]


@engine.on("gui:button:reset:click")
def on_reset_click(payload):
    reset_positions()


if __name__ == "__main__":
    engine.cli()
