# Boids Simulation Design

## Rules (Craig Reynolds, 1986)

Three local rules per boid, each computed from neighbors within a perception radius:

1. **Separation** — steer away from nearby neighbors (avoid crowding)
2. **Alignment** — match average velocity of neighbors (flock heading)
3. **Cohesion** — steer toward center of mass of neighbors (stay together)

Plus one global rule:
4. **Attractor** — steer toward a central point (prevents flock from drifting to infinity)

## Vectorization Strategy

All rules reduce to: "for each boid, compute a weighted average over neighbors."

### Step 1: All-pairs distance matrix
```
diff = pos[None, :] - pos[:, None]   # (N, N, 3)
dist = np.linalg.norm(diff, axis=2)  # (N, N)
```

### Step 2: Neighbor mask
```
neighbors = (dist < PERCEPTION_RADIUS) & (dist > 0)  # (N, N) bool
neighbor_count = neighbors.sum(axis=1, keepdims=True)  # (N, 1)
```

### Step 3: Separation (inverse-distance weighted repulsion)
For each boid i, sum of (normalized direction away from each neighbor j) weighted by 1/dist:
```
# Direction from j to i (away from neighbor)
sep_dir = -diff  # (N, N, 3) — points away from each neighbor
# Weight by inverse distance (closer = stronger repulsion)
inv_dist = np.zeros_like(dist)
inv_dist[neighbors] = 1.0 / dist[neighbors]
sep_force = (sep_dir * inv_dist[:, :, None]).sum(axis=1)  # only where neighbors=True
```
But we need to mask to only neighbors. Use:
```
masked = sep_dir * (neighbors[:, :, None] * inv_dist[:, :, None])
sep_force = masked.sum(axis=1)
```

### Step 4: Alignment (match neighbor velocity)
Average velocity of neighbors:
```
# Sum neighbor velocities
vel_sum = (vel[None, :] * neighbors[:, :, None]).sum(axis=1)  # (N, 3)
avg_vel = vel_sum / np.maximum(neighbor_count, 1)
align_force = avg_vel - vel  # steer toward average
```

### Step 5: Cohesion (steer toward neighbor center of mass)
```
pos_sum = (pos[None, :] * neighbors[:, :, None]).sum(axis=1)  # (N, 3)
center = pos_sum / np.maximum(neighbor_count, 1)
cohesion_force = center - pos  # steer toward center
```

### Step 6: Central attractor
```
attractor_force = -pos  # always points toward origin
# Scale by distance so it's gentle near center, strong far away
```

### Step 7: Speed clamping
Boids should have a min and max speed to look natural:
```
speed = np.linalg.norm(vel, axis=1, keepdims=True)
vel = np.where(speed > MAX_SPEED, vel * MAX_SPEED / speed, vel)
vel = np.where(speed < MIN_SPEED, vel * MIN_SPEED / np.maximum(speed, 1e-6), vel)
```

## Parameters (tuning)

| Parameter | Value | Notes |
|-----------|-------|-------|
| NUM_BOIDS | 500 | Good visual density |
| PERCEPTION_RADIUS | 3.0 | How far each boid "sees" |
| SEPARATION_WEIGHT | 1.5 | Strongest — avoid collisions |
| ALIGNMENT_WEIGHT | 1.0 | Match heading |
| COHESION_WEIGHT | 0.8 | Group together |
| ATTRACTOR_WEIGHT | 0.3 | Gentle pull to center |
| MAX_SPEED | 8.0 | Visual speed cap |
| MIN_SPEED | 2.0 | Prevents hovering |
| SPHERE_RADIUS | 0.1 | Small particles |
| SPAWN_RADIUS | 10.0 | Initial spread |

## Visual

- Small spheres (0.1 radius), all same size
- Could color by speed or by which sub-flock they're nearest to
- Camera fit to ~15 radius

## Memory Concern

N=500 → (500, 500, 3) diff matrix = 3M floats = 12MB. Fine.
N=1000 → 12M floats = 48MB. Still fine.
N=2000 → 48M floats = 192MB. Getting tight but doable.

## No Python Loops

Every rule is a masked sum over axis=1 of an (N, N, 3) tensor.
The only branching is `if neighbor_count > 0` which we handle with `np.maximum(count, 1)`.
