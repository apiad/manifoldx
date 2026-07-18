# Labyrinth FPS Demo Design

Status: **spec / approved, not yet implemented** (2026-07-18).

A first-person demo: walk a procedurally-generated maze holding a flashlight in
the dark. Showcases three engine features together — keyboard movement, mouse
look, and a single shadow-casting spot light — inside one self-contained example.

## Goals

- Show off `engine.input` (WASD polling + mouse-drag look) and the spot-light +
  spot-shadow subsystem in one cohesive, fun demo.
- FPS feel: eye-height camera, grounded movement (no flying), wall collision, a
  flashlight that lights only where you aim and casts shadows down the corridors.
- Fully procedural + seeded (reproducible) maze.
- Self-contained in one example file, following the `examples/input_fly.py`
  pattern (inline generation, module-level engine, `@engine.on("startup")` build,
  `@engine.system` per-frame controller, `engine.cli()` entry).
- A scripted flythrough so `--render` can produce an MP4 (live input can't be
  captured headlessly).

## Non-goals

- Enemies, pickups, minimap, HUD, sound — pure movement + light showcase.
- Physics beyond axis-aligned wall collision (no gravity, jumping, slopes).
- A ceiling (open-top maze; see Lighting).
- Pointer-lock mouse look (the input layer has none — line 235 of `input.py`,
  "No locking"). Look is drag-based.
- A reusable maze library. Generation lives inline in the example. (If a tested,
  importable generator is wanted later, extract to `manifoldx.maze` then.)

## Engine facts this relies on (verified)

- `engine.input.is_pressed("w"/"a"/"s"/"d")`, `is_mouse_pressed(2)` (right button),
  `mouse_delta` (this-frame accumulator). No cursor capture.
- `engine.camera.position` / `.target` are settable numpy vec3; `get_forward()`,
  `get_right()` return basis vectors.
- `@engine.system def f(query: Query[Transform], dt): ...` runs per frame;
  `@engine.on("startup")` runs once before the first frame.
- `engine.spawn(Mesh(geo), Material(mat), Transform(pos=, scale=), n=1)`.
- `cube(w, h, d)` geometry (`manifoldx.resources.cube`).
- **`StandardMaterial` is required** for floor + walls — the spot-light term lives
  only in the `StandardMaterial` shader. `PhongMaterial` / `BasicMaterial` use other
  shaders with no spot term, so a flashlight would not light them.
- `engine.set_spot(SpotLight(...))` + `engine.enable_shadows(...)` (shipped).
- `engine.cli()` parses `--render/--duration/--fps/--output`; `engine.render(...)`
  drives headless frames.

## Components

### 1. Maze generation — `generate_maze(cols, rows, seed) -> np.ndarray`

Recursive backtracker (randomized DFS) on a grid where cell coordinates are odd
and walls sit between them. Produces a **perfect maze** (fully connected, exactly
one path between any two open cells, winding corridors, no loops).

- Grid dimensions are odd: `W = 2*cols + 1`, `H = 2*rows + 1`. `True` = wall.
- Start all-wall; carve from cell (1,1): push a stack, at each step pick a random
  unvisited neighbour two cells away, knock out the wall between, recurse; backtrack
  when stuck.
- Seeded via Python's `random.Random(seed)` (a normal script — `random` is fine).
- Returns a 2D `bool` numpy array `wall[r][c]`.
- Chosen over random wall placement (can strand unreachable cells) and Prim's (same
  result class, no advantage here).

Rationale for a helper (not a one-off loop): connectivity is the one thing that can
silently break, so it's a named, independently-checkable function.

### 2. Geometry build — `@engine.on("startup")`

- **Floor**: one wide thin cube under the whole maze
  (`scale=(W*CELL, 0.1, H*CELL)`, `pos y = -0.05`), dark `StandardMaterial`.
- **Walls**: one cube per `wall[r][c] == True` cell, `scale=(CELL, WALL_H, CELL)`,
  positioned at that cell's world centre, `pos y = WALL_H/2`. Mid-grey
  `StandardMaterial` (required so the flashlight lights them — see Engine facts).
- One shared `cube` mesh + per-entity `Transform` → the renderer batches into **two
  draw calls** (floor + walls) regardless of maze size (same trick as `input_fly`).
- World mapping: cell `(r, c)` → world `x = (c - W/2)*CELL`, `z = (r - H/2)*CELL`.
- Constants: `CELL = 3.0`, `WALL_H = 3.0`, maze `cols = rows = 8` (→ 17×17 grid;
  tune for framerate). Player spawns at the start cell `(1,1)` centre, eye height
  `EYE = 1.6`.

### 3. FPS controller — `@engine.system`

- **Look (drag)**: while `is_mouse_pressed(2)`, read `mouse_delta`; `yaw += dx*SENS`,
  `pitch = clamp(pitch - dy*SENS, -89, 89)`. Forward vector from yaw/pitch (same
  `_direction_from` math as `input_fly`).
- **Move (grounded)**: horizontal forward = forward with `y=0` renormalised; right =
  `cross(worldUp, hforward)`. `W/S` ± hforward, `A/D` ± right. No vertical movement —
  camera stays at `EYE`. Speed `MOVE_SPEED` (e.g. 5 u/s) × `dt`.
- Writes `camera.position` (with collision, below) and
  `camera.target = position + forward`.

### 4. Collision — inside the controller, before applying movement

Axis-separated grid collision so the player slides along walls instead of sticking:

```
proposed = pos + move_x            # try X only
if not _hits_wall(proposed):  pos = proposed
proposed = pos + move_z            # then Z only
if not _hits_wall(proposed):  pos = proposed
```

`_hits_wall(world_pos)`: sample the player's circle (centre ± `PLAYER_R` on X and Z,
`PLAYER_R ≈ 0.3*CELL`), convert each sample to a grid cell
(`c = round(x/CELL + W/2)`, `r = round(z/CELL + H/2)`), return `True` if any sample
lands in a `wall` cell or out of bounds. O(1) per call.

### 5. Flashlight — updated each frame in the controller

- One `SpotLight` (`engine.set_spot`), warm colour, `inner_angle ≈ 0.22`,
  `outer_angle ≈ 0.38`, intensity tuned for the corridor scale, `distance` (range) a
  few cells.
- Each frame: `spot.position = camera.position`,
  `spot.direction = camera.get_forward()`. So the beam always points where you look.
- `engine.enable_shadows(resolution=2048, pcf_radius=2, ...)` → walls cast shadows
  down the corridors as you sweep the beam.
- **Lighting**: no sun. The scene is dark (only the shader's 0.03 ambient) except the
  cone. Open top (no ceiling) keeps perspective framing clean and lets the beam read
  against the dark.

### 6. Flythrough (headless MP4) — `--flythrough`, or auto under `--render`

Live input can't be captured headlessly, so a scripted camera path drives the MP4:

- Precompute an ordered list of open-cell centres to visit — a DFS/BFS traversal of
  the corridors from the start (covers more of the maze than the shortest path).
- A `@engine.system` (active only in flythrough mode) advances a path parameter by
  `dt`: lerp `camera.position` between consecutive cell centres, and turn `yaw`
  smoothly toward the next cell's direction. Flashlight follows as in §5.
- Under `engine.render(output=...)` this yields a smooth walkthrough MP4.
- Mode select: a module-level flag flipped by a `--flythrough` arg (or simply: when
  `--render` is passed, use the flythrough controller instead of the input one).

## Data flow

```
startup: generate_maze(seed) -> wall grid -> spawn floor + wall cubes;
         set_spot(); enable_shadows(); place camera at start cell.
each frame (interactive): input -> yaw/pitch + move vector -> collision -> camera
                          -> flashlight follows camera.
each frame (flythrough):  path param += dt -> camera lerp/turn -> flashlight follows.
```

## File structure

- `examples/labyrinth_fps.py` — everything (generation, build, controller,
  collision, flashlight, flythrough), self-contained like `examples/input_fly.py`.

## Verification

- **Dev-time connectivity check**: flood-fill from the start cell over open cells;
  assert every open cell is reachable (catches a broken generator). Run once during
  implementation (a scratch script or a temporary `assert` in `__main__`).
- **Headless smoke**: run the flythrough to MP4; eyeball a frame — a warm cone on the
  walls, corridors falling to black, wall shadows across the floor.
- **Interactive**: manual (WASD walks, right-drag looks, can't pass through walls,
  beam tracks the view). Not automatable without live input.
- No formal `tests/` entry — this is a demo, not library logic. (Revisit if the maze
  generator is extracted to `manifoldx.maze`.)

## Tunables (one place at the top of the file)

`CELL`, `WALL_H`, `cols/rows`, `seed`, `EYE`, `MOVE_SPEED`, `SENS`, `PLAYER_R`,
spot `intensity/angles/range`, flythrough speed.

## Open questions

- Maze size vs framerate: start 8×8 (17×17 grid ≈ a few hundred wall cubes,
  2 batches); bump if it renders comfortably.
- Flythrough path: DFS traversal (explores everything, longer) vs start→exit
  shortest path (shorter, more directed). Default DFS traversal; trivial to switch.
