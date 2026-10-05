# Characters and previz rendering

**Status:** approved 2026-10-05. Package name: `manifoldx.humanoid`.

Sub-project 1 of the mosaico scene-storytelling effort
(`vault/Atlas/Architecture/2026-10-04-mosaico-scene-storytelling-design.md` in the
workspace). mosaico is the first client; the module is meant to be reusable for
games and simulations, and will grow animation and inverse kinematics later.

## Goal

Let a program pose humanoid figures with anatomical joint angles, give them
measured or stylised proportions, place them in a scene, and get back from one
call a correctly encoded still frame plus a flat per-object id map. Everything
here was prototyped in mosaico's playground on 2026-10-04 (`rig.py`, `scene.py`)
and in mosaico PR #2 (`src/mosaico/figure.py`, the ANSUR II table); this spec
moves it into the engine with tests.

## Scope

In:

1. `manifoldx.humanoid`: skeleton, poses, proportions, posing solver,
   mannequin builder.
2. `Engine.render_frame(...)`: one still frame to a numpy array.
3. The double-gamma fix for sRGB targets.
4. `FlatMaterial` and an id pass.
5. `Camera.project(...)`: world points to pixel coordinates and depth.
6. `examples/characters.py`.

Out (later sub-projects or later versions): inverse kinematics, animation and
skinning, hands with fingers, faces and expressions, clothing, props and
prefabs (they stay in mosaico for now), outline compositing (mosaico does it
from the id map), running in the browser.

## 1. `manifoldx.humanoid`

### Skeleton

A fixed humanoid joint table with 15 joints: `pelvis` (root), `spine`, `neck`,
and per side `shoulder`, `elbow`, `wrist`, `hip`, `knee`, `ankle`, suffixed
`_l` / `_r`. Each joint has a parent, a rest offset from the parent, a side
(+1 left, -1 right, 0 centre) and a flex sign. The rest pose stands with arms
down, facing +Z, Y up. Units are fractions of stature until the figure is
scaled.

Derived points the solver also reports: `head` (centre of the head sphere),
`nose`, `hand_l` / `hand_r` (fingertip ends). mosaico ties ropes to them and
anchors speech bubbles on `head`.

### Poses

A pose maps joint names to angles in degrees:

- `flex`: forward bend. The flex sign makes positive mean the anatomical
  forward bend for every joint (knees bend backward with positive flex, as in
  anatomy).
- `abduct`: outward from the midline, mirrored per side, so one value serves
  both arms.
- `twist`: rotation about the bone, mirrored per side.
- `lean`: sideways tilt for `spine`, `neck`, `pelvis`.

Shorthands apply to both sides: `shoulders`, `elbows`, `wrists`, `hips`,
`knees`, `ankles`. A pose can name a `base` from the library and override
joints:

```python
Pose.parse({"base": "stand", "shoulder_r": {"flex": 70}})
```

Library v1: `stand`, `t`, `seated`, `arms_crossed`, `hands_on_hips`,
`hands_behind`, `lean_rail`, with the angles validated in the playground.

### Proportions

`Proportions.measured(body)` with `body` in `{"male", "female"}` returns the
ANSUR II means as fractions of stature (US Army anthropometric survey, 2012:
4,082 men, 1,986 women; the table in mosaico PR #2). Head height is 0.130 of
stature (Drillis and Contini 1966), which ANSUR II does not measure.

`Proportions.styled(body, preset)` applies a style preset on top: `heroic`,
`disney`, `anime`, `fashion`, `child`, `toddler`, `chibi`. A preset is a small
set of knobs: heads tall, crotch height, shoulder, waist and hip widths in
heads, limb thickness, head width ratio, hand and foot size, and hanging
fingertip reach. Heights between chin and crotch stretch to the new torso,
heights below the crotch to the new legs. `measured` is the identity preset.

The skeleton's rest offsets are derived from the proportions, so a styled
figure has a styled skeleton, not a styled mesh on a fixed skeleton. (The
playground used hand-tuned offsets; deriving them is new work.)

The arm segments are scaled so the T-pose reaches the measured span. Summed
end to end, the ANSUR II segment lengths overshoot the span by about 11%.

### Solver

```python
fig = humanoid.solve(pose, proportions, height=1.45, yaw=55)
fig.joints["wrist_r"]      # world position (3,)
fig.rotations["wrist_r"]   # world quaternion (x, y, z, w)
fig.points["head"]         # derived points
fig.parts                  # [(kind, centre, rotation, size)] mannequin volumes
fig.bounds                 # world AABB
```

Forward kinematics over the joint table. The figure is grounded: its lowest
point sits at y = 0, so a seated pose rests its feet on the floor and its
pelvis at knee height. `yaw` may also be a target point; `facing(target)`
computes it, because mosaico's scenes say `facing: archimedes`.

### Mannequin

```python
ids = humanoid.spawn_mannequin(engine, fig, at=(x, y, z), color="#d1495b")
```

Spawns the parts as rounded volumes: limbs and torso as ellipsoids (unit
`sphere` mesh with a non-uniform `Transform.scale`), joints as spheres, the
head as a sphere with a small darker nose so gaze direction reads in a still
frame. Returns the entity ids so the id pass can colour the whole figure as
one object.

## 2. `Engine.render_frame`

```python
rgb = engine.render_frame(width=1376, height=768, supersample=2)   # (768, 1376, 3) uint8
```

Creates the offscreen canvas, fires `startup`, draws, reads back, downsamples
with an area filter, and returns sRGB-encoded pixels. Today a client has to call
`_init_canvas`, `_event_bus.dispatch_immediate`, `_draw_frame` and
`_render_canvas.draw()` directly; the playground also drew two frames because
the first one was not reliable. Whether the second frame is needed (shadow
map, IBL prefilter) gets found out and documented, not copied.

## 3. Double gamma on sRGB targets

`get_preferred_format` returns `rgba8unorm-srgb` for the offscreen canvas, and
the shader of `StandardMaterial` ends with `color = pow(color, vec3<f32>(1.0 /
2.2))`. The hardware encodes again on write. A red albedo `#cc2222` read back as
(198, 148, 148) instead of about (144, 76, 76).

Fix: apply the manual gamma only when the target format is not sRGB, by
compiling the shader variant per format (the format is already known when the
pipeline is created). Other materials with the same line (`resources.py`
around lines 246 and 414) get the same treatment.

Risk: on-screen canvases may also be sRGB, so every demo may have been tuned
under double gamma. Check the GLFW canvas format; if it is sRGB, the demos get
darker and more saturated after the fix and their light intensities need
re-tuning. That re-tuning is part of this sub-project.

## 4. `FlatMaterial` and the id pass

`BasicMaterial`'s docstring says "Unlit material with flat color", but its
fragment shader applies a fixed light: `brightness = 0.3 + 0.7 * diffuse`. An id
pass built on it carries shading, and the playground had to compare
chromaticity instead of exact colours to find object edges.

`FlatMaterial(color)` outputs its colour unmodified: no light, no tone mapping,
no gamma. `Engine.render_frame(..., pass_="ids", groups=...)` renders every
entity with a `FlatMaterial` whose colour encodes its group index, at full
resolution with no downsampling, and returns an `(H, W)` int32 label image
plus the group table. Groups come from the caller (mosaico: one per actor, one
per named prop, one per room surface). `BasicMaterial`'s docstring gets
corrected.

## 5. `Camera.project`

```python
xy, depth = engine.camera.project(points, width, height)
```

Uses `get_view_projection_matrix(aspect)` and returns pixel coordinates (origin
top-left) and camera-space depth. mosaico uses it to place bubbles, to size
face obstacles and to write "on the left of the frame, in the foreground" in
the prompt. The playground reimplemented the projection by hand; one source of
truth avoids a mismatch with what the GPU drew.

## Testing

Per-op tests, as elsewhere in the repo:

- **Poses:** T-pose fingertips at half the measured span and at shoulder
  height; left and right mirror each other; every segment starts where its
  parent ends; a pose with the wrong joint name fails with that name.
- **Proportions:** `measured` reproduces the ANSUR II table; every preset hits
  its head count; feet on the floor for every preset.
- **Gamma:** render a flat-lit `StandardMaterial` quad of known albedo and
  check the readback within tolerance. This test must fail on current main.
- **Id pass:** every group maps to exactly one label value; no label varies
  across a lit sphere.
- **Projection:** `Camera.project` of a spawned sphere's centre lands on the
  sphere in `render_frame`'s output.
- Render tests gate on `get_offscreen_canvas` and skip without a backend.
- Smoke: `uv run python examples/characters.py --render`, three posed figures
  with different presets.

## Decisions at review

- Package name: `manifoldx.humanoid`.
- Style presets live in manifoldx next to the measured proportions, because
  games need stylised bodies too.
