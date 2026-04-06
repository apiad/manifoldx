# Camera API Design

## Problem

The `Camera` class currently only has `get_view_matrix()` and `get_projection_matrix()`. It lacks the basic navigation methods needed for a game engine: positioning, orbiting, panning, zooming, and fitting content in view.

## Solution

Design a comprehensive `Camera` API that supports:

1. **Direct positioning** — set position/target explicitly
2. **Orbit controls** — rotate camera around a target point (arcball-style)
3. **Pan** — translate camera in screen-space (local X/Y axes)
4. **Zoom/Dolly** — move toward/away from target
5. **Fit/Frame** — automatically position camera to frame content

## Design

### Current State

```python
class Camera:
    def __init__(self, position=(0, 10, 20), target=(0, 0, 0), fov=60, up=(0, 1, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.fov = fov
        self.up = np.array(up, dtype=np.float32)
    
    def get_view_matrix(self): ...
    def get_projection_matrix(self, aspect_ratio, near=0.1, far=100.0): ...
```

### Proposed API

```python
class Camera:
    def __init__(self, position=(0, 10, 20), target=(0, 0, 0), fov=60, up=(0, 1, 0)):
        ...
    
    # === Matrix getters ===
    def get_view_matrix(self): ...
    def get_projection_matrix(self, aspect_ratio, near=0.1, far=100.0): ...
    
    # === Convenience matrices ===
    def get_view_projection_matrix(self, aspect_ratio): ...
    def get_forward(self) -> np.ndarray: ...
    def get_right(self) -> np.ndarray: ...
    def get_up(self) -> np.ndarray: ...
    
    # === Direct positioning ===
    def move_to(self, position):
        """Set camera world position."""
    
    def move_by(self, delta):
        """Move camera by delta in world space."""
    
    def look_at(self, target):
        """Set the point the camera looks at."""
    
    def set_pose(self, position, target):
        """Set both position and target at once."""
    
    # === Orbit controls (rotate around target) ===
    def orbit(self, d_azimuth=0, d_elevation=0):
        """Rotate camera around target by delta angles.
        
        Args:
            d_azimuth: Delta azimuth angle in degrees (horizontal rotation)
            d_elevation: Delta elevation angle in degrees (vertical rotation)
        
        Elevation clamped to avoid gimbal lock at poles (±89°).
        """
    
    def get_azimuth_elevation(self) -> tuple[float, float]:
        """Get current azimuth and elevation angles in degrees."""
    
    # === Pan (screen-space translation) ===
    def pan(self, dx, dy, relative_to_viewport=True):
        """Pan the camera perpendicular to view direction.
        
        Args:
            dx: Horizontal displacement
            dy: Vertical displacement
            relative_to_viewport: If True, dx/dy are in viewport units (0-1).
                                  If False, dx/dy are in world units.
        """
    
    # === Zoom / Dolly ===
    def zoom(self, factor):
        """Zoom by multiplying distance to target by factor.
        
        Args:
            factor: Multiplier for distance. >1 zooms in, <1 zooms out.
        
        Example: zoom(2.0) halves the distance to target (zooms in).
        """
    
    def dolly(self, distance):
        """Move camera toward/away from target by distance.
        
        Args:
            distance: World units to move toward (positive) or away (negative).
        """
    
    def get_distance(self) -> float:
        """Get current distance from camera to target."""
    
    # === Fit / Frame ===
    def fit(self, radius, center=(0, 0, 0), margin=0.8, azimuth=45, elevation=30):
        """Position camera to frame a sphere.
        
        Args:
            radius: Radius of sphere to fit in view
            center: Center of sphere in world space
            margin: Fraction of viewport height sphere should occupy (0.8 = 80%)
            azimuth: Camera azimuth angle in degrees
            elevation: Camera elevation angle in degrees
        """
    
    def fit_bounds(self, center, extent, margin=0.8, azimuth=45, elevation=30):
        """Position camera to frame an axis-aligned bounding box.
        
        Args:
            center: Center of bounding box
            extent: Half-extents (half-widths) of bounding box, or scalar for cube
            margin: Fraction of viewport to leave as padding
            azimuth: Camera azimuth angle in degrees
            elevation: Camera elevation angle in degrees
        
        extent can be:
          - float: cube with given half-extent
          - tuple(n,): uniform box (nx, ny, nz) all equal
          - tuple(3,): non-uniform box (hx, hy, hz)
        """
```

## Internal State

To support orbit controls, the camera needs to track its spherical coordinates:

```python
# New internal state
self._azimuth = ...      # degrees, 0 = looking along -Z, 90 = looking along +X
self._elevation = ...    # degrees, 0 = horizon, 90 = straight up
self._distance = ...     # distance from position to target
```

When `position` or `target` changes, recalculate spherical coords.
When spherical coords change, recalculate position from target + spherical.

## Implementation Phases

### Phase 1: Helper matrices and properties
- `get_forward()`, `get_right()`, `get_up()`
- `get_view_projection_matrix()` — convenience method used by renderer
- Recompute spherical coords from `position`/`target` when either changes

### Phase 2: Direct positioning
- `move_to(position)`
- `move_by(delta)` — useful for keyboard-driven camera movement
- `look_at(target)`
- `set_pose(position, target)` — atomic set, avoids intermediate state

### Phase 3: Orbit controls
- Store `_azimuth`, `_elevation`, `_distance` internally
- `orbit(d_azimuth, d_elevation)` — clamped elevation
- `get_azimuth_elevation()`
- Properties: `azimuth`, `elevation` (optional, for systems to read)

### Phase 4: Pan
- `pan(dx, dy)` — moves target in camera's local X/Y plane
- `relative_to_viewport=True` means dx/dy are normalized (0-1), scale by distance

### Phase 5: Zoom / Dolly
- `zoom(factor)` — multiply distance
- `dolly(distance)` — add to distance
- `get_distance()`

### Phase 6: Fit / Frame
- `fit(radius, center, ...)` — position camera to frame a sphere
- `fit_bounds(center, extent, ...)` — position camera to frame a bounding box
- Both should optionally set azimuth/elevation, or use current angles

## Backward Compatibility

All existing code works unchanged:
- `position` property still exists, but setting it updates spherical coords
- `target` property still exists, but setting it updates spherical coords
- `get_view_matrix()` unchanged
- `get_projection_matrix()` unchanged

## Renderer Integration

The renderer calls:
```python
vp = camera.get_view_projection_matrix(aspect)  # New convenience method
```

## Success Criteria

1. `Camera` has all methods listed above
2. `camera.fit()` works correctly in `cubes.py` — positions camera to see all cubes
3. `camera.orbit()`, `camera.pan()`, `camera.zoom()` work for orbit controls (mouse-driven)
4. All existing examples (`cube.py`, `cubes.py`) continue to work
5. Tests pass

## Risks

1. **Gimbal lock at poles**: Elevation clamped to ±89° to prevent degenerate behavior
2. **Orbit vs direct position conflict**: If user sets `camera.position` directly after using orbit, spherical coords become inconsistent. Solution: mark `_orbit_used = True` and recalc spherical from position/target on next orbit call.
3. **Zoom edge cases**: Minimum distance clamp to avoid camera entering target
