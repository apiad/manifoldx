# Add Global Elapsed Time to Engine

## Problem
Need `engine.elapsed` to track total time since engine started, enabling time-based animations and logic.

## Solution
Track cumulative elapsed time in Engine and expose it as a public property.

## Phases

1. **Add initialization** - In `__init__`, add `_start_time` and `elapsed` attributes
2. **Set start time** - In `run()`, initialize both `_start_time` and `_last_time` 
3. **Update elapsed** - In `_compute_dt()`, update `elapsed` each frame from start time

## Changes

### File: src/manifoldx/engine.py

1. **Line ~53** (`__init__`):
   - Add `self._start_time: float = None`
   - Add `self.elapsed: float = 0.0`

2. **Line ~312** (`run()`):
   - Initialize `self._start_time = perf_counter_ns()`
   - Initialize `self.elapsed = 0.0`

3. **Lines ~125-132** (`_compute_dt()`):
   - On first frame, set `_start_time`
   - Each frame: `self.elapsed = (current_time - self._start_time) / 1_000_000_000`

## Risks
- None - simple time tracking

## Success Criteria
- `engine.elapsed` returns total seconds since engine started
- Works correctly with both fixed timestep and wall-clock mode