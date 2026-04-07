---
id: engine-cli-unified-entry-point
created: 2025-04-07
type: plan
status: draft
expires: 2025-04-14
phases:
  - name: Phase 1: Design CLI API
    done: false
  - name: Phase 2: Implement engine.cli() method
    done: false
  - name: Phase 3: Update examples to use cli()
    done: false
  - name: Phase 4: Add tests
    done: false
---

# Plan: Unified CLI Entry Point for Engine

## Context

Currently each example has manual CLI parsing:
```python
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--render":
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60
        engine.render(...)
    else:
        engine.run()
```

This is repetitive and error-prone. We want:
```python
if __name__ == "__main__":
    engine.cli()
```

## Design

### CLI Behavior

| Command | Action |
|---------|--------|
| `python example.py` | `engine.run()` - interactive window |
| `python example.py --render` | `engine.render()` - default 60s video |
| `python example.py --render --fps 60 --duration 120` | Custom params |
| `python example.py --render --output custom.mp4` | Custom filename |

### API

```python
def cli(
    self,
    *,
    fps: int = 30,
    duration: float = 60,
    output: str | None = None,
    quality: str = "high",
) -> None:
```

**Parameters:**
- `fps`: Frames per second (default: 30)
- `duration`: Video duration in seconds (default: 60)
- `output`: Output filename. If None, inferred from `sys.argv[0]` → `example.py` → `example.mp4`
- `quality`: "low", "medium", "high" (default: "high")

### Implementation

```python
def cli(
    self,
    *,
    fps: int = 30,
    duration: float = 60,
    output: str | None = None,
    quality: str = "high",
) -> None:
    """Command-line interface for the engine.
    
    Usage:
        python example.py           # Interactive window
        python example.py --render  # Render 60s video
        python example.py --render --fps 60 --duration 120
        python example.py --render --output custom.mp4
    """
    import argparse
    
    parser = argparse.ArgumentParser(description=self.title)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render to video instead of showing window",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=fps,
        help=f"Frames per second (default: {fps})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=duration,
        help=f"Video duration in seconds (default: {duration})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: <script>.mp4)",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default=quality,
        choices=["low", "medium", "high"],
        help=f"Video quality (default: {quality})",
    )
    
    args = parser.parse_args()
    
    if args.render:
        # Infer output filename from script name
        if args.output is None:
            import sys
            from pathlib import Path
            script_path = Path(sys.argv[0]).resolve()
            output = script_path.with_suffix(".mp4")
        else:
            output = args.output
        
        self.render(
            output=str(output),
            fps=args.fps,
            duration=args.duration,
            quality=args.quality,
        )
    else:
        self.run()
```

### Updated Example Pattern

```python
if __name__ == "__main__":
    # Set up engine, systems, entities...
    engine.cli()
```

## Benefits

1. **Single line** to add CLI support
2. **Consistent behavior** across all examples
3. **Sensible defaults** - just `--render` works
4. **Auto-filename** - no need to specify output
5. **Extensible** - easy to add more CLI options later

## Examples Update

After implementation, update examples to use:
```python
if __name__ == "__main__":
    engine.cli()
```

## Risks

| Risk | Mitigation |
|------|------------|
| argparse conflicts with user args | Only parse known args, use `parse_known_args()` if needed |
| sys.argv[0] unreliable | Fall back to "output.mp4" |

## Files to Modify

- `src/manifoldx/engine.py` - Add `cli()` method
- `examples/*.py` - Replace CLI boilerplate with `engine.cli()`
- `tests/test_engine.py` - Add CLI tests
