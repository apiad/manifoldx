# examples/ — focused feature examples

Each file demonstrates **one** engine feature with the least code possible.
These are **documentation**: read one to learn how a single capability works.

Naming: files are named after the feature (`shadows.py`, `fields.py`,
`booleans.py`), not `*_demo.py`.

For multi-system, game-like / sim-like showcases, see [`../demos/`](../demos).

Run one:

```bash
uv run python examples/<name>.py                                   # interactive
uv run python examples/<name>.py --render --duration 3 --output /tmp/<name>.mp4
```
