# demos/ — end-to-end showcases

Game-like / sim-like experiences that combine **many** systems at once
(rendering, physics, procedural generation, streaming, ...). These exist to
**show off** the engine — expect more code, and sometimes multiple files or
bundled assets per demo.

For minimal, one-feature examples, see [`../examples/`](../examples).

Current demos:

| Demo | What it shows |
| --- | --- |
| `terrain_stream.py` | Infinite procedural terrain flyby: `fields` + shader fog + patches generated in a worker process (`submit_process`) and streamed into a recycled slot pool |
| `labyrinth_fps.py` | First-person maze crawl with a shadow-casting flashlight (input + spot lights + spot shadows) |
| `sun_earth_moon.py` | Textured Earth/Moon orbiting a sun (textured PBR + orbital motion; bundled NASA imagery) |
| `smoke.py` | Volumetric Perlin-FBM smoke with camera motion (direct volume rendering) |

Run one:

```bash
uv run python demos/<name>.py                                   # interactive
uv run python demos/<name>.py --render --duration 8 --output /tmp/<name>.mp4
```
