"""Offline render must advance `engine.elapsed` on the video clock, not the wall clock.

`render()` computes `dt = 1.0 / fps` under a "use fixed timestep for video
rendering" comment, but the engine only honours a fixed timestep once
`set_fixed_timestep` has flipped `_use_fixed_dt`. Without that, `_compute_dt()`
takes the wall-clock branch and overwrites `self.elapsed` with real elapsed
seconds every frame, so a scripted flythrough is paced by how fast the machine
happens to render rather than by the video timeline — the same script yields a
different animation on a different GPU.
"""

import pytest


def _offscreen_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas

        get_offscreen_canvas(width=width, height=height)
    except Exception as e:  # pragma: no cover - depends on the host GPU
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx

    return mx.Engine("render clock", width=width, height=height)


def test_render_elapsed_follows_the_video_clock(tmp_path):
    engine = _offscreen_engine()
    fps, frames = 10, 12
    seen = []

    @engine.system
    def sample(query, dt):
        seen.append((engine.elapsed, dt))

    engine.render(
        output=str(tmp_path / "clock.mp4"),
        frame_count=frames,
        fps=fps,
        progress=False,
    )

    assert len(seen) == frames
    # Frame i observes the clock after i completed frames.
    for i, (elapsed, _) in enumerate(seen):
        assert elapsed == pytest.approx(i / fps, abs=1e-6), (
            f"frame {i}: elapsed={elapsed:.4f}, expected {i / fps:.4f}"
        )
    # And the timestep handed to systems is the video's, not the host's.
    for _, dt in seen:
        assert dt == pytest.approx(1.0 / fps, abs=1e-6)
