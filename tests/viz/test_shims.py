"""Unit tests for the Plan 4 declarative shim layer."""
import numpy as np
import pytest


def test_chart_construction_no_marks():
    """A Chart can be constructed with no marks; build() returns an Engine."""
    import manifoldx.viz as mxv

    chart = mxv.Chart()
    engine = chart.engine
    assert engine is not None


def test_chart_addition_returns_chart_with_combined_marks():
    """`mark + mark` returns a Chart whose mark list is the concatenation."""
    import manifoldx.viz as mxv
    from manifoldx.viz.shims import Mark

    class _DummyMark(Mark):
        def apply(self, engine):
            pass

    a = _DummyMark()
    b = _DummyMark()
    chart = a + b
    assert isinstance(chart, mxv.Chart)
    assert len(chart.marks) == 2
    assert chart.marks[0] is a and chart.marks[1] is b


def test_chart_addition_associative():
    """Three marks combined in different groupings yield the same mark order."""
    import manifoldx.viz as mxv
    from manifoldx.viz.shims import Mark

    class _DummyMark(Mark):
        def __init__(self, tag):
            self.tag = tag

        def apply(self, engine):
            pass

    a, b, c = _DummyMark("a"), _DummyMark("b"), _DummyMark("c")
    left = (a + b) + c
    right = a + (b + c)
    assert [m.tag for m in left.marks] == ["a", "b", "c"]
    assert [m.tag for m in right.marks] == ["a", "b", "c"]


def test_color_channel_wraps_bare_array():
    """A bare numpy array passed to mxv.color is held as the data field."""
    import manifoldx.viz as mxv

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ch = mxv.color(arr, cmap="viridis", domain=(0, 5))
    assert ch.data is arr
    assert ch.cmap == "viridis"
    assert ch.domain == (0, 5)


def test_color_channel_defaults():
    """Defaults: cmap='viridis', domain inferred from data."""
    import manifoldx.viz as mxv

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ch = mxv.color(arr)
    assert ch.cmap == "viridis"
    assert ch.domain is None  # None means "infer at build time"


def test_points_mark_spawns_entities_on_build():
    """mxv.points(...) materializes a sprite point cloud when the chart builds."""
    import manifoldx.viz as mxv

    n = 10
    positions = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    speeds = np.random.uniform(0, 5, n).astype(np.float32)
    radii = np.full(n, 0.05, dtype=np.float32)

    chart = mxv.points(positions=positions, color=speeds, size=radii)
    engine = chart.engine

    # The Plan-1 components for the sprite path must be registered.
    assert "PointCloud" in engine.store._components
    assert "ScalarValue" in engine.store._components
    assert "Radius" in engine.store._components
    # n alive entities.
    assert int(np.sum(engine.store._alive)) == n


def test_points_mark_with_explicit_color_channel():
    """A wrapped color channel sets cmap and domain on the underlying material."""
    import manifoldx.viz as mxv
    from manifoldx.viz import ColormapMaterial

    n = 8
    positions = np.zeros((n, 3), dtype=np.float32)
    speeds = np.full(n, 2.5, dtype=np.float32)

    chart = mxv.points(
        positions=positions,
        color=mxv.color(speeds, cmap="inferno", domain=(0.0, 5.0)),
    )
    engine = chart.engine

    # Find the ColormapMaterial registered by the points mark.
    found = [
        m for m in engine._material_registry._materials.values()
        if isinstance(m, ColormapMaterial)
    ]
    assert len(found) == 1
    mat = found[0]
    assert mat.cmap == "inferno"
    assert mat.vmin == 0.0
    assert mat.vmax == 5.0


def test_chart_simulate_decorator_registers_callback():
    """@chart.simulate appends to simulate_callbacks before build."""
    import manifoldx.viz as mxv

    chart = mxv.points(
        positions=np.zeros((3, 3), dtype=np.float32),
        color=np.zeros(3, dtype=np.float32),
        size=np.full(3, 0.05, dtype=np.float32),
    )

    @chart.simulate
    def step(dt):
        pass

    assert step in chart.simulate_callbacks
