"""Plan 2 public surface — what users import."""
def test_plan2_public_surface_imports():
    from manifoldx.viz import (
        ColormapMaterial,
        LabelMaterial,
        LabelTextureAtlas,
        PointCloud,
        Radius,
        ScalarValue,
        TextLabel,
    )
    assert LabelMaterial is not None
    assert LabelTextureAtlas is not None
    assert TextLabel is not None


def test_plan2_public_surface_in_all():
    import manifoldx.viz as viz
    expected = {
        "PointCloud", "ScalarValue", "Radius", "TextLabel",
        "ColormapMaterial", "LabelMaterial", "LabelTextureAtlas",
    }
    assert expected.issubset(set(viz.__all__))
