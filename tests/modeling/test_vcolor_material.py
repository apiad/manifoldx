from manifoldx.resources import StandardMaterial


def test_vertex_colors_subtype():
    assert StandardMaterial(color="#ffffff", vertex_colors=True).pipeline_subtype == "vcolor"
    assert StandardMaterial(color="#ffffff").pipeline_subtype is None


def test_vcolor_shader_variant_wires_vertex_color():
    src = StandardMaterial._compile(vertex_colors=True)
    assert "@location(2) color:" in src          # vertex color attribute
    assert "out.vcolor = in.color;" in src        # passed to fragment
    assert "vertex_albedo" in src                 # tints the albedo
    # sun path must use the vertex color, not the flat material albedo
    assert "calculateSun(N, V, F0, vertex_albedo" in src


def test_scalar_shader_unchanged_has_no_vcolor():
    assert "in.vcolor" not in StandardMaterial._compile()
