"""modeling.Mesh operators on large meshes. Size n is an icosphere subdivision level:
5, 6 and 7 give 20 480, 81 920 and 327 680 faces."""

from benchmarks.harness import case, measure

LEVELS = [5, 6, 7]


def _sphere(level):
    from manifoldx.modeling import Mesh

    return Mesh.icosphere(subdivisions=level)


def _faces(level):
    return {"faces": 20 * 4**level}


@case("modeling", "adjacency", sizes=LEVELS, smoke=1)
def adjacency(level):
    """Mesh.adjacency() on a fresh mesh (no cache)."""
    from manifoldx.modeling import Mesh

    m = _sphere(level)
    return {"ms": measure(lambda: Mesh(positions=m.positions, faces=m.faces).adjacency()), **_faces(level)}


@case("modeling", "smooth", sizes=LEVELS, smoke=1)
def smooth(level):
    """Mesh.smooth(iterations=1), adjacency already cached."""
    m = _sphere(level)
    m.adjacency()
    return {"ms": measure(lambda: m.smooth(iterations=1)), **_faces(level)}


@case("modeling", "subdivide", sizes=LEVELS, smoke=1)
def subdivide(level):
    """Mesh.subdivide(1), midpoint scheme."""
    m = _sphere(level)
    return {"ms": measure(lambda: m.subdivide(1)), **_faces(level)}


@case("modeling", "decimate", sizes=LEVELS, smoke=1)
def decimate(level):
    """Mesh.decimate(grid=64)."""
    m = _sphere(level)
    return {"ms": measure(lambda: m.decimate(64)), **_faces(level)}


@case("modeling", "recompute_normals", sizes=LEVELS, smoke=1)
def recompute_normals(level):
    """Mesh.recompute_normals()."""
    m = _sphere(level)
    return {"ms": measure(m.recompute_normals), **_faces(level)}
