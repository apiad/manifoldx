"""ECS store and system costs, without rendering."""

import numpy as np

from benchmarks.harness import case, measure

SIZES = [1_000, 10_000, 100_000]


@case("ecs", "spawn", sizes=SIZES, smoke=10)
def spawn(n):
    """EntityStore.spawn(n) of one 3-float component into an empty store (#107)."""
    from manifoldx.ecs import EntityStore

    data = np.zeros((n, 3), np.float32)

    def run():
        store = EntityStore(max_entities=n + 16)
        store.register_component("position", np.float32, (3,))
        store.spawn(n, position=data)

    return measure(run)


@case("ecs", "system_field_add", sizes=SIZES, smoke=10)
def system_field_add(n):
    """One system doing q[Transform].position += c over n entities, plus the command flush."""
    import manifoldx as mx
    from manifoldx.components import Transform

    from benchmarks.harness import headless_engine

    engine = headless_engine(n)
    engine.spawn(Transform(pos=np.zeros((n, 3), np.float32)), n=n)

    @engine.system
    def drift(q: mx.Query[Transform], dt: float):
        q[Transform].position += np.float32(0.01)

    def run():
        engine.systems.run_all(engine, 1 / 30)
        engine.commands.execute(engine.store)
        engine.commands.clear()

    return measure(run)
