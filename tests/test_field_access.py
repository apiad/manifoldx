"""Field reads and writes from systems: q[Comp].field, +=, assignment, deferred vs direct."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.commands import CommandBuffer
from manifoldx.components import Transform
from manifoldx.ecs import ComponentView, EntityStore
from manifoldx.types import Float, Vector3


class _Engine:
    def __init__(self):
        self.commands = CommandBuffer()


def _transform_store(n=6):
    store = EntityStore(max_entities=16)
    Transform.register(store)
    data = np.arange(n * 10, dtype=np.float32).reshape(n, 10)
    data[:, 3:7] = [0, 0, 0, 1]
    store.spawn(n, Transform=data)
    return store, data


# Every other entity, out of order, so slicing by index matters.
IDX = np.array([4, 0, 2])


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("name,cols", [("position", (0, 3)), ("pos", (0, 3)), ("scale", (7, 10))])
def test_transform_vector_field_add(name, cols, deferred):
    store, data = _transform_store()
    engine = _Engine() if deferred else None
    acc = ComponentView(store, ["Transform"], IDX, engine)["Transform"]
    a, b = cols

    field = getattr(acc, name)
    assert field.data.shape == (3, b - a)
    assert np.array_equal(field.data, data[IDX, a:b])

    field += np.array([1, 2, 3], np.float32)
    if deferred:
        assert np.array_equal(store._components["Transform"][:6], data)
        engine.commands.execute(store)

    expected = data.copy()
    expected[IDX, a:b] += [1, 2, 3]
    assert np.array_equal(store._components["Transform"][:6], expected)


@pytest.mark.parametrize("deferred", [False, True])
def test_transform_field_assign_broadcasts(deferred):
    store, data = _transform_store()
    engine = _Engine() if deferred else None
    acc = ComponentView(store, ["Transform"], IDX, engine)["Transform"]

    acc.position = (9, 8, 7)
    if deferred:
        engine.commands.execute(store)

    expected = data.copy()
    expected[IDX, 0:3] = [9, 8, 7]
    assert np.array_equal(store._components["Transform"][:6], expected)


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("op", ["isub", "imul"])
def test_transform_field_isub_imul(op, deferred):
    store, data = _transform_store()
    engine = _Engine() if deferred else None
    acc = ComponentView(store, ["Transform"], IDX, engine)["Transform"]
    other = np.array([[2], [3], [4]], np.float32)

    pos = acc.position
    if op == "isub":
        pos -= other
    else:
        pos *= other
    if deferred:
        engine.commands.execute(store)

    expected = data.copy()
    if op == "isub":
        expected[IDX, 0:3] -= other
    else:
        expected[IDX, 0:3] *= other
    assert np.array_equal(store._components["Transform"][:6], expected)


@pytest.mark.parametrize("deferred", [False, True])
def test_rotation_iadd_composes_quaternions(deferred):
    store, data = _transform_store()
    engine = _Engine() if deferred else None
    acc = ComponentView(store, ["Transform"], IDX, engine)["Transform"]

    half = np.sqrt(0.5).astype(np.float32)
    quarter_z = np.array([0, 0, half, half], np.float32)
    acc.rotation += quarter_z
    acc.rot += quarter_z  # read sees the first write only when direct
    if deferred:
        engine.commands.execute(store)
        expected_rot = quarter_z  # both commands composed onto identity
    else:
        expected_rot = np.array([0, 0, 1, 0], np.float32)  # half turn

    out = store._components["Transform"][:6]
    assert np.allclose(out[IDX, 3:7], expected_rot, atol=1e-6)
    others = np.setdiff1d(np.arange(6), IDX)
    assert np.array_equal(out[others], data[others])
    assert np.array_equal(out[IDX][:, [0, 1, 2, 7, 8, 9]], data[IDX][:, [0, 1, 2, 7, 8, 9]])


def _custom_engine():
    engine = mx.Engine("FieldAccessTest")

    @engine.component
    class Particle:
        velocity: Vector3
        life: Float
        heat: Float

    return engine, Particle


@pytest.mark.parametrize("deferred", [False, True])
def test_custom_component_vector_and_scalar_fields(deferred):
    engine, Particle = _custom_engine()
    n = 5
    vel = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    life = np.arange(n, dtype=np.float32) + 10
    engine.spawn(Particle(velocity=vel, life=life, heat=0.5), n=n)
    store = engine.store
    before = store._components["Particle"][:n].copy()

    idx = np.array([3, 0, 4])
    acc = ComponentView(store, ["Particle"], idx, engine if deferred else None)["Particle"]

    assert acc.velocity.data.shape == (3, 3)
    assert np.array_equal(acc.velocity.data, before[idx, 0:3])
    assert acc.life.data.shape == (3, 1)
    assert np.array_equal(acc.life.data, before[idx, 3:4])

    acc.velocity += np.float32(1)
    acc.life -= np.float32(0.5)
    acc.heat = 2.0
    if deferred:
        assert np.array_equal(store._components["Particle"][:n], before)
        engine.commands.execute(store)

    after = store._components["Particle"][:n]
    expected = before.copy()
    expected[idx, 0:3] += 1
    expected[idx, 3] -= 0.5
    expected[idx, 4] = 2.0
    assert np.array_equal(after, expected)


def test_builtin_scalar_field_value():
    store = EntityStore(max_entities=8)
    store.register_component("ScalarValue", np.dtype("f4"), (1,))
    store.spawn(4, ScalarValue=np.arange(4, dtype=np.float32).reshape(4, 1))
    idx = np.array([3, 1])
    acc = ComponentView(store, ["ScalarValue"], idx)["ScalarValue"]

    assert np.array_equal(acc.value.data, [[3], [1]])
    acc.value += np.float32(10)
    assert np.array_equal(store._components["ScalarValue"][:4, 0], [0, 11, 2, 13])


def test_system_writes_are_deferred_until_flush():
    engine = mx.Engine("FieldAccessTest")
    n = 4
    engine.spawn(Transform(pos=np.zeros((n, 3), np.float32)), n=n)
    seen = []

    @engine.system
    def drift(q: mx.Query[Transform], dt: float):
        q[Transform].position += np.float32(1)
        seen.append(q[Transform].position.data.copy())

    engine.systems.run_all(engine, 1 / 30)
    assert np.array_equal(seen[0], np.zeros((n, 3)))
    engine.commands.execute(engine.store)
    engine.commands.clear()
    assert np.array_equal(engine.store._components["Transform"][:n, 0:3], np.ones((n, 3)))


def test_unknown_field_raises_attribute_error():
    store, _ = _transform_store()
    acc = ComponentView(store, ["Transform"], IDX)["Transform"]
    with pytest.raises(AttributeError):
        acc.nope


def test_field_of_unregistered_component_raises():
    store, _ = _transform_store()
    acc = ComponentView(store, ["Ghost"], IDX)["Ghost"]
    with pytest.raises(ValueError, match="not registered"):
        acc.position
