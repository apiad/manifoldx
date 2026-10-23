import itertools

import pytest

from manifoldx.workers import ProcessStream


def drain(stream):
    items = []
    while not stream.done:
        item = stream.take(timeout=30)
        if item is not None:
            items.append(item)
    return items


def test_a_stream_hands_back_each_item_in_order():
    assert drain(ProcessStream(range, 4)) == [0, 1, 2, 3]


def test_take_never_waits_by_default():
    stream = ProcessStream(range, 2)
    first = stream.take()  # the child is still starting: nothing has arrived
    assert first is None
    assert drain(stream) == [0, 1]
    assert stream.take() is None


def test_a_child_whose_parent_stops_listening_exits_quietly():
    stream = ProcessStream(itertools.count)  # endless: only the closed pipe stops it
    assert stream.take(timeout=30) == 0
    stream._proc.stdout.close()
    assert stream._proc.wait(timeout=30) == 0


def test_a_failing_worker_raises_its_traceback_in_the_parent():
    stream = ProcessStream(int, "not a number")
    with pytest.raises(RuntimeError, match="ValueError"):
        drain(stream)
    assert stream.done
