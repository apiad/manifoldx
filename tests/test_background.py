import time
import manifoldx as mx


def _proc_square(x):        # top-level so it is picklable for the process worker
    return x * x


def test_background_runs_off_thread_and_delivers():
    engine = mx.Engine("bg-test")

    @engine.background
    def work(a, b):
        time.sleep(0.05)
        return a + b

    task = work(2, 3)
    assert task.wait() == 5
    assert task.ready and task.result == 5


def test_on_ready_fires_on_drain():
    engine = mx.Engine("bg-test2")

    @engine.background
    def work(x):
        return x * 10

    got = []
    work(4).on_ready(lambda r: got.append(r))
    for _ in range(200):
        engine._drain_tasks()
        if got:
            break
        time.sleep(0.005)
    assert got == [40]


def test_submit_process_runs_in_separate_process():
    engine = mx.Engine("proc-test")
    task = engine.submit_process(_proc_square, 6)
    assert task.wait() == 36
    assert task.ready and task.result == 36
