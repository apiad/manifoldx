"""Every benchmark case runs at its smoke size, so the suites keep working as the code changes.

GPU cases skip on a machine without an adapter (tests/conftest.py)."""

import pytest

from benchmarks import harness
from benchmarks.__main__ import SUITES, load, run

CASES = load(SUITES)


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(harness, "REPEAT", 1)


def test_every_suite_has_cases():
    assert {c.suite for c in CASES} == set(SUITES)


@pytest.mark.parametrize("c", CASES, ids=lambda c: f"{c.suite}.{c.name}")
def test_case_runs_at_smoke_size(c):
    out = c.fn(c.smoke)
    ms = out["ms"] if isinstance(out, dict) else out
    assert ms > 0


def test_run_records_a_failing_case_as_an_error():
    def broken(n):
        raise AttributeError("no such API")

    c = harness.Case("x", "broken", broken, [1], 1)
    (r,) = run([c], smoke=True)
    assert r.ms is None and "no such API" in r.error


def test_compare_reports_speedup_and_missing_cases():
    from benchmarks.__main__ import compare

    base = {"results": [{"suite": "s", "case": "a", "n": 1, "ms": 10.0, "extra": {}}]}
    head = {"results": [{"suite": "s", "case": "a", "n": 1, "ms": 2.0, "extra": {}},
                        {"suite": "s", "case": "new", "n": 1, "ms": 1.0, "extra": {}}]}
    table = compare(base, head, "main")
    assert "| s.a | 1 | 10.0 | 2.0 | 5.0× |" in table
    assert "| s.new | 1 | n/a | 1.0 | – |" in table
