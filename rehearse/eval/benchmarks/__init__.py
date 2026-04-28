"""Benchmark registry. Add a new benchmark by importing its class and
registering its constructor here."""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.benchmarks.eq_bench import EQBenchBenchmark
from rehearse.eval.benchmarks.noop import NoopBenchmark
from rehearse.eval.protocols import Benchmark

BENCHMARKS: dict[str, Callable[[], Benchmark]] = {
    "noop": NoopBenchmark,
    "eq-bench": EQBenchBenchmark,
}


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}. registered: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]()


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)
