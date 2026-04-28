"""Compatibility benchmark registry.

New code should import from `rehearse.eval.evals`.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.benchmarks.mme_emotion import MMEEmotionBenchmark
from rehearse.eval.benchmarks.noop import NoopBenchmark
from rehearse.eval.protocols import Benchmark

BENCHMARKS: dict[str, Callable[[], Benchmark]] = {
    "noop": NoopBenchmark,
    "mme-emotion": MMEEmotionBenchmark,
}


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}. registered: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]()


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)
