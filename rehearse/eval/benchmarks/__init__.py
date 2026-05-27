"""External benchmark evals — open-source datasets that test foundation model capability.

These are distinct from product evals (in suites/): they measure whether an
underlying model can perform a capability, not whether the rehearse agent behaves well.
"""

from __future__ import annotations
from collections.abc import Callable
from rehearse.eval.benchmarks.mme_emotion import MMEEmotionEval
from rehearse.eval.protocols import Eval

BENCHMARKS: dict[str, Callable[[], Eval]] = {
    "mme-emotion": MMEEmotionEval,
}

def get_benchmark(name: str) -> Eval:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}. registered: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]()

def list_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)
