"""Every registered benchmark and target satisfies its Protocol."""

from __future__ import annotations

from rehearse.eval.benchmarks import BENCHMARKS, get_benchmark
from rehearse.eval.protocols import Benchmark, Target
from rehearse.eval.targets import TARGETS, get_target


def test_every_benchmark_satisfies_protocol():
    for name in BENCHMARKS:
        bench = get_benchmark(name)
        assert isinstance(bench, Benchmark), f"{name} is not a Benchmark"
        assert bench.name == name
        assert isinstance(bench.version, str) and bench.version
        assert bench.preferred_target in bench.supported_targets


def test_every_target_satisfies_protocol():
    for name in TARGETS:
        target = get_target(name, model_slots={})
        assert isinstance(target, Target), f"{name} is not a Target"
        assert target.name == name
        assert isinstance(target.version, str) and target.version


def test_unknown_benchmark_raises():
    import pytest

    with pytest.raises(KeyError):
        get_benchmark("does-not-exist")


def test_unknown_target_raises():
    import pytest

    with pytest.raises(KeyError):
        get_target("does-not-exist", {})
