"""Compatibility wrapper for the noop eval."""

from __future__ import annotations

from rehearse.eval.evals.noop import NoopEval


class NoopBenchmark(NoopEval):
    supported_targets = NoopEval.supported_environments
    preferred_target = NoopEval.preferred_environment
