"""End-to-end runner test on the noop eval + echo environment.

Exercises: example loading, subprocess executor, scoring, artifact writes
(run.json, results.jsonl, summary.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from rehearse.eval.runner import RunConfig, execute_run


async def test_noop_run_produces_full_artifact_bundle(tmp_path: Path):
    config = RunConfig(
        eval_name="noop",
        environment="echo",
        concurrency=2,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config)

    assert outcome.n_examples == 2
    assert outcome.n_ok == 2
    assert outcome.n_error == 0
    assert outcome.aggregate_scores["noop_score"] == 1.0

    run_json = json.loads((outcome.run_dir / "run.json").read_text())
    assert run_json["id"] == outcome.run_id
    assert run_json["pipeline_version"].startswith("noop@")
    assert len(run_json["example_ids"]) == 2

    lines = (outcome.run_dir / "results.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert row["dimension"] == "noop_score"
        assert row["value"] == 1.0

    summary = (outcome.run_dir / "summary.md").read_text()
    assert "noop" in summary
    assert "ok=2" in summary
    assert "noop_score" in summary


async def test_unsupported_environment_rejected(tmp_path: Path):
    import pytest

    config = RunConfig(eval_name="noop", environment="raw-llm", runs_root=tmp_path)
    with pytest.raises(ValueError, match="does not support environment"):
        await execute_run(config)


async def test_repetitions_runs_each_example_n_times_with_distinct_seeds(
    tmp_path: Path,
):
    """With reps=3 and 2 examples, the runner must produce 6 rollouts; per
    rep the rng_seed varies so downstream environments can produce
    differentiated outputs."""
    from rehearse.eval.suites import get_eval
    from rehearse.eval.harness.executor import InProcessExecutor
    from rehearse.eval.protocols import RolloutResult

    seen_seeds: list[int] = []
    real_executor = InProcessExecutor()

    class _SeedTrackingExecutor:
        async def submit(self, **kwargs) -> RolloutResult:
            seen_seeds.append(kwargs["rng_seed"])
            return await real_executor.submit(**kwargs)

    config = RunConfig(
        eval_name="noop",
        environment="echo",
        repetitions=3,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config, executor=_SeedTrackingExecutor())

    eval_spec = get_eval("noop")
    n_examples = len(list(eval_spec.load()))
    assert outcome.n_examples == n_examples  # examples, not rollouts
    assert len(seen_seeds) == n_examples * 3
    assert len(set(seen_seeds)) == n_examples * 3  # all distinct

    # results.jsonl gets one row per (rollout, scorer); no meta-scorer attached.
    lines = (outcome.run_dir / "results.jsonl").read_text().splitlines()
    assert len(lines) == n_examples * 3


async def test_meta_scoring_plan_invoked_with_grouped_rollouts(tmp_path: Path):
    """An eval that exposes `meta_scoring_plan()` gets its meta-scorer called
    once per example with all repetitions of that example grouped."""
    from rehearse.eval.suites import EVALS
    from rehearse.eval.suites.noop import NoopEval
    from rehearse.eval.scorers.stability import StabilityScorer

    class _NoopWithStability(NoopEval):
        def meta_scoring_plan(self):  # noqa: D401
            return [StabilityScorer()]

    EVALS["noop-stability"] = _NoopWithStability
    _NoopWithStability.supported_environments = frozenset({"echo"})
    try:
        config = RunConfig(
            eval_name="noop-stability",
            environment="echo",
            repetitions=3,
            runs_root=tmp_path,
        )
        outcome = await execute_run(config)

        # Stability of identical 1.0 noop_scores → 1.0 per dimension.
        assert outcome.aggregate_scores["stability.noop_score"] == 1.0
    finally:
        del EVALS["noop-stability"]


async def test_repetitions_must_be_at_least_one():
    import pytest

    with pytest.raises(ValueError, match="repetitions"):
        RunConfig(eval_name="noop", repetitions=0)
