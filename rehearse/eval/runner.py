"""Eval orchestrator.

Resolves an eval + environment by name, schedules rollouts through the executor
with bounded concurrency, runs the eval's scoring plan against each
rollout, and writes:

  evals/runs/{run_id}/run.json          # EvalRun manifest
  evals/runs/{run_id}/results.jsonl     # one RubricScore per line
  evals/runs/{run_id}/summary.md        # human-facing aggregate
  evals/runs/{run_id}/sessions/{id}/    # full-session bundles, if produced
  evals/runs/{run_id}/failures/{id}/    # error details for failed rollouts

The runner imports nothing from outside `rehearse/eval/` plus `rehearse/types.py`.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from rehearse.eval.environments import get_environment
from rehearse.eval.evals import get_eval
from rehearse.eval.executors import LocalSubprocessExecutor
from rehearse.eval.protocols import BenchmarkExample, Executor, RolloutResult
from rehearse.types import EvalRun, RubricScore


class RunConfig:
    def __init__(
        self,
        eval_name: str | None = None,
        environment: str | None = None,
        *,
        benchmark: str | None = None,
        target: str | None = None,
        limit: int | None = None,
        concurrency: int = 4,
        seed: int = 0,
        model_slots: dict[str, str] | None = None,
        tag: str | None = None,
        runs_root: Path = Path("evals/runs"),
    ) -> None:
        self.eval_name = eval_name or benchmark
        self.environment = environment or target
        self.limit = limit
        self.concurrency = concurrency
        self.seed = seed
        self.model_slots = model_slots
        self.tag = tag
        self.runs_root = runs_root
        if not self.eval_name:
            raise ValueError("RunConfig requires eval_name (or deprecated benchmark)")

    eval_name: str | None
    environment: str | None
    limit: int | None = None
    concurrency: int = 4
    seed: int = 0
    model_slots: dict[str, str] | None = None
    tag: str | None = None
    runs_root: Path = Path("evals/runs")


@dataclass
class RunOutcome:
    run_id: str
    run_dir: Path
    n_examples: int
    n_ok: int
    n_error: int
    n_timeout: int
    aggregate_scores: dict[str, float]


async def execute_run(config: RunConfig, executor: Executor | None = None) -> RunOutcome:
    eval_spec = get_eval(config.eval_name or "")
    environment_name = config.environment or eval_spec.preferred_environment
    if environment_name not in eval_spec.supported_environments:
        raise ValueError(
            f"eval {eval_spec.name!r} does not support environment {environment_name!r}; "
            f"supported: {sorted(eval_spec.supported_environments)}"
        )

    model_slots = config.model_slots or {}
    environment = get_environment(environment_name, model_slots)
    executor = executor or LocalSubprocessExecutor()

    examples = list(eval_spec.load())
    if config.limit is not None:
        examples = examples[: config.limit]

    run_id = _new_run_id()
    run_dir = config.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sessions").mkdir(exist_ok=True)
    (run_dir / "failures").mkdir(exist_ok=True)

    started_at = datetime.now()
    timeout_s = eval_spec.rollout_timeout_s()
    semaphore = asyncio.Semaphore(config.concurrency)

    async def run_one(idx: int, ex: BenchmarkExample) -> RolloutResult:
        async with semaphore:
            return await executor.submit(
                target_name=environment.name,
                target_version=environment.version,
                model_slots=model_slots,
                example=ex,
                run_dir=run_dir / "sessions" / ex.id,
                timeout_s=timeout_s,
                rng_seed=config.seed + idx,
            )

    rollouts: list[RolloutResult] = await asyncio.gather(
        *(run_one(i, ex) for i, ex in enumerate(examples))
    )

    scorers = eval_spec.scoring_plan()
    all_scores: list[RubricScore] = []
    for ex, ro in zip(examples, rollouts, strict=True):
        if ro.status != "ok":
            (run_dir / "failures" / ex.id).mkdir(parents=True, exist_ok=True)
            (run_dir / "failures" / ex.id / "error.txt").write_text(ro.error or "")
        for scorer in scorers:
            try:
                scores = await scorer.score(ex, ro, run_id=run_id)
            except Exception as exc:
                scores = [
                    RubricScore(
                        run_id=run_id,
                        example_id=ex.id,
                        dimension=scorer.dimension,
                        value=0.0,
                        scorer="deterministic",
                        rationale=f"scorer {scorer.name} crashed: {exc}",
                    )
                ]
            all_scores.extend(scores)

    completed_at = datetime.now()

    results_path = run_dir / "results.jsonl"
    with results_path.open("w") as f:
        for s in all_scores:
            f.write(s.model_dump_json() + "\n")

    aggregates = _aggregate(all_scores)
    eval_run = EvalRun(
        id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        example_ids=[ex.id for ex in examples],
        pipeline_version=(
            f"{eval_spec.name}@{eval_spec.version}/"
            f"{environment.name}@{environment.version}"
        ),
        model_slots=model_slots,
        results_path=results_path,
        aggregate_scores=aggregates,  # type: ignore[arg-type]
    )
    (run_dir / "run.json").write_text(eval_run.model_dump_json(indent=2))

    summary = _render_summary(
        eval_name=eval_spec.name,
        environment_name=environment.name,
        run_id=run_id,
        config=config,
        examples=examples,
        rollouts=rollouts,
        aggregates=aggregates,
        started_at=started_at,
        completed_at=completed_at,
    )
    (run_dir / "summary.md").write_text(summary)

    return RunOutcome(
        run_id=run_id,
        run_dir=run_dir,
        n_examples=len(examples),
        n_ok=sum(1 for r in rollouts if r.status == "ok"),
        n_error=sum(1 for r in rollouts if r.status == "error"),
        n_timeout=sum(1 for r in rollouts if r.status == "timeout"),
        aggregate_scores=aggregates,
    )


def _new_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{ts}-{uuid4().hex[:8]}"


def _aggregate(scores: list[RubricScore]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        key = s.dimension if isinstance(s.dimension, str) else s.dimension.value
        grouped[key].append(s.value)
    return {k: statistics.fmean(v) for k, v in grouped.items() if v}


def _render_summary(
    *,
    eval_name: str,
    environment_name: str,
    run_id: str,
    config: RunConfig,
    examples: list[BenchmarkExample],
    rollouts: list[RolloutResult],
    aggregates: dict[str, float],
    started_at: datetime,
    completed_at: datetime,
) -> str:
    duration_s = (completed_at - started_at).total_seconds()
    n_ok = sum(1 for r in rollouts if r.status == "ok")
    n_err = sum(1 for r in rollouts if r.status == "error")
    n_to = sum(1 for r in rollouts if r.status == "timeout")
    lines = [
        f"# Eval run `{run_id}`",
        "",
        f"- Eval: **{eval_name}**",
        f"- Environment: **{environment_name}**",
        f"- Examples: {len(examples)} (ok={n_ok}, error={n_err}, timeout={n_to})",
        f"- Concurrency: {config.concurrency}",
        f"- Seed: {config.seed}",
        f"- Started: {started_at.isoformat(timespec='seconds')}",
        f"- Duration: {duration_s:.1f}s",
        "",
        "## Aggregate scores",
        "",
        "| Dimension | Mean |",
        "|---|---|",
    ]
    for dim, mean in sorted(aggregates.items()):
        lines.append(f"| `{dim}` | {mean:.3f} |")

    if n_err or n_to:
        lines.extend(["", "## Failures", ""])
        for r in rollouts:
            if r.status != "ok":
                snippet = (r.error or "").splitlines()[0] if r.error else ""
                lines.append(f"- `{r.example_id}` ({r.status}): {snippet}")

    return "\n".join(lines) + "\n"
