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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from rehearse.eval.benchmarks import get_benchmark
from rehearse.eval.environments import get_environment
from rehearse.eval.suites import EVALS, get_eval
from rehearse.eval.harness.executor import LocalSubprocessExecutor
from rehearse.eval.protocols import BenchmarkExample, Executor, MetaScorer, RolloutResult
from rehearse.eval.scorers.composite import supports_publish
from rehearse.eval.harness.stream import ScoreStreamWriter
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
        repetitions: int = 1,
    ) -> None:
        self.eval_name = eval_name or benchmark
        self.environment = environment or target
        self.limit = limit
        self.concurrency = concurrency
        self.seed = seed
        self.model_slots = model_slots
        self.tag = tag
        self.runs_root = runs_root
        if repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        self.repetitions = repetitions
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
    repetitions: int = 1


@dataclass
class RunOutcome:
    run_id: str
    run_dir: Path
    n_examples: int
    n_ok: int
    n_error: int
    n_timeout: int
    aggregate_scores: dict[str, float]
    eval_name: str = ""
    environment: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    duration_s: float = 0.0
    total_tokens: int = 0


def _resolve_eval(name: str):
    """Try suites first, fall back to benchmarks."""
    if name in EVALS:
        return get_eval(name)
    return get_benchmark(name)


async def execute_run(config: RunConfig, executor: Executor | None = None) -> RunOutcome:
    eval_spec = _resolve_eval(config.eval_name or "")
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

    reps = config.repetitions

    async def run_one(idx: int, rep: int, ex: BenchmarkExample) -> RolloutResult:
        async with semaphore:
            session_subdir = ex.id if reps == 1 else f"{ex.id}/rep-{rep}"
            return await executor.submit(
                target_name=environment.name,
                target_version=environment.version,
                model_slots=model_slots,
                example=ex,
                run_dir=run_dir / "sessions" / session_subdir,
                timeout_s=timeout_s,
                rng_seed=config.seed + idx * reps + rep,
            )

    plan: list[tuple[int, int, BenchmarkExample]] = [
        (i, rep, ex) for i, ex in enumerate(examples) for rep in range(reps)
    ]
    rollouts: list[RolloutResult] = await asyncio.gather(
        *(run_one(i, rep, ex) for i, rep, ex in plan)
    )
    rollouts_by_example: dict[str, list[RolloutResult]] = defaultdict(list)
    for (_, _, ex), ro in zip(plan, rollouts, strict=True):
        rollouts_by_example[ex.id].append(ro)

    scorers = eval_spec.scoring_plan()
    all_scores: list[RubricScore] = []
    per_rollout_scores: list[list[RubricScore]] = []
    with ScoreStreamWriter(run_dir) as stream:
        publish = stream.publish
        for (_, _, ex), ro in zip(plan, rollouts, strict=True):
            if ro.status != "ok":
                failure_dir = run_dir / "failures" / ex.id
                failure_dir.mkdir(parents=True, exist_ok=True)
                (failure_dir / "error.txt").write_text(ro.error or "")
            rollout_scores: list[RubricScore] = []
            for scorer in scorers:
                try:
                    if supports_publish(scorer):
                        scores = await scorer.score(
                            ex, ro, run_id=run_id, publish=publish
                        )
                    else:
                        scores = await scorer.score(ex, ro, run_id=run_id)
                        for s in scores:
                            publish(s)
                except Exception as exc:
                    crash = RubricScore(
                        run_id=run_id,
                        example_id=ex.id,
                        dimension=scorer.dimension,
                        value=0.0,
                        scorer="deterministic",
                        rationale=f"scorer {scorer.name} crashed: {exc}",
                    )
                    publish(crash)
                    scores = [crash]
                rollout_scores.extend(scores)
            per_rollout_scores.append(rollout_scores)
            all_scores.extend(rollout_scores)

    meta_scorers = _meta_scoring_plan(eval_spec)
    if meta_scorers:
        # Group per_rollout_scores by example_id, preserving rollout order.
        scores_by_example: dict[str, list[list[RubricScore]]] = defaultdict(list)
        for (_, _, ex), rscores in zip(plan, per_rollout_scores, strict=True):
            scores_by_example[ex.id].append(rscores)
        for ex in examples:
            ex_rollouts = rollouts_by_example[ex.id]
            ex_per_rollout = scores_by_example[ex.id]
            for meta in meta_scorers:
                try:
                    meta_rows = await meta.score_meta(
                        ex, ex_rollouts, ex_per_rollout, run_id=run_id
                    )
                except Exception as exc:
                    meta_rows = [
                        RubricScore(
                            run_id=run_id,
                            example_id=ex.id,
                            dimension=f"meta.{meta.name}",
                            value=0.0,
                            scorer="deterministic",
                            modality="meta",
                            rationale=f"meta-scorer {meta.name} crashed: {exc}",
                        )
                    ]
                all_scores.extend(meta_rows)

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

    total_tokens = sum(
        (r.token_usage or {}).get("total_tokens", 0) for r in rollouts
    )
    return RunOutcome(
        run_id=run_id,
        run_dir=run_dir,
        n_examples=len(examples),
        n_ok=sum(1 for r in rollouts if r.status == "ok"),
        n_error=sum(1 for r in rollouts if r.status == "error"),
        n_timeout=sum(1 for r in rollouts if r.status == "timeout"),
        aggregate_scores=aggregates,
        eval_name=eval_spec.name,
        environment=environment.name,
        started_at=started_at,
        duration_s=(completed_at - started_at).total_seconds(),
        total_tokens=total_tokens,
    )


def _meta_scoring_plan(eval_spec) -> list[MetaScorer]:
    """Read optional `meta_scoring_plan()` off an Eval; default to empty.

    Kept off the `Eval` Protocol so existing evals stay protocol-conformant
    without a no-op override.
    """
    factory = getattr(eval_spec, "meta_scoring_plan", None)
    if factory is None:
        return []
    return list(factory())


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
    import os

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
        f"- Repetitions: {config.repetitions}",
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

    # Per-rollout timing and token usage (when available).
    durations_s = [r.duration_ms / 1000 for r in rollouts if r.duration_ms]
    if durations_s:
        avg_s = statistics.fmean(durations_s)
        min_s = min(durations_s)
        max_s = max(durations_s)
        lines.extend([
            "",
            "## Runtime & tokens",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Avg rollout time | {avg_s:.1f}s |",
            f"| Min / max rollout time | {min_s:.1f}s / {max_s:.1f}s |",
        ])
        usages = [r.token_usage for r in rollouts if r.token_usage]
        if usages:
            coach_prompt = sum(u.get("coach_prompt_tokens", 0) for u in usages)
            coach_compl = sum(u.get("coach_completion_tokens", 0) for u in usages)
            cust_prompt = sum(u.get("customer_prompt_tokens", 0) for u in usages)
            cust_compl = sum(u.get("customer_completion_tokens", 0) for u in usages)
            total = sum(u.get("total_tokens", 0) for u in usages)
            lines.extend([
                f"| Coach tokens (prompt / completion) | {coach_prompt:,} / {coach_compl:,} |",
                f"| Customer tokens (prompt / completion) | {cust_prompt:,} / {cust_compl:,} |",
                f"| **Total tokens** | **{total:,}** |",
            ])

    if environment_name == "runtime-sandbox":
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        hume_key = os.environ.get("HUME_API_KEY")
        tts_status = "real (post-hoc)" if hume_key else "stubbed (silent WAV)"
        audio_status = "real (Gemini multimodal)" if hume_key else "degraded (audio_missing)"
        coach_key_status = "ANTHROPIC_API_KEY set" if anthropic_key else "ANTHROPIC_API_KEY missing"
        lines.extend([
            "",
            "## Runtime provenance",
            "",
            "```",
            "RuntimeHost          real",
            "IntakeProcessor      real (deterministic)",
            "PersonaCompiler      real (deterministic)",
            f"CoachVoice           real (TextOnlyCoachAdapter, {coach_key_status})",
            f"Hume TTS             {tts_status}",
            f"Audio judges         {audio_status}",
            "```",
        ])

    if n_err or n_to:
        lines.extend(["", "## Failures", ""])
        for r in rollouts:
            if r.status != "ok":
                snippet = (r.error or "").splitlines()[0] if r.error else ""
                lines.append(f"- `{r.example_id}` ({r.status}): {snippet}")

    return "\n".join(lines) + "\n"
