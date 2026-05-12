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
import json
import resource
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from rehearse.eval.environments import get_environment
from rehearse.eval.evals import get_eval
from rehearse.eval.executors import LocalSubprocessExecutor
from rehearse.eval.protocols import (
    BenchmarkExample,
    Environment,
    Executor,
    MetaScorer,
    RolloutResult,
    Scorer,
)
from rehearse.eval.score_stream import ScoreStreamWriter
from rehearse.eval.scorers.composite import supports_publish
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
        scheduler: Literal["sync", "async"] = "sync",
        rollout_workers: int | None = None,
        scoring_workers: int = 1,
        report_interval_s: float = 10.0,
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
        if scheduler not in ("sync", "async"):
            raise ValueError("scheduler must be 'sync' or 'async'")
        if rollout_workers is not None and rollout_workers < 1:
            raise ValueError("rollout_workers must be >= 1")
        if scoring_workers < 1:
            raise ValueError("scoring_workers must be >= 1")
        self.scheduler = scheduler
        self.rollout_workers = rollout_workers or concurrency
        self.scoring_workers = scoring_workers
        self.report_interval_s = report_interval_s
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
    scheduler: Literal["sync", "async"] = "sync"
    rollout_workers: int | None = None
    scoring_workers: int = 1
    report_interval_s: float = 10.0
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


@dataclass(frozen=True)
class RolloutJob:
    idx: int
    rep: int
    example: BenchmarkExample
    run_dir: Path
    rng_seed: int


@dataclass(frozen=True)
class RolloutEnvelope:
    job: RolloutJob
    rollout: RolloutResult
    started_at: datetime
    completed_at: datetime
    worker_id: int


@dataclass(frozen=True)
class ScoredEnvelope:
    rollout_envelope: RolloutEnvelope
    scores: list[RubricScore]
    started_at: datetime
    completed_at: datetime
    worker_id: int


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

    reps = config.repetitions
    jobs: list[RolloutJob] = [
        RolloutJob(
            idx=i,
            rep=rep,
            example=ex,
            run_dir=run_dir
            / "sessions"
            / (ex.id if reps == 1 else f"{ex.id}/rep-{rep}"),
            rng_seed=config.seed + i * reps + rep,
        )
        for i, ex in enumerate(examples)
        for rep in range(reps)
    ]

    scorers = eval_spec.scoring_plan()
    telemetry = SchedulerTelemetry(run_dir, scheduler=config.scheduler)
    if config.scheduler == "async":
        rollouts, per_rollout_scores, all_scores = await _execute_async_pipeline(
            config=config,
            executor=executor,
            environment=environment,
            model_slots=model_slots,
            jobs=jobs,
            scorers=scorers,
            run_id=run_id,
            run_dir=run_dir,
            timeout_s=timeout_s,
            telemetry=telemetry,
        )
    else:
        rollouts, per_rollout_scores, all_scores = await _execute_sync_pipeline(
            config=config,
            executor=executor,
            environment=environment,
            model_slots=model_slots,
            jobs=jobs,
            scorers=scorers,
            run_id=run_id,
            run_dir=run_dir,
            timeout_s=timeout_s,
            telemetry=telemetry,
        )

    rollouts_by_example: dict[str, list[RolloutResult]] = defaultdict(list)
    for job, ro in zip(jobs, rollouts, strict=True):
        rollouts_by_example[job.example.id].append(ro)

    meta_scorers = _meta_scoring_plan(eval_spec)
    if meta_scorers:
        # Group per_rollout_scores by example_id, preserving rollout order.
        scores_by_example: dict[str, list[list[RubricScore]]] = defaultdict(list)
        for job, rscores in zip(jobs, per_rollout_scores, strict=True):
            scores_by_example[job.example.id].append(rscores)
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
    telemetry.write_metrics(
        planned_rollouts=len(jobs),
        scored_rollouts=len(per_rollout_scores),
        wall_time_s=(completed_at - started_at).total_seconds(),
        rollout_workers=(
            config.rollout_workers if config.scheduler == "async" else config.concurrency
        ),
        scoring_workers=config.scoring_workers if config.scheduler == "async" else 1,
    )

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


async def _execute_sync_pipeline(
    *,
    config: RunConfig,
    executor: Executor,
    environment: Environment,
    model_slots: dict[str, str],
    jobs: list[RolloutJob],
    scorers: list[Scorer],
    run_id: str,
    run_dir: Path,
    timeout_s: int,
    telemetry: SchedulerTelemetry,
) -> tuple[list[RolloutResult], list[list[RubricScore]], list[RubricScore]]:
    semaphore = asyncio.Semaphore(config.concurrency)

    async def run_one(job: RolloutJob) -> RolloutResult:
        async with semaphore:
            telemetry.event("rollout_started", job=job)
            started = datetime.now()
            try:
                return await _submit_rollout(
                    executor=executor,
                    environment=environment,
                    model_slots=model_slots,
                    job=job,
                    timeout_s=timeout_s,
                )
            finally:
                telemetry.add_busy("rollout", (datetime.now() - started).total_seconds())
                telemetry.event("rollout_completed", job=job)

    rollouts = await asyncio.gather(*(run_one(job) for job in jobs))
    per_rollout_scores: list[list[RubricScore]] = []
    all_scores: list[RubricScore] = []
    with ScoreStreamWriter(run_dir) as stream:
        for job, rollout in zip(jobs, rollouts, strict=True):
            scores = await _score_rollout(
                job=job,
                rollout=rollout,
                scorers=scorers,
                run_id=run_id,
                publish=stream.publish,
                failure_root=run_dir / "failures",
            )
            per_rollout_scores.append(scores)
            all_scores.extend(scores)
    return rollouts, per_rollout_scores, all_scores


async def _execute_async_pipeline(
    *,
    config: RunConfig,
    executor: Executor,
    environment: Environment,
    model_slots: dict[str, str],
    jobs: list[RolloutJob],
    scorers: list[Scorer],
    run_id: str,
    run_dir: Path,
    timeout_s: int,
    telemetry: SchedulerTelemetry,
) -> tuple[list[RolloutResult], list[list[RubricScore]], list[RubricScore]]:
    rollout_workers = max(1, config.rollout_workers or config.concurrency)
    scoring_workers = max(1, config.scoring_workers)
    pending: asyncio.Queue[RolloutJob | None] = asyncio.Queue()
    completed: asyncio.Queue[RolloutEnvelope | None] = asyncio.Queue(
        maxsize=max(1, scoring_workers * 2)
    )
    rollouts_by_job: dict[tuple[int, int], RolloutResult] = {}
    scores_by_job: dict[tuple[int, int], list[RubricScore]] = {}
    all_scores: list[RubricScore] = []

    for job in jobs:
        telemetry.event("job_queued", job=job)
        await pending.put(job)
    for _ in range(rollout_workers):
        await pending.put(None)

    async def rollout_worker(worker_id: int) -> None:
        while True:
            job = await pending.get()
            try:
                if job is None:
                    return
                started = datetime.now()
                telemetry.event("rollout_started", job=job, worker_id=worker_id)
                try:
                    rollout = await _submit_rollout(
                        executor=executor,
                        environment=environment,
                        model_slots=model_slots,
                        job=job,
                        timeout_s=timeout_s,
                    )
                except Exception as exc:
                    now = datetime.now()
                    rollout = RolloutResult(
                        example_id=job.example.id,
                        target_name=environment.name,
                        target_version=environment.version,
                        status="error",
                        started_at=started,
                        completed_at=now,
                        duration_ms=int((now - started).total_seconds() * 1000),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                completed_at = datetime.now()
                telemetry.add_busy(
                    "rollout", (completed_at - started).total_seconds()
                )
                telemetry.event(
                    "rollout_completed",
                    job=job,
                    worker_id=worker_id,
                    status=rollout.status,
                )
                await completed.put(
                    RolloutEnvelope(
                        job=job,
                        rollout=rollout,
                        started_at=started,
                        completed_at=completed_at,
                        worker_id=worker_id,
                    )
                )
            finally:
                pending.task_done()

    async def scoring_worker(worker_id: int, publish: Callable[[RubricScore], None]) -> None:
        while True:
            envelope = await completed.get()
            try:
                if envelope is None:
                    return
                started = datetime.now()
                job = envelope.job
                telemetry.event("scoring_started", job=job, worker_id=worker_id)
                scores = await _score_rollout(
                    job=job,
                    rollout=envelope.rollout,
                    scorers=scorers,
                    run_id=run_id,
                    publish=publish,
                    failure_root=run_dir / "failures",
                )
                completed_at = datetime.now()
                telemetry.add_busy(
                    "scoring", (completed_at - started).total_seconds()
                )
                telemetry.event(
                    "scoring_completed",
                    job=job,
                    worker_id=worker_id,
                    scores=len(scores),
                )
                rollouts_by_job[(job.idx, job.rep)] = envelope.rollout
                scores_by_job[(job.idx, job.rep)] = scores
                all_scores.extend(scores)
            finally:
                completed.task_done()

    rollout_tasks = [
        asyncio.create_task(rollout_worker(i)) for i in range(rollout_workers)
    ]
    with ScoreStreamWriter(run_dir) as stream:
        scoring_tasks = [
            asyncio.create_task(scoring_worker(i, stream.publish))
            for i in range(scoring_workers)
        ]
        await pending.join()
        await asyncio.gather(*rollout_tasks)
        for _ in range(scoring_workers):
            await completed.put(None)
        await completed.join()
        await asyncio.gather(*scoring_tasks)

    rollouts = [rollouts_by_job[(job.idx, job.rep)] for job in jobs]
    per_rollout_scores = [scores_by_job[(job.idx, job.rep)] for job in jobs]
    return rollouts, per_rollout_scores, all_scores


async def _submit_rollout(
    *,
    executor: Executor,
    environment: Environment,
    model_slots: dict[str, str],
    job: RolloutJob,
    timeout_s: int,
) -> RolloutResult:
    return await executor.submit(
        target_name=environment.name,
        target_version=environment.version,
        model_slots=model_slots,
        example=job.example,
        run_dir=job.run_dir,
        timeout_s=timeout_s,
        rng_seed=job.rng_seed,
    )


async def _score_rollout(
    *,
    job: RolloutJob,
    rollout: RolloutResult,
    scorers: list[Scorer],
    run_id: str,
    publish: Callable[[RubricScore], None],
    failure_root: Path,
) -> list[RubricScore]:
    if rollout.status != "ok":
        failure_dir = failure_root / job.example.id
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "error.txt").write_text(rollout.error or "")
    rollout_scores: list[RubricScore] = []
    for scorer in scorers:
        try:
            if supports_publish(scorer):
                scores = await scorer.score(
                    job.example, rollout, run_id=run_id, publish=publish
                )
            else:
                scores = await scorer.score(job.example, rollout, run_id=run_id)
                for score in scores:
                    publish(score)
        except Exception as exc:
            crash = RubricScore(
                run_id=run_id,
                example_id=job.example.id,
                dimension=scorer.dimension,
                value=0.0,
                scorer="deterministic",
                rationale=f"scorer {scorer.name} crashed: {exc}",
            )
            publish(crash)
            scores = [crash]
        rollout_scores.extend(scores)
    return rollout_scores


class SchedulerTelemetry:
    def __init__(self, run_dir: Path, *, scheduler: str) -> None:
        self._run_dir = run_dir
        self._scheduler = scheduler
        self._started = time.monotonic()
        self._events_path = run_dir / "scheduler_events.jsonl"
        self._busy_seconds: dict[str, float] = defaultdict(float)

    def event(
        self,
        event: str,
        *,
        job: RolloutJob | None = None,
        worker_id: int | None = None,
        **extra: Any,
    ) -> None:
        row: dict[str, Any] = {
            "t": round(time.monotonic() - self._started, 6),
            "scheduler": self._scheduler,
            "event": event,
        }
        if worker_id is not None:
            row["worker_id"] = worker_id
        if job is not None:
            row.update(
                {
                    "example_id": job.example.id,
                    "idx": job.idx,
                    "rep": job.rep,
                }
            )
        row.update(extra)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def add_busy(self, kind: Literal["rollout", "scoring"], seconds: float) -> None:
        self._busy_seconds[kind] += max(0.0, seconds)

    def write_metrics(
        self,
        *,
        planned_rollouts: int,
        scored_rollouts: int,
        wall_time_s: float,
        rollout_workers: int,
        scoring_workers: int,
    ) -> None:
        rollout_den = max(0.001, rollout_workers * wall_time_s)
        scoring_den = max(0.001, scoring_workers * wall_time_s)
        metrics = {
            "scheduler": self._scheduler,
            "planned_rollouts": planned_rollouts,
            "scored_rollouts": scored_rollouts,
            "rollout_workers": rollout_workers,
            "scoring_workers": scoring_workers,
            "wall_time_s": wall_time_s,
            "rollout_worker_utilization": self._busy_seconds["rollout"] / rollout_den,
            "scoring_worker_utilization": self._busy_seconds["scoring"] / scoring_den,
            "rollout_busy_s": self._busy_seconds["rollout"],
            "scoring_busy_s": self._busy_seconds["scoring"],
            "peak_rss_mb": _peak_rss_mb(),
        }
        (self._run_dir / "pipeline_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        (self._run_dir / "utilization.jsonl").write_text(json.dumps(metrics) + "\n")


def _peak_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB. The project currently runs on
    # macOS locally, but keep the conversion sensible in CI.
    if rss > 10_000_000:
        return rss / (1024 * 1024)
    return rss / 1024


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
