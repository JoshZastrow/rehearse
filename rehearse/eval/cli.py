"""rehearse-eval — eval harness CLI.

Subcommands:
  list-evals          print registered eval names
  list-datasets       print registered dataset names
  list-environments   print registered environment names
  run                 execute an eval against an environment
  show               print summary.md for a run_id
  watch              tail scores.jsonl and render a live aggregate table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from rehearse.eval.datasets import list_datasets
from rehearse.eval.environments import list_environments
from rehearse.eval.evals import get_eval, list_evals
from rehearse.eval.executors import InProcessExecutor
from rehearse.eval.providers import list_providers
from rehearse.eval.report import ensure_run_recorded, record_run, render_report
from rehearse.eval.runner import RunConfig, execute_run
from rehearse.eval.transports import TransportEvent
from rehearse.eval.watch import watch as run_watch


def _print_event(event: TransportEvent) -> None:
    if event.kind == "text":
        text = str(event.payload.get("text", ""))
        print(f"[{event.source} text] {text}", flush=True)
    elif event.kind == "control":
        marker = event.payload.get("event", "")
        print(f"[{event.source} control] {marker}", flush=True)
    else:
        print(f"[{event.source} {event.kind}] {event.payload}", flush=True)


def _parse_model_slot(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--model-slot must be key=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rehearse-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-evals", help="list registered evals")
    sub.add_parser("list-datasets", help="list registered datasets")
    sub.add_parser("list-environments", help="list registered environments")
    sub.add_parser("list-providers", help="list registered audio LLM providers")
    sub.add_parser("list-benchmarks", help="deprecated alias for list-evals")
    sub.add_parser("list-targets", help="deprecated alias for list-environments")

    run = sub.add_parser("run", help="run an eval against an environment")
    eval_group = run.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("--eval", dest="eval_name")
    eval_group.add_argument("--benchmark", dest="eval_name")
    run.add_argument(
        "--environment",
        default=None,
        help="defaults to eval.preferred_environment",
    )
    run.add_argument("--target", dest="environment", default=None)
    run.add_argument(
        "--provider",
        choices=["gemini", "vllm"],
        default=None,
        help="shortcut for --model-slot provider=... when environment=multimodal-llm",
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument(
        "--scheduler",
        choices=["sync", "async"],
        default="sync",
        help="sync waits for all rollouts before scoring; async scores rollouts as they finish",
    )
    run.add_argument(
        "--rollout-workers",
        type=int,
        default=None,
        help="number of concurrent rollout workers; defaults to --concurrency",
    )
    run.add_argument(
        "--scoring-workers",
        type=int,
        default=1,
        help="number of concurrent scoring workers for --scheduler async",
    )
    run.add_argument("--report-interval-s", type=float, default=10.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help=(
            "run each example N times with distinct rng_seeds; meta-scorers "
            "(e.g. stability) consume the grouped rollouts"
        ),
    )
    run.add_argument(
        "--model-slot",
        action="append",
        default=[],
        type=_parse_model_slot,
        help="repeat to override model slots, e.g. --model-slot feedback=claude-opus-4-7",
    )
    run.add_argument("--tag", default=None)
    run.add_argument("--runs-root", default="evals/runs", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "stream transport events to stdout as the rollout runs. "
            "Forces in-process execution (no subprocess isolation)."
        ),
    )

    show = sub.add_parser("show", help="print summary.md for a run_id")
    show.add_argument("run_id")
    show.add_argument("--runs-root", default="evals/runs", type=Path)

    watch = sub.add_parser(
        "watch",
        help="tail scores.jsonl for a run and render a live aggregate table",
    )
    watch_target = watch.add_mutually_exclusive_group(required=True)
    watch_target.add_argument(
        "--run-id",
        help="resolve <runs_root>/<run_id> as the run dir",
    )
    watch_target.add_argument(
        "--run-dir",
        type=Path,
        help="explicit run directory containing scores.jsonl",
    )
    watch.add_argument("--runs-root", default="evals/runs", type=Path)

    bench = sub.add_parser(
        "benchmark-scheduler",
        help="compare sync vs async scheduler wall time and utilization",
    )
    bench_eval = bench.add_mutually_exclusive_group(required=True)
    bench_eval.add_argument("--eval", dest="eval_name")
    bench_eval.add_argument("--benchmark", dest="eval_name")
    bench.add_argument("--environment", default=None)
    bench.add_argument("--limit", type=int, default=10)
    bench.add_argument("--runs-root", default="evals/runs", type=Path)
    bench.add_argument("--sync-concurrency", type=int, default=10)
    bench.add_argument("--async-rollout-workers", type=int, default=8)
    bench.add_argument("--async-scoring-workers", type=int, default=2)
    bench.add_argument(
        "--model-slot",
        action="append",
        default=[],
        type=_parse_model_slot,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd in {"list-evals", "list-benchmarks"}:
        for name in list_evals():
            print(name)
        return 0

    if args.cmd == "list-datasets":
        for name in list_datasets():
            print(name)
        return 0

    if args.cmd in {"list-environments", "list-targets"}:
        for name in list_environments():
            print(name)
        return 0

    if args.cmd == "list-providers":
        for name in list_providers():
            print(name)
        return 0

    if args.cmd == "show":
        run_dir = args.runs_root / args.run_id
        if not run_dir.exists():
            print(f"no run at {run_dir}", file=sys.stderr)
            return 1
        ensure_run_recorded(args.run_id, args.runs_root)
        render_report(args.runs_root, highlight_run_id=args.run_id)
        return 0

    if args.cmd == "watch":
        run_dir = args.run_dir or (args.runs_root / args.run_id)
        return run_watch(run_dir)

    if args.cmd == "benchmark-scheduler":
        return _benchmark_scheduler(args)

    if args.cmd == "run":
        eval_spec = get_eval(args.eval_name)
        environment = args.environment or eval_spec.preferred_environment
        model_slots = dict(args.model_slot)
        if args.provider:
            model_slots["provider"] = args.provider

        if args.dry_run:
            n_examples = len(list(eval_spec.load()))
            if args.limit is not None:
                n_examples = min(n_examples, args.limit)
            print(f"eval: {eval_spec.name}@{eval_spec.version}")
            print(f"dataset: {eval_spec.dataset.name}@{eval_spec.dataset.version}")
            print(f"environment: {environment}")
            print(f"examples: {n_examples}")
            print(f"concurrency: {args.concurrency}")
            print(f"scheduler: {args.scheduler}")
            print(f"rollout_workers: {args.rollout_workers or args.concurrency}")
            print(f"scoring_workers: {args.scoring_workers}")
            print(f"model_slots: {model_slots}")
            return 0

        config = RunConfig(
            eval_name=args.eval_name,
            environment=environment,
            limit=args.limit,
            concurrency=args.concurrency,
            scheduler=args.scheduler,
            rollout_workers=args.rollout_workers,
            scoring_workers=args.scoring_workers,
            report_interval_s=args.report_interval_s,
            seed=args.seed,
            model_slots=model_slots,
            tag=args.tag,
            runs_root=args.runs_root,
            repetitions=args.repetitions,
        )
        executor = InProcessExecutor(on_event=_print_event) if args.verbose else None
        outcome = asyncio.run(execute_run(config, executor=executor))
        record_run(
            run_id=outcome.run_id,
            eval_name=outcome.eval_name,
            environment=outcome.environment,
            run_date=outcome.started_at,
            n_examples=outcome.n_examples,
            duration_s=outcome.duration_s,
            total_tokens=outcome.total_tokens,
            scores=outcome.aggregate_scores,
            runs_root=config.runs_root,
        )
        render_report(config.runs_root, highlight_run_id=outcome.run_id)
        return 0

    raise AssertionError(f"unhandled cmd: {args.cmd}")


def _benchmark_scheduler(args: argparse.Namespace) -> int:
    eval_spec = get_eval(args.eval_name)
    environment = args.environment or eval_spec.preferred_environment
    model_slots = dict(args.model_slot)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    bench_dir = args.runs_root / f"scheduler-benchmark-{stamp}"
    sync_root = bench_dir / "sync"
    async_root = bench_dir / "async"

    sync_config = RunConfig(
        eval_name=args.eval_name,
        environment=environment,
        limit=args.limit,
        concurrency=args.sync_concurrency,
        scheduler="sync",
        model_slots=model_slots,
        runs_root=sync_root,
    )
    async_config = RunConfig(
        eval_name=args.eval_name,
        environment=environment,
        limit=args.limit,
        concurrency=args.async_rollout_workers,
        scheduler="async",
        rollout_workers=args.async_rollout_workers,
        scoring_workers=args.async_scoring_workers,
        model_slots=model_slots,
        runs_root=async_root,
    )
    sync_outcome = asyncio.run(execute_run(sync_config))
    async_outcome = asyncio.run(execute_run(async_config))
    sync_metrics = _read_metrics(sync_outcome.run_dir)
    async_metrics = _read_metrics(async_outcome.run_dir)
    _write_benchmark_artifacts(
        bench_dir=bench_dir,
        sync_metrics=sync_metrics,
        async_metrics=async_metrics,
    )
    _print_benchmark_chart(sync_metrics, async_metrics)
    print(f"\nBenchmark artifacts: {bench_dir}")
    return 0


def _read_metrics(run_dir: Path) -> dict[str, float | int | str]:
    return json.loads((run_dir / "pipeline_metrics.json").read_text())


def _write_benchmark_artifacts(
    *,
    bench_dir: Path,
    sync_metrics: dict[str, float | int | str],
    async_metrics: dict[str, float | int | str],
) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ("wall_time_s", sync_metrics["wall_time_s"], async_metrics["wall_time_s"]),
        ("peak_rss_mb", sync_metrics["peak_rss_mb"], async_metrics["peak_rss_mb"]),
        (
            "rollout_worker_utilization",
            sync_metrics["rollout_worker_utilization"],
            async_metrics["rollout_worker_utilization"],
        ),
        (
            "scoring_worker_utilization",
            sync_metrics["scoring_worker_utilization"],
            async_metrics["scoring_worker_utilization"],
        ),
    ]
    csv_lines = ["metric,sync,async"]
    csv_lines.extend(f"{name},{sync},{async_}" for name, sync, async_ in rows)
    (bench_dir / "pipeline_speedup.csv").write_text("\n".join(csv_lines) + "\n")
    (bench_dir / "pipeline_speedup.svg").write_text(_speedup_svg(rows))


def _print_benchmark_chart(
    sync_metrics: dict[str, float | int | str],
    async_metrics: dict[str, float | int | str],
) -> None:
    sync_wall = float(sync_metrics["wall_time_s"])
    async_wall = float(async_metrics["wall_time_s"])
    speedup = sync_wall / async_wall if async_wall else 0.0
    print("\nScheduler Benchmark\n")
    _print_bar("sync wall", sync_wall, max(sync_wall, async_wall), suffix="s")
    _print_bar(
        "async wall",
        async_wall,
        max(sync_wall, async_wall),
        suffix=f"s  {speedup:.2f}x faster",
    )
    sync_rss = float(sync_metrics["peak_rss_mb"])
    async_rss = float(async_metrics["peak_rss_mb"])
    max_rss = max(sync_rss, async_rss)
    print("")
    _print_bar("sync rss", sync_rss, max_rss, suffix="MB")
    _print_bar("async rss", async_rss, max_rss, suffix="MB")


def _print_bar(label: str, value: float, max_value: float, *, suffix: str) -> None:
    width = 40
    n = int(width * value / max(max_value, 0.001))
    print(f"{label:<11} {value:>7.1f} |{'#' * n:<40}| {suffix}")


def _speedup_svg(rows: list[tuple[str, object, object]]) -> str:
    width = 760
    row_h = 52
    height = 60 + len(rows) * row_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="monospace" font-size="18">Scheduler benchmark</text>',
    ]
    for i, (name, sync, async_) in enumerate(rows):
        y = 60 + i * row_h
        sync_f = float(sync)
        async_f = float(async_)
        max_v = max(sync_f, async_f, 0.001)
        sync_w = int(360 * sync_f / max_v)
        async_w = int(360 * async_f / max_v)
        parts.extend(
            [
                f'<text x="20" y="{y}" font-family="monospace" font-size="12">{name}</text>',
                f'<rect x="220" y="{y - 14}" width="{sync_w}" height="14" fill="#8884d8"/>',
                f'<rect x="220" y="{y + 6}" width="{async_w}" height="14" fill="#82ca9d"/>',
                (
                    f'<text x="590" y="{y - 3}" font-family="monospace" '
                    f'font-size="12">sync {sync_f:.2f}</text>'
                ),
                (
                    f'<text x="590" y="{y + 17}" font-family="monospace" '
                    f'font-size="12">async {async_f:.2f}</text>'
                ),
            ]
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
