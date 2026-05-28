"""rehearse-eval — eval harness CLI.

Subcommands:
  list-evals          print registered eval names
  list-datasets       print registered dataset names
  list-environments   print registered environment names
  list-runs           list recent runs with per-rollout scores and audio paths
  run                 execute an eval against an environment
  show                print summary.md for a run_id
  watch               tail scores.jsonl and render a live aggregate table
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rehearse.eval.benchmarks import list_benchmarks
from rehearse.eval.datasets import list_datasets
from rehearse.eval.environments import list_environments
from rehearse.eval.suites import list_evals
from rehearse.eval.harness.executor import InProcessExecutor
from rehearse.eval.judges import list_providers
from rehearse.eval.harness.report import ensure_run_recorded, list_runs, record_run, render_report
from rehearse.eval.runner import RunConfig, _resolve_eval, execute_run
from rehearse.eval.transports import TransportEvent
from rehearse.eval.harness.watch import watch as run_watch


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
    run.add_argument(
        "--caller",
        default=None,
        metavar="MODEL",
        help=(
            "model name for the synthetic caller "
            "(default: claude-sonnet-4-6); "
            "e.g. claude-sonnet-4-6, gemma-4-26b, kyutai/moshi"
        ),
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--concurrency", type=int, default=4)
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

    lr = sub.add_parser("list-runs", help="list recent runs and their per-rollout scores and audio paths")
    lr.add_argument("--n", type=int, default=10, help="number of runs to show (default 10)")
    lr.add_argument("--eval", dest="eval_filter", default=None, help="filter by eval name substring")
    lr.add_argument("--scenario", dest="scenario_filter", default=None, help="filter rollouts by scenario id substring")
    lr.add_argument("--play", dest="play_session", default=None, metavar="SESSION_ID",
                    help="open the most recent audio.wav matching SESSION_ID")
    lr.add_argument("--runs-root", default="evals/runs", type=Path)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd in {"list-evals", "list-benchmarks"}:
        for name in sorted(list_evals() + list_benchmarks()):
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

    if args.cmd == "list-runs":
        list_runs(
            args.runs_root,
            n=args.n,
            eval_filter=args.eval_filter,
            scenario_filter=args.scenario_filter,
            play_session=args.play_session,
        )
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

    if args.cmd == "run":
        eval_spec = _resolve_eval(args.eval_name)
        environment = args.environment or eval_spec.preferred_environment
        model_slots = dict(args.model_slot)
        if args.provider:
            model_slots["provider"] = args.provider
        if args.caller:
            model_slots["caller"] = args.caller

        if args.dry_run:
            n_examples = len(list(eval_spec.load()))
            if args.limit is not None:
                n_examples = min(n_examples, args.limit)
            print(f"eval: {eval_spec.name}@{eval_spec.version}")
            print(f"dataset: {eval_spec.dataset.name}@{eval_spec.dataset.version}")
            print(f"environment: {environment}")
            print(f"examples: {n_examples}")
            print(f"concurrency: {args.concurrency}")
            print(f"model_slots: {model_slots}")
            return 0

        config = RunConfig(
            eval_name=args.eval_name,
            environment=environment,
            limit=args.limit,
            concurrency=args.concurrency,
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


if __name__ == "__main__":
    sys.exit(main())
