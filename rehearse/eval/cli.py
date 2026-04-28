"""rehearse-eval — eval harness CLI.

Subcommands:
  list-benchmarks    print registered benchmark names
  list-targets       print registered target names
  run                execute a benchmark against a target
  show               print summary.md for a run_id
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rehearse.eval.benchmarks import get_benchmark, list_benchmarks
from rehearse.eval.runner import RunConfig, execute_run
from rehearse.eval.targets import list_targets


def _parse_model_slot(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--model-slot must be key=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rehearse-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-benchmarks", help="list registered benchmarks")
    sub.add_parser("list-targets", help="list registered targets")

    run = sub.add_parser("run", help="run a benchmark against a target")
    run.add_argument("--benchmark", required=True)
    run.add_argument("--target", default=None, help="defaults to benchmark.preferred_target")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--seed", type=int, default=0)
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

    show = sub.add_parser("show", help="print summary.md for a run_id")
    show.add_argument("run_id")
    show.add_argument("--runs-root", default="evals/runs", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "list-benchmarks":
        for name in list_benchmarks():
            print(name)
        return 0

    if args.cmd == "list-targets":
        for name in list_targets():
            print(name)
        return 0

    if args.cmd == "show":
        path = args.runs_root / args.run_id / "summary.md"
        if not path.exists():
            print(f"no summary at {path}", file=sys.stderr)
            return 1
        print(path.read_text())
        return 0

    if args.cmd == "run":
        bench = get_benchmark(args.benchmark)
        target = args.target or bench.preferred_target
        model_slots = dict(args.model_slot)

        if args.dry_run:
            n_examples = len(list(bench.load()))
            if args.limit is not None:
                n_examples = min(n_examples, args.limit)
            print(f"benchmark: {bench.name}@{bench.version}")
            print(f"target: {target}")
            print(f"examples: {n_examples}")
            print(f"concurrency: {args.concurrency}")
            print(f"model_slots: {model_slots}")
            return 0

        config = RunConfig(
            benchmark=args.benchmark,
            target=target,
            limit=args.limit,
            concurrency=args.concurrency,
            seed=args.seed,
            model_slots=model_slots,
            tag=args.tag,
            runs_root=args.runs_root,
        )
        outcome = asyncio.run(execute_run(config))
        print(f"run_id: {outcome.run_id}")
        print(f"run_dir: {outcome.run_dir}")
        print(
            f"examples: {outcome.n_examples} "
            f"(ok={outcome.n_ok} error={outcome.n_error} timeout={outcome.n_timeout})"
        )
        for dim, mean in sorted(outcome.aggregate_scores.items()):
            print(f"  {dim}: {mean:.3f}")
        return 0

    raise AssertionError(f"unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
