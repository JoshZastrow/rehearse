"""rehearse-eval — eval harness CLI.

Subcommands:
  list-evals          print registered eval names
  list-datasets       print registered dataset names
  list-environments   print registered environment names
  run                 execute an eval against an environment
  show               print summary.md for a run_id
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rehearse.eval.datasets import list_datasets
from rehearse.eval.environments import list_environments
from rehearse.eval.evals import get_eval, list_evals
from rehearse.eval.providers import list_providers
from rehearse.eval.runner import RunConfig, execute_run


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
        path = args.runs_root / args.run_id / "summary.md"
        if not path.exists():
            print(f"no summary at {path}", file=sys.stderr)
            return 1
        print(path.read_text())
        return 0

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
