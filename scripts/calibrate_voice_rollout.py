"""Drive the 2-at-a-time calibration loop for `voice-rollout-judges`.

Walks the scenario manifest from start to N, running the eval with
`--limit` increasing by 2 each step. After each step prints
the aggregate `weighted_reward` and the deviation from the 0.5 target.
At each checkpoint (default every 10 steps) it also dumps the per-dim
means so drift can be observed.

Usage:
    set -a && source .env && set +a
    uv run python scripts/calibrate_voice_rollout.py \
        --start 10 --stop 100 --step 2 --checkpoint-every 5

Each call shells out to `rehearse-eval run`. Cumulative cost grows
linearly with the highest --limit reached, so prefer larger steps + a
final full run over many tiny incremental ones.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--stop", type=int, default=100)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--runs-root", default="evals/runs")
    ap.add_argument("--log", type=Path, default=Path("evals/runs/calibration_log.jsonl"))
    args = ap.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.log.open("a")

    step_idx = 0
    for limit in range(args.start, args.stop + 1, args.step):
        step_idx += 1
        print(f"\n=== step {step_idx}: limit={limit} ===", flush=True)
        result = subprocess.run(
            [
                "uv",
                "run",
                "rehearse-eval",
                "run",
                "--eval",
                "voice-rollout-judges",
                "--limit",
                str(limit),
                "--concurrency",
                str(args.concurrency),
                "--runs-root",
                args.runs_root,
                "--tag",
                f"calibrate-step-{step_idx:03d}-limit-{limit:03d}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print(f"step failed at limit={limit}", file=sys.stderr)
            return 1

        run_id = _parse_run_id(result.stdout)
        agg = _aggregate(Path(args.runs_root) / run_id / "results.jsonl")
        weighted = agg.get("weighted_reward")
        deviation = (weighted - args.target) if weighted is not None else None
        record = {
            "step": step_idx,
            "limit": limit,
            "run_id": run_id,
            "aggregate": agg,
            "deviation_from_target": deviation,
        }
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        if weighted is not None:
            print(
                f"  weighted_reward={weighted:.3f} "
                f"(Δ {deviation:+.3f} from {args.target})"
            )
        else:
            print("  weighted_reward=<missing>")

        if step_idx % args.checkpoint_every == 0:
            print("  per-dim:")
            for dim, mean in sorted(agg.items()):
                print(f"    {dim:<38} {mean:.3f}")

    log_handle.close()
    return 0


def _parse_run_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("run_id:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"could not find run_id in: {stdout!r}")


def _aggregate(results_path: Path) -> dict[str, float]:
    by_dim: dict[str, list[float]] = defaultdict(list)
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by_dim[row["dimension"]].append(float(row["value"]))
    return {dim: statistics.fmean(vs) for dim, vs in by_dim.items() if vs}


if __name__ == "__main__":
    raise SystemExit(main())
