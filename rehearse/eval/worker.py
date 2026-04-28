"""Subprocess worker entry point.

Reads one JSON request from stdin, runs one Target rollout, writes one
RolloutResult JSON to stdout. Stderr is for human-readable logs.

Request schema:
    {
      "target_name": str,
      "model_slots": dict[str, str],
      "example": BenchmarkExample.model_dump(),
      "run_dir": str,
      "seed": int,
    }
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.targets import get_target


async def _run() -> int:
    raw = sys.stdin.read()
    req = json.loads(raw)
    example = BenchmarkExample.model_validate(req["example"])
    target = get_target(req["target_name"], req["model_slots"])
    run_dir = Path(req["run_dir"])
    seed = int(req["seed"])

    started = datetime.now()
    try:
        result = await target.rollout(example, run_dir, seed)
    except Exception as exc:
        completed = datetime.now()
        result = RolloutResult(
            example_id=example.id,
            target_name=target.name,
            target_version=target.version,
            status="error",
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    sys.stdout.write(result.model_dump_json())
    sys.stdout.flush()
    return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
