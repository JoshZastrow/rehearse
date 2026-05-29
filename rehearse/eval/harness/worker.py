"""Subprocess worker entry point.

Reads one JSON request from stdin, runs one Environment rollout, writes one
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
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from rehearse.eval.environments import get_environment
from rehearse.eval.protocols import BenchmarkExample, RolloutResult


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )


async def _run() -> int:
    _configure_logging()
    raw = sys.stdin.read()
    req = json.loads(raw)
    example = BenchmarkExample.model_validate(req["example"])
    environment = get_environment(req["target_name"], req["model_slots"])
    run_dir = Path(req["run_dir"])
    seed = int(req["seed"])

    started = datetime.now()
    try:
        result = await environment.rollout(example, run_dir, seed)
    except Exception as exc:
        completed = datetime.now()
        result = RolloutResult(
            example_id=example.id,
            target_name=environment.name,
            target_version=environment.version,
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
