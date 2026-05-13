"""LocalSubprocessExecutor — one subprocess per rollout.

Crash isolation, clean memory between rollouts, and a clear upgrade path to
container/Modal executors that share the same protocol.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult


def _last_json_line(stdout_b: bytes) -> bytes:
    """Return the last line of stdout that begins with '{'.

    The worker's contract is that the JSON result is the final write to stdout.
    Third-party libraries (e.g. Hume SDK via structlog) may emit log lines to
    stdout before it, so we scan backwards rather than parsing the full buffer.
    """
    for line in reversed(stdout_b.splitlines()):
        stripped = line.strip()
        if stripped.startswith(b"{"):
            return stripped
    return stdout_b  # fallback: let the caller's except block handle the parse error


class LocalSubprocessExecutor:
    """Spawn `python -m rehearse.eval.worker` per rollout. Hard timeout."""

    async def submit(
        self,
        target_name: str,
        target_version: str,
        model_slots: dict[str, str],
        example: BenchmarkExample,
        run_dir: Path,
        timeout_s: int,
        rng_seed: int,
    ) -> RolloutResult:
        request = {
            "target_name": target_name,
            "model_slots": model_slots,
            "example": example.model_dump(mode="json"),
            "run_dir": str(run_dir),
            "seed": rng_seed,
        }
        started = datetime.now()
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "rehearse.eval.worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit parent stderr so progress prints are visible
        )
        try:
            stdout_b, _ = await asyncio.wait_for(
                proc.communicate(json.dumps(request).encode()),
                timeout=timeout_s,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            completed = datetime.now()
            return RolloutResult(
                example_id=example.id,
                target_name=target_name,
                target_version=target_version,
                status="timeout",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                error=f"rollout exceeded {timeout_s}s",
            )

        completed = datetime.now()
        if proc.returncode != 0:
            return RolloutResult(
                example_id=example.id,
                target_name=target_name,
                target_version=target_version,
                status="error",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                error=f"worker exit {proc.returncode}",
            )

        # The worker writes one JSON line to stdout as the last thing it does.
        # Libraries (e.g. Hume SDK structlog) may write log lines to stdout
        # before it. Scan backwards for the last line that begins with '{'.
        json_b = _last_json_line(stdout_b)
        try:
            result = RolloutResult.model_validate_json(json_b)
        except Exception as exc:
            return RolloutResult(
                example_id=example.id,
                target_name=target_name,
                target_version=target_version,
                status="error",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                error=f"could not parse worker stdout: {exc}\n{stdout_b!r}",
            )
        return result
